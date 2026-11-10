"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _simulate_aco(prizes, distance_matrix, max_len, heuristic, n_ants=20, n_iter=30, seed=42):
    """Simulate ACO with the given heuristic and return the best total prize found."""
    n = len(prizes)
    rng = np.random.default_rng(seed)
    
    # Pheromone matrix starts at 1
    pheromone = np.ones((n, n), dtype=np.float64)
    
    # Precompute valid moves: for each i, which j are reachable such that
    # path_so_far + d[i,j] + d[j,0] <= max_len
    # This depends on path_so_far, so we compute on the fly
    
    best_prize_overall = 0.0
    
    for iteration in range(n_iter):
        best_prize_this_iter = 0.0
        best_ant_prize = 0.0
        
        for ant in range(n_ants):
            # Each ant starts at node 0
            current = 0
            path_length = 0.0
            total_prize = 0.0
            visited = set([0])
            
            while True:
                # Find all valid next moves
                valid_moves = []
                probs = []
                
                for j in range(1, n):
                    if j in visited:
                        continue
                    
                    d_ij = distance_matrix[current, j]
                    d_j0 = distance_matrix[j, 0]
                    
                    if path_length + d_ij + d_j0 <= max_len:
                        valid_moves.append(j)
                        # Probability proportional to pheromone[i,j] * heuristic[i,j]
                        p = pheromone[current, j] * heuristic[current, j]
                        probs.append(p)
                
                if not valid_moves:
                    break
                
                # Normalize probabilities
                total_p = sum(probs)
                if total_p <= 0:
                    break
                
                # Select next node
                cum = np.cumsum(probs) / total_p
                r = rng.random()
                idx = np.searchsorted(cum, r)
                if idx >= len(valid_moves):
                    idx = len(valid_moves) - 1
                
                next_node = valid_moves[idx]
                d_next = distance_matrix[current, next_node]
                
                path_length += d_next
                total_prize += prizes[next_node]
                visited.add(next_node)
                current = next_node
            
        best_prize_this_iter = max(best_prize_this_iter, total_prize)
        
        best_prize_overall = max(best_prize_overall, best_prize_this_iter)
        
        # Update pheromone
        pheromone *= 0.9
        pheromone += best_prize_this_iter / 10.0
    
    return best_prize_overall


def _make_heuristic_v2(prizes, distance_matrix, max_len, 
                         alpha_prize=1.0, alpha_density=1.0, 
                         alpha_eigen=0.5, alpha_betweenness=0.3,
                         budget_gamma=1.5, dist_decay=1.0):
    """
    Build heuristic using a combination of:
    1. Prize-to-cost ratio
    2. Local prize density (neighborhood value)
    3. Prize-weighted eigenvector centrality
    4. Budget-aware penalty
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    n = len(prizes)
    
    heuristic = np.full((n, n), 1e-9, dtype=np.float64)
    
    # Compute prize-weighted eigenvector centrality
    # Use the adjacency-like matrix based on proximity
    # A[i,j] = exp(-distance[i,j]) * prizes[j]
    adj = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i != j:
                adj[i, j] = np.exp(-distance_matrix[i, j]) * prizes[j]
    
    # Power iteration for eigenvector centrality
    eigvec = np.ones(n) / n
    for _ in range(50):
        new_eigvec = adj @ eigvec
        norm = np.linalg.norm(new_eigvec)
        if norm > 0:
            new_eigvec /= norm
        else:
            break
        if np.linalg.norm(new_eigvec - eigvec) < 1e-10:
            break
        eigvec = new_eigvec
    
    # Compute local prize density for each node
    # density[j] = sum of prizes of neighbors within some radius
    radius = max_len * 0.5  # Use half the budget as radius
    density = np.zeros(n, dtype=np.float64)
    for j in range(n):
        for k in range(n):
            if k != j and distance_matrix[j, k] < radius:
                density[j] += prizes[k]
    
    # Compute betweenness-like centrality (simplified: count of shortest paths through j)
    # For simplicity, use degree centrality weighted by prize
    degree_centrality = np.zeros(n, dtype=np.float64)
    for j in range(n):
        count = 0
        for k in range(n):
            if k != j and distance_matrix[j, k] < max_len:
                count += 1
        degree_centrality[j] = count * prizes[j]
    
    # Normalize centrality measures
    max_eig = np.max(eigvec) if np.max(eigvec) > 0 else 1.0
    max_density = np.max(density) if np.max(density) > 0 else 1.0
    max_degree = np.max(degree_centrality) if np.max(degree_centrality) > 0 else 1.0
    
    eigvec_norm = eigvec / max_eig
    density_norm = density / max_density
    degree_norm = degree_centrality / max_degree
    
    # Build the heuristic matrix
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            
            cost = distance_matrix[i, j] + distance_matrix[j, 0]
            
            if cost > max_len:
                continue
            
            # Prize-to-cost ratio
            prize_cost = prizes[j] / (cost + 1e-9)
            
            # Budget penalty
            budget_fraction = cost / max_len
            budget_penalty = (1.0 - budget_fraction) ** budget_gamma
            
            # Distance decay
            dist_decay_factor = np.exp(-dist_decay * distance_matrix[i, j])
            
            # Combine centrality features
            centrality_score = (
                alpha_eigen * eigvec_norm[j] +
                alpha_betweenness * degree_norm[j]
            )
            
            # Local density bonus
            density_bonus = alpha_density * density_norm[j]
            
            # Final heuristic
            h = (prize_cost * budget_penalty * dist_decay_factor *
                 (1.0 + centrality_score + density_bonus))
            
            heuristic[i, j] = h
    
    # Normalize
    max_val = np.max(heuristic)
    if max_val > 0:
        heuristic = heuristic / max_val
    
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic


def _optimize_parameters(prizes, distance_matrix, max_len, deadline=280.0):
    """
    Optimize heuristic parameters by evaluating ACO performance.
    Uses a simple grid search + local refinement with time budget.
    """
    start_time = time.time()
    
    # Define parameter space
    alpha_prize_range = [0.5, 1.0, 1.5, 2.0]
    alpha_density_range = [0.0, 0.5, 1.0, 1.5]
    alpha_eigen_range = [0.0, 0.3, 0.5, 0.8]
    alpha_betweenness_range = [0.0, 0.2, 0.4]
    budget_gamma_range = [0.5, 1.0, 1.5, 2.0]
    dist_decay_range = [0.5, 1.0, 1.5]
    
    best_score = 0.0
    best_params = {
        'alpha_prize': 1.0,
        'alpha_density': 1.0,
        'alpha_eigen': 0.5,
        'alpha_betweenness': 0.3,
        'budget_gamma': 1.5,
        'dist_decay': 1.0
    }
    
    # Coarse grid search
    for ap in alpha_prize_range:
        for ad in alpha_density_range:
            for ae in alpha_eigen_range:
                for ab in alpha_betweenness_range:
                    for bg in budget_gamma_range:
                        for dd in dist_decay_range:
                            if time.time() - start_time > deadline - 60:
                                break
                            
                            params = {
                                'alpha_prize': ap,
                                'alpha_density': ad,
                                'alpha_eigen': ae,
                                'alpha_betweenness': ab,
                                'budget_gamma': bg,
                                'dist_decay': dd
                            }
                            
                            heuristic = _make_heuristic_v2(
                                prizes, distance_matrix, max_len, **params
                            )
                            
                            score = _simulate_aco(
                                prizes, distance_matrix, max_len, heuristic,
                                n_ants=20, n_iter=30, seed=42
                            )
                            
                            if score > best_score:
                                best_score = score
                                best_params = params.copy()
    
    return best_params, best_score


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute edge-attractiveness matrix using a learned combination of
    graph-theoretic centrality features and constraint-aware prize metrics.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    n = len(prizes)
    
    start_time = time.time()
    deadline = 280.0  # Leave margin before 330s kill
    
    # Try to optimize parameters within time budget
    try:
        best_params, best_score = _optimize_parameters(
            prizes, distance_matrix, max_len, deadline=deadline
        )
    except Exception:
        best_params = {
            'alpha_prize': 1.0,
            'alpha_density': 1.0,
            'alpha_eigen': 0.5,
            'alpha_betweenness': 0.3,
            'budget_gamma': 1.5,
            'dist_decay': 1.0
        }
    
    # Build final heuristic with best parameters
    heuristic = _make_heuristic_v2(
        prizes, distance_matrix, max_len, **best_params
    )
    
    # Ensure valid output
    if heuristic.shape != (n, n):
        heuristic = np.full((n, n), 1e-9, dtype=np.float64)
    
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
