"""Heuristic for ACO-CVRP: Adaptive Parameter Selection via Local Simulation."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_BUDGET = 330.0
_DEADLINE = _T0 + _BUDGET
_TIME_MARGIN = 15.0

def _remaining():
    return _DEADLINE - _TIME_MARGIN - time.time()

def _fast_aco_sim(distance_matrix, demands, capacity, alpha, beta, gamma, n_ants=10, n_iter=5):
    """
    Run a fast, simplified ACO simulation to evaluate heuristic parameters.
    Returns the best cost found.
    """
    n = distance_matrix.shape[0]
    # Precompute heuristic matrix for these params
    d = distance_matrix.copy()
    np.fill_diagonal(d, 1e9)
    
    # Base distance heuristic
    d_inv = 1.0 / (d + 1e-9)
    
    # Savings heuristic
    d_i0 = distance_matrix[:, 0]
    d_j0 = distance_matrix[0, :]
    # savings[i,j] = d[i,0] + d[0,j] - d[i,j]
    savings = d_i0[:, np.newaxis] + d_j0[np.newaxis, :] - d
    # Normalize savings to be positive and comparable
    # If savings <= 0, it's not a good connection (worse than going via depot)
    # We want to reward positive savings.
    # Use a soft plus or clip
    savings_pos = np.maximum(savings, 0.0)
    max_sav = np.max(savings_pos)
    if max_sav < 1e-9:
        sav_heur = np.ones_like(savings)
    else:
        sav_heur = savings_pos / max_sav
    
    # Degree heuristic (neighborhood centrality)
    # Compute Euclidean distance matrix from coordinates? No, we don't have coordinates here.
    # We can use the number of edges in a k-NN graph or just use distance based degree.
    # Let's compute a simple degree based on inverse distance sum.
    # Degree[i] = sum_j (1 / d[i,j]) for j != i
    # This is a proxy for how central a node is.
    # Or we can just use the sum of d_inv row.
    degree = np.sum(d_inv, axis=1)
    # Normalize degree
    max_deg = np.max(degree)
    if max_deg > 1e-9:
        deg_heur = degree / max_deg
    else:
        deg_heur = np.ones_like(degree)
    
    # Expand to matrix: deg_heur[i] * deg_heur[j]
    deg_mat = np.outer(deg_heur, deg_heur)
    
    # Combine
    # H = d_inv^alpha * sav_heur^beta * deg_mat^gamma
    # Use logs to avoid under/overflow
    log_h = alpha * np.log(d_inv + 1e-9) + beta * np.log(sav_heur + 1e-9) + gamma * np.log(deg_mat + 1e-9)
    # Exponentiate back, normalize to keep scale reasonable
    h = np.exp(log_h)
    max_h = np.max(h)
    if max_h > 1e-9:
        h = h / max_h
    else:
        h = np.ones_like(h)
    
    np.fill_diagonal(h, 0.0)
    
    # Pheromone initialization
    P = np.ones((n, n))
    
    best_cost = float('inf')
    
    for _ in range(n_iter):
        iteration_costs = []
        for _ in range(n_ants):
            node = 0
            remaining_cap = capacity
            cost = 0.0
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            
            while True:
                # Find feasible neighbors
                feasible = []
                for j in range(1, n):
                    if not visited[j] and demands[j] <= remaining_cap + 1e-9:
                        feasible.append(j)
                
                if not feasible:
                    # Return to depot
                    cost += distance_matrix[node, 0]
                    break
                
                # Calculate probabilities: P * H
                # P is uniform initially, so prob is proportional to H
                # But as P updates, it matters. For speed in sim, we can just use H 
                # or update P simply. Let's update P properly but keep it simple.
                
                probs = np.zeros(len(feasible))
                for k, j in enumerate(feasible):
                    probs[k] = P[node, j] * h[node, j]
                
                total_prob = np.sum(probs)
                if total_prob <= 1e-9:
                    # Fallback to random
                    idx = np.random.randint(len(feasible))
                else:
                    probs /= total_prob
                    r = np.random.random()
                    cumsum = np.cumsum(probs)
                    idx = np.searchsorted(cumsum, r)
                    if idx >= len(feasible):
                        idx = len(feasible) - 1
                
                next_node = feasible[idx]
                cost += distance_matrix[node, next_node]
                remaining_cap -= demands[next_node]
                visited[next_node] = True
                node = next_node
            
        iteration_costs.append(cost)
        
        best_iter = min(iteration_costs)
        if best_iter < best_cost:
            best_cost = best_iter
        
        # Update Pheromone (Simplified: Evaporate and Add)
        # P = rho * P + Q / best_cost
        # For speed in this inner sim, we might skip complex updates or do a very fast one
        P *= 0.9
        P += 1.0 / best_iter
        
    return best_cost

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ACO on CVRP.
    Uses adaptive parameter selection via local simulation.
    """
    dm = np.asarray(distance_matrix, dtype=float)
    D = np.asarray(demands, dtype=float)
    cap = float(capacity)
    n = dm.shape[0]
    
    # Basic validation
    if dm.shape != (n, n) or not np.isfinite(dm).all() or not np.isfinite(D).all():
        # Fallback to simple distance heuristic
        d = dm.copy()
        np.fill_diagonal(d, 1e9)
        return 1.0 / (d + 1e-9)
    
    # If time is running out, return a safe default
    if _remaining() < 20.0:
        d = dm.copy()
        np.fill_diagonal(d, 1e9)
        return 1.0 / (d + 1e-9)
    
    # Grid search over alpha (distance weight), beta (savings weight), gamma (degree weight)
    # Typical good ranges: alpha in [1, 3], beta in [0, 2], gamma in [0, 1]
    alphas = [1.5, 2.0, 2.5]
    betas = [0.0, 0.5, 1.0]
    gammas = [0.0, 0.5]
    
    best_score = float('inf')
    best_alpha = 2.0
    best_beta = 0.5
    best_gamma = 0.0
    
    # We have ~315s. 
    # Each simulation: 10 ants * 5 iters * 50 nodes * 50 checks ~ 125k ops. Very fast.
    # We can afford many simulations.
    
    for a in alphas:
        for b in betas:
            for g in gammas:
                if _remaining() < 30.0:
                    break
                try:
                    score = _fast_aco_sim(dm, D, cap, a, b, g, n_ants=10, n_iter=5)
                    if score < best_score:
                        best_score = score
                        best_alpha = a
                        best_beta = b
                        best_gamma = g
                except Exception:
                    pass
            if _remaining() < 30.0:
                break
        if _remaining() < 30.0:
            break
    
    # Refine around the best found parameters
    if _remaining() > 60.0:
        ref_alphas = np.linspace(best_alpha - 0.5, best_alpha + 0.5, 3)
        ref_betas = np.linspace(best_beta - 0.5, best_beta + 0.5, 3)
        ref_gammas = np.linspace(max(0.0, best_gamma - 0.5), best_gamma + 0.5, 3)
        for a in ref_alphas:
            if a <= 0: continue
            for b in ref_betas:
                if b < 0: continue
                for g in ref_gammas:
                    if g < 0: continue
                    if _remaining() < 40.0:
                        break
                    try:
                        score = _fast_aco_sim(dm, D, cap, a, b, g, n_ants=10, n_iter=5)
                        if score < best_score:
                            best_score = score
                            best_alpha = a
                            best_beta = b
                            best_gamma = g
                    except Exception:
                        pass
                if _remaining() < 40.0:
                    break
            if _remaining() < 40.0:
                break
    
    # Construct the final heuristic matrix with the best parameters
    d = dm.copy()
    np.fill_diagonal(d, 1e9)
    
    d_inv = 1.0 / (d + 1e-9)
    
    d_i0 = dm[:, 0]
    d_j0 = dm[0, :]
    savings = d_i0[:, np.newaxis] + d_j0[np.newaxis, :] - d
    savings_pos = np.maximum(savings, 0.0)
    max_sav = np.max(savings_pos)
    if max_sav < 1e-9:
        sav_heur = np.ones_like(savings)
    else:
        sav_heur = savings_pos / max_sav
    
    degree = np.sum(d_inv, axis=1)
    max_deg = np.max(degree)
    if max_deg > 1e-9:
        deg_heur = degree / max_deg
    else:
        deg_heur = np.ones_like(degree)
    
    deg_mat = np.outer(deg_heur, deg_heur)
    
    log_h = best_alpha * np.log(d_inv + 1e-9) + best_beta * np.log(sav_heur + 1e-9) + best_gamma * np.log(deg_mat + 1e-9)
    h = np.exp(log_h)
    
    # Normalize to prevent numerical issues in the actual ACO run
    max_h = np.max(h)
    if max_h > 1e-9:
        h = h / max_h
    else:
        h = np.ones_like(h)
        
    # Ensure positivity and handle diagonal
    h = np.maximum(h, 1e-9)
    np.fill_diagonal(h, 0.0)
    
    return h
# EVOLVE-BLOCK-END
