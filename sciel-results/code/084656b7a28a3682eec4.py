"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _run_aco(prizes, distance_matrix, max_len, heuristic, n_ants=20, n_iters=30, seed=42):
    """Run the fixed ACO procedure and return the best total prize collected by any ant."""
    n = len(prizes)
    rng = np.random.default_rng(seed)
    
    # Pheromone starts at 1
    pheromone = np.ones((n, n), dtype=np.float64)
    
    best_total = 0.0
    
    for it in range(n_iters):
        iter_best = 0.0
        
        for ant in range(n_ants):
            visited = [0]
            current = 0
            length_so_far = 0.0
            total_prize = prizes[0]  # prize 0 at node 0
            
            while True:
                # Find unvisited nodes j such that length_so_far + d[current,j] + d[j,0] <= max_len
                candidates = []
                for j in range(n):
                    if j in visited:
                        continue
                    if length_so_far + distance_matrix[current, j] + distance_matrix[j, 0] <= max_len:
                        candidates.append(j)
                
                if not candidates:
                    break
                
                # Compute probabilities proportional to pheromone * heuristic
                probs = np.array([pheromone[current, j] * heuristic[current, j] for j in candidates])
                probs = np.maximum(probs, 1e-15)  # ensure positive
                probs = probs / probs.sum()
                
                # Select next node
                next_j = candidates[rng.choice(len(candidates), p=probs)]
                visited.append(next_j)
                length_so_far += distance_matrix[current, next_j]
                total_prize += prizes[next_j]
                current = next_j
            
            if total_prize > iter_best:
                iter_best = total_prize
        
        if iter_best > best_total:
            best_total = iter_best
        
        # Pheromone update: multiply by 0.9, add (best prize of iteration) / 10
        pheromone = pheromone * 0.9 + (iter_best / 10.0)
    
    return best_total


def _compute_heuristic_ratio(prizes, distance_matrix, max_len, alpha=1.0):
    """Greedy ratio heuristic: prize / (d[i,j] + d[j,0])^alpha."""
    n = len(prizes)
    total_cost = distance_matrix + distance_matrix[:, 0:1]
    total_cost_safe = np.maximum(total_cost, 1e-9)
    heuristic = prizes[None, :] / (total_cost_safe ** alpha)
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    return heuristic


def _compute_heuristic_prize_weighted(prizes, distance_matrix, max_len, beta=2.0):
    """Heuristic emphasizing high prizes with distance penalty."""
    n = len(prizes)
    # Normalize prizes to [0,1]
    max_prize = np.max(prizes)
    if max_prize > 0:
        norm_prizes = prizes / max_prize
    else:
        norm_prizes = prizes.copy()
    
    # Distance penalty: closer is better
    d_to_origin = distance_matrix[:, 0]
    d_from_current = distance_matrix  # will be used in full version
    
    # Simple: prize^beta / (1 + d[i,j])
    heuristic = (norm_prizes[None, :] ** beta) / (1.0 + distance_matrix)
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    return heuristic


def _compute_heuristic_centrality(prizes, distance_matrix, max_len):
    """Centrality-based: boost nodes that are central to high-prize areas."""
    n = len(prizes)
    
    # Compute a centrality score for each node: weighted by prizes of nearby nodes
    # centrality[j] = sum over i of prizes[i] / (1 + d[i,j])
    centrality = np.zeros(n)
    for j in range(n):
        for i in range(n):
            if i != j:
                centrality[j] += prizes[i] / (1.0 + distance_matrix[i, j])
    
    # Heuristic: combination of direct prize and centrality
    # heuristic[i,j] = (prizes[j] + 0.5 * centrality[j]) / (1 + d[i,j])
    direct = prizes[None, :]
    central = centrality[None, :]
    heuristic = (direct + 0.5 * central) / (1.0 + distance_matrix)
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    return heuristic


def _compute_heuristic_cluster(prizes, distance_matrix, max_len):
    """Cluster-aware: favor nodes that are close to other high-prize nodes (enables multi-visit)."""
    n = len(prizes)
    
    # For each node j, compute the sum of prizes of nodes within a radius
    # This encourages visiting nodes in dense high-prize regions
    radius = max_len / 2.0  # half the max length as a reasonable radius
    cluster_score = np.zeros(n)
    for j in range(n):
        for i in range(n):
            if i != j and distance_matrix[j, i] < radius:
                cluster_score[j] += prizes[i]
    
    # Heuristic: (prize + cluster_score) / (d[i,j] + d[j,0])
    total_cost = distance_matrix + distance_matrix[:, 0:1]
    total_cost_safe = np.maximum(total_cost, 1e-9)
    heuristic = (prizes[None, :] + 0.3 * cluster_score[None, :]) / total_cost_safe
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    return heuristic


def heuristics(prizes, distance_matrix, max_len):
    """
    Return an edge-attractiveness matrix for ACO orienteering.
    
    Uses a cluster-aware heuristic that combines direct prize value,
    local cluster density, and distance cost to encourage ants to
    build routes through dense high-prize regions.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    n = len(prizes)
    
    # Deadline guard
    start_time = time.time()
    deadline = 300.0 - 10.0  # 10 second safety margin
    
    # Generate multiple heuristic candidates and evaluate them with ACO
    heuristic_candidates = [
        ("ratio", lambda: _compute_heuristic_ratio(prizes, distance_matrix, max_len, alpha=1.0)),
        ("prize_weighted", lambda: _compute_heuristic_prize_weighted(prizes, distance_matrix, max_len, beta=2.0)),
        ("centrality", lambda: _compute_heuristic_centrality(prizes, distance_matrix, max_len)),
        ("cluster", lambda: _compute_heuristic_cluster(prizes, distance_matrix, max_len)),
    ]
    
    best_score = -1.0
    best_heuristic = None
    
    for name, gen_func in heuristic_candidates:
        if time.time() > deadline:
            break
        
        try:
            h = gen_func()
            # Validate
            if h.shape != (n, n):
                continue
            if not np.all(np.isfinite(h)):
                continue
            if not np.all(h > 0):
                continue
            
            # Evaluate with ACO using a fixed seed for consistency
            score = _run_aco(prizes, distance_matrix, max_len, h, n_ants=20, n_iters=30, seed=42)
            
            if score > best_score:
                best_score = score
                best_heuristic = h
        except Exception:
            continue
    
    # If no heuristic was evaluated, fall back to a simple one
    if best_heuristic is None:
        best_heuristic = _compute_heuristic_ratio(prizes, distance_matrix, max_len, alpha=1.0)
    
    # Final safety: ensure all entries are positive and finite
    best_heuristic = np.where(best_heuristic > 0, best_heuristic, 1e-9)
    best_heuristic = np.where(np.isfinite(best_heuristic), best_heuristic, 1e-9)
    
    return best_heuristic
# EVOLVE-BLOCK-END
