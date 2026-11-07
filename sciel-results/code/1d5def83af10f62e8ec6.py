import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    """
    Computes an opportunity-cost based heuristic matrix for ACO.
    The heuristic penalizes nodes with high return-to-start distance and high edge distance,
    weighted by the node's prize.
    """
    prizes = np.array(prizes, dtype=np.float64)
    dist = np.array(distance_matrix, dtype=np.float64)
    n = len(prizes)
    
    # Deadline guard
    t_start = time.time()
    deadline = 300.0
    safety_margin = 10.0 # seconds
    
    # Precompute constants
    d_j0 = dist[:, 0]  # Distance from each node to start
    eps = 1e-9
    
    def simulate_aco(prizes, dist, max_len, eta, seed):
        """
        Fast simulation of the ACO process.
        """
        n = len(prizes)
        np.random.seed(seed)
        tau = np.ones((n, n))
        best_overall = 0.0
        
        for _ in range(30):
            best_iter = 0.0
            for _ in range(20):
                current = 0
                visited_set = [False] * n
                visited_set[0] = True
                path_len = 0.0
                total_prize = 0.0
                
                while True:
                    # Find valid nodes and their weights
                    valid_indices = []
                    weights = []
                    
                    # Optimization: Pre-compute potential valid nodes? 
                    # Since path_len changes, the set of valid nodes shrinks.
                    # We must check all unvisited nodes.
                    for j in range(n):
                        if visited_set[j]:
                            continue
                        # Check constraint: path_len + d[i,j] + d[j,0] <= max_len
                        if path_len + dist[current, j] + d_j0[j] <= max_len:
                            valid_indices.append(j)
                            val = tau[current, j] * eta[current, j]
                            weights.append(max(val, 1e-9))
                    
                    if not valid_indices:
                        break
                    
                    w = np.array(weights)
                    p = w / w.sum()
                    idx = np.random.choice(len(valid_indices), p=p)
                    next_node = valid_indices[idx]
                    
                    path_len += dist[current, next_node]
                    total_prize += prizes[next_node]
                    visited_set[next_node] = True
                    current = next_node
                
                if total_prize > best_iter:
                    best_iter = total_prize
            
            # Pheromone update
            tau = tau * 0.9 + best_iter / 10.0
            
            if best_iter > best_overall:
                best_overall = best_iter
        
        return best_overall

    best_eta = None
    best_score = -1.0
    best_params = (1.0, 1.0)
    
    # Grid search for alpha (weight on d[j,0]) and beta (weight on d[i,j])
    # alpha: 0 to 3
    # beta: 0 to 3
    # We use a coarse grid first, then refine.
    
    alpha_candidates = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    beta_candidates = [0.0, 0.5, 1.0, 1.5, 2.0]
    
    # Evaluate candidates
    for alpha in alpha_candidates:
        if time.time() - t_start > deadline - safety_margin:
            break
        for beta in beta_candidates:
            if time.time() - t_start > deadline - safety_margin:
                break
            
            # Construct heuristic
            # eta[i,j] = prize[j] / (d[j,0]^alpha * d[i,j]^beta)
            # Handle zeros in dist
            d_ij_safe = dist + eps
            d_j0_safe = d_j0[None, :] + eps
            
            eta = prizes[None, :] / (d_j0_safe ** alpha) * (1.0 / (d_ij_safe ** beta))
            eta[eta <= 0] = 1e-9
            
            # Evaluate on a few seeds
            # Using seeds 0, 1, 2 for robustness
            scores = []
            for s in [0, 1, 2]:
                score = simulate_aco(prizes, dist, max_len, eta, s)
                scores.append(score)
            mean_score = np.mean(scores)
            
            if mean_score > best_score:
                best_score = mean_score
                best_eta = eta.copy()
                best_params = (alpha, beta)
    
    # If we have time, refine around the best parameters
    if time.time() - t_start < deadline - safety_margin and best_eta is not None:
        alpha_best, beta_best = best_params
        
        # Local search
        for da in [-0.5, -0.25, 0.25, 0.5]:
            for db in [-0.5, -0.25, 0.25, 0.5]:
                if time.time() - t_start > deadline - safety_margin:
                    break
                
                alpha = alpha_best + da
                beta = beta_best + db
                
                if alpha < 0.1: alpha = 0.1
                if beta < 0.0: beta = 0.0
                
                d_ij_safe = dist + eps
                d_j0_safe = d_j0[None, :] + eps
                
                eta = prizes[None, :] / (d_j0_safe ** alpha) * (1.0 / (d_ij_safe ** beta))
                eta[eta <= 0] = 1e-9
                
                scores = []
                for s in [0, 1, 2]:
                    score = simulate_aco(prizes, dist, max_len, eta, s)
                    scores.append(score)
                mean_score = np.mean(scores)
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_eta = eta.copy()
                    best_params = (alpha, beta)

    if best_eta is None:
        # Fallback: simple prize / distance to start
        d_j0_safe = d_j0[None, :] + eps
        best_eta = prizes[None, :] / d_j0_safe
        best_eta[best_eta <= 0] = 1e-9
        
    return best_eta
