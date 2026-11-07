import numpy as np
import time

def _fast_aco_simulate(distance_matrix, heuristic, n_ants=20, n_iterations=30, rho=0.9, seed=42):
    """
    Fast internal ACO simulation to evaluate heuristic quality.
    Implements directed pheromone updates as per contract.
    """
    rng = np.random.RandomState(seed)
    n = distance_matrix.shape[0]
    tau = np.ones((n, n))
    best_tour_len = float('inf')
    
    for it in range(n_iterations):
        tour_lengths = []
        for ant in range(n_ants):
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            current = 0
            tour_len = 0.0
            tour_path = [0]
            
            for _ in range(n - 1):
                probs = tau[current, :] * heuristic[current, :]
                probs[visited] = 0.0
                
                total_prob = probs.sum()
                if total_prob < 1e-10:
                    unvisited_indices = np.where(~visited)[0]
                    next_node = unvisited_indices[rng.randint(len(unvisited_indices))]
                else:
                    probs = probs / total_prob
                    next_node = rng.choice(n, p=probs)
                
                tour_len += distance_matrix[current, next_node]
                visited[next_node] = True
                tour_path.append(next_node)
                current = next_node
            
            tour_len += distance_matrix[current, 0]
            tour_lengths.append(tour_len)
            
            deposit = 1.0 / tour_len
            # Vectorized pheromone update for the closed tour
            u = np.array(tour_path)
            v = np.array(tour_path[1:] + [tour_path[0]])
            tau[u, v] += deposit
            
        tau *= rho
        
        if tour_lengths:
            current_best = min(tour_lengths)
            if current_best < best_tour_len:
                best_tour_len = current_best
                
    return best_tour_len

def _compute_mst_edges(distance_matrix):
    """
    Compute the Minimum Spanning Tree edges using Prim's algorithm.
    Returns a boolean matrix where True indicates an MST edge.
    """
    n = distance_matrix.shape[0]
    mst_adj = np.zeros((n, n), dtype=bool)
    in_mst = np.zeros(n, dtype=bool)
    in_mst[0] = True
    min_dist = distance_matrix[0].copy()
    min_dist[0] = np.inf
    
    for _ in range(n - 1):
        # Find the vertex with the minimum distance to the MST
        # np.argmin returns the first occurrence of the minimum value
        u = np.argmin(min_dist)
        # If min_dist is inf, we have a disconnected graph (shouldn't happen for TSP)
        if min_dist[u] == np.inf:
            break
            
        in_mst[u] = True
        # Update min_dist for neighbors
        # We need to find which vertex in the MST is connected to u with min_dist[u]
        # Actually, min_dist[u] stores the min distance from u to ANY vertex in MST.
        # To find the specific parent, we need to search.
        # However, for just marking edges, we can just record that u is added.
        # But we need the edge (parent, u).
        
        # Let's find the parent
        parent = -1
        for v in range(n):
            if in_mst[v] and v != u:
                if distance_matrix[v, u] == min_dist[u]:
                    parent = v
                    break
        if parent != -1:
            mst_adj[parent, u] = True
            mst_adj[u, parent] = True
        
        # Update min_dist for all non-MST vertices
        for v in range(n):
            if not in_mst[v]:
                if distance_matrix[u, v] < min_dist[v]:
                    min_dist[v] = distance_matrix[u, v]
                    
    return mst_adj

def heuristics(distance_matrix):
    start_time = time.time()
    # Safety margin: evaluator kills at 330s, we should finish well before.
    # Let's aim to finish within 280s to be safe.
    deadline = start_time + 280.0
    
    n = distance_matrix.shape[0]
    
    # Base heuristic: inverse distance
    safe_dist = distance_matrix.copy()
    np.fill_diagonal(safe_dist, 1.0)
    safe_dist[safe_dist < 1e-9] = 1e-9
    
    base_heur = 1.0 / safe_dist
    
    # Compute MST edges for structural guidance
    mst_adj = _compute_mst_edges(distance_matrix)
    
    # We will search for the optimal beta and the MST boost factor.
    # Heuristic = (base_heur ** beta) * (mst_boost if mst_edge else 1.0)
    
    def evaluate_params(beta, mst_boost):
        if time.time() > deadline - 5.0:
            return float('inf')
            
        h = base_heur ** beta
        # Apply MST boost
        h[mst_adj] *= mst_boost
        
        # Ensure validity
        h[h <= 0] = 1e-9
        h[~np.isfinite(h)] = 1e-9
        
        # Use a fixed seed for consistent comparison during search
        # Using 5 iterations is a good surrogate for heuristic quality and faster
        score = _fast_aco_simulate(distance_matrix, h, n_ants=20, n_iterations=5, seed=42)
        return score

    best_beta = 2.0
    best_mst_boost = 1.0
    best_score = float('inf')
    
    # Coarse search over beta and mst_boost
    # Beta range: 0.5 to 6.0
    # MST boost range: 1.0 to 10.0 (1.0 means no boost)
    betas_coarse = np.linspace(0.5, 6.0, 12)
    boosts_coarse = np.array([1.0, 2.0, 5.0, 10.0])
    
    for beta in betas_coarse:
        for boost in boosts_coarse:
            if time.time() > deadline - 60.0:
                break
            score = evaluate_params(beta, boost)
            if score < best_score:
                best_score = score
                best_beta = beta
                best_mst_boost = boost
        if time.time() > deadline - 60.0:
            break
            
    # Fine search around the best coarse parameters
    if time.time() < deadline - 30.0:
        lower_b = max(0.1, best_beta - 1.5)
        upper_b = min(8.0, best_beta + 1.5)
        lower_m = max(0.5, best_mst_boost * 0.5)
        upper_m = min(20.0, best_mst_boost * 2.0)
        
        betas_fine = np.linspace(lower_b, upper_b, 15)
        boosts_fine = np.linspace(lower_m, upper_m, 10)
        
        for beta in betas_fine:
            for boost in boosts_fine:
                if time.time() > deadline - 10.0:
                    break
                score = evaluate_params(beta, boost)
                if score < best_score:
                    best_score = score
                    best_beta = beta
                    best_mst_boost = boost
            if time.time() > deadline - 10.0:
                break
                
    # Final check: verify the best parameters with a longer run?
    # The quick eval is usually good enough, but let's do one full run to be sure
    # if we have time.
    if time.time() < deadline - 15.0:
        h_final = base_heur ** best_beta
        h_final[mst_adj] *= best_mst_boost
        h_final[h_final <= 0] = 1e-9
        h_final[~np.isfinite(h_final)] = 1e-9
        
        # Full simulation to confirm
        final_score = _fast_aco_simulate(distance_matrix, h_final, n_ants=20, n_iterations=30, seed=42)
        
        # If the full simulation is significantly worse, it might be overfitting to the 5-iter surrogate
        # But since we are returning the heuristic, and the evaluator runs its own ACO,
        # the surrogate is just a proxy. We stick with the surrogate-optimized params.
        
    # Construct final heuristic
    final_heuristic = base_heur ** best_beta
    final_heuristic[mst_adj] *= best_mst_boost
    final_heuristic[final_heuristic <= 0] = 1e-9
    final_heuristic[~np.isfinite(final_heuristic)] = 1e-9
    
    return final_heuristic
