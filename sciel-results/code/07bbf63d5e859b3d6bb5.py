import numpy as np
import time

def _aco_simulate(distance_matrix, heuristic, n_ants=20, n_iterations=30, rho=0.9, seed=42):
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
        u = np.argmin(min_dist)
        if min_dist[u] == np.inf:
            break
            
        in_mst[u] = True
        
        parent = -1
        for v in range(n):
            if in_mst[v] and v != u:
                if distance_matrix[v, u] == min_dist[u]:
                    parent = v
                    break
        if parent != -1:
            mst_adj[parent, u] = True
            mst_adj[u, parent] = True
        
        for v in range(n):
            if not in_mst[v]:
                if distance_matrix[u, v] < min_dist[v]:
                    min_dist[v] = distance_matrix[u, v]
                    
    return mst_adj

def _compute_knn_edges(distance_matrix, k=3):
    """
    Compute k-nearest neighbor edges.
    Returns a boolean matrix where True indicates a KNN edge.
    """
    n = distance_matrix.shape[0]
    knn_adj = np.zeros((n, n), dtype=bool)
    
    for i in range(n):
        dists = distance_matrix[i].copy()
        dists[i] = np.inf
        # Get indices of k smallest distances
        k_indices = np.argpartition(dists, k)[:k]
        for j in k_indices:
            knn_adj[i, j] = True
            knn_adj[j, i] = True
            
    return knn_adj

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
    
    # Compute KNN edges
    knn_adj = _compute_knn_edges(distance_matrix, k=3)
    
    # Combine structural edges: MST is strongest, KNN is secondary
    # We will search for optimal weights and exponents.
    
    def evaluate_params(beta, mst_w, knn_w):
        if time.time() > deadline - 5.0:
            return float('inf')
            
        h = base_heur ** beta
        
        # Apply structural boosts
        # MST edges get mst_w boost
        # KNN edges (not already boosted by MST) get knn_w boost
        # We want to encourage the union of these structures.
        
        # Start with base
        h_new = h.copy()
        
        # Boost MST
        h_new[mst_adj] *= mst_w
        
        # Boost KNN (only if not already boosted by MST to avoid double counting in logic, 
        # but actually multiplicative is fine. Let's just apply both.)
        # If an edge is in both, it gets mst_w * knn_w.
        h_new[knn_adj] *= knn_w
        
        # Ensure validity
        h_new[h_new <= 0] = 1e-9
        h_new[~np.isfinite(h_new)] = 1e-9
        
        # Use a fixed seed for consistent comparison during search
        # Using 5 iterations is a good surrogate for heuristic quality and faster
        score = _aco_simulate(distance_matrix, h_new, n_ants=20, n_iterations=5, seed=42)
        return score

    best_beta = 2.0
    best_mst_w = 2.0
    best_knn_w = 1.5
    best_score = float('inf')
    
    # Coarse search
    # Beta: 0.5 to 6.0
    # MST weight: 1.0 to 10.0
    # KNN weight: 1.0 to 5.0
    
    betas_coarse = np.linspace(0.5, 6.0, 12)
    msts_coarse = np.array([1.0, 2.0, 4.0, 8.0])
    knns_coarse = np.array([1.0, 1.5, 2.5, 4.0])
    
    for beta in betas_coarse:
        for mst_w in msts_coarse:
            for knn_w in knns_coarse:
                if time.time() > deadline - 60.0:
                    break
                score = evaluate_params(beta, mst_w, knn_w)
                if score < best_score:
                    best_score = score
                    best_beta = beta
                    best_mst_w = mst_w
                    best_knn_w = knn_w
            if time.time() > deadline - 60.0:
                break
        if time.time() > deadline - 60.0:
            break
            
    # Fine search around the best coarse parameters
    if time.time() < deadline - 30.0:
        lower_b = max(0.1, best_beta - 1.5)
        upper_b = min(8.0, best_beta + 1.5)
        lower_m = max(0.5, best_mst_w * 0.5)
        upper_m = min(15.0, best_mst_w * 2.0)
        lower_k = max(0.5, best_knn_w * 0.5)
        upper_k = min(8.0, best_knn_w * 2.0)
        
        betas_fine = np.linspace(lower_b, upper_b, 15)
        msts_fine = np.linspace(lower_m, upper_m, 10)
        knns_fine = np.linspace(lower_k, upper_k, 10)
        
        for beta in betas_fine:
            for mst_w in msts_fine:
                for knn_w in knns_fine:
                    if time.time() > deadline - 10.0:
                        break
                    score = evaluate_params(beta, mst_w, knn_w)
                    if score < best_score:
                        best_score = score
                        best_beta = beta
                        best_mst_w = mst_w
                        best_knn_w = knn_w
                if time.time() > deadline - 10.0:
                    break
            if time.time() > deadline - 10.0:
                break
                
    # Final heuristic construction
    final_heuristic = base_heur ** best_beta
    final_heuristic[mst_adj] *= best_mst_w
    final_heuristic[knn_adj] *= best_knn_w
    final_heuristic[final_heuristic <= 0] = 1e-9
    final_heuristic[~np.isfinite(final_heuristic)] = 1e-9
    
    return final_heuristic
