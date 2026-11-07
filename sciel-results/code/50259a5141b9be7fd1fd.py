import numpy as np
import time
from scipy.sparse.csgraph import minimum_spanning_tree

def _fast_aco_simulate(distance_matrix, heuristic, n_ants=20, n_iterations=30, rho=0.9, seed=42):
    """
    Fast internal ACO simulation to evaluate heuristic quality.
    Implements directed pheromone updates as per contract.
    Uses a local RNG for reproducibility.
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
            for i in range(n):
                u = tour_path[i]
                v = tour_path[(i + 1) % n]
                tau[u, v] += deposit
            
        tau *= rho
        
        if tour_lengths:
            current_best = min(tour_lengths)
            if current_best < best_tour_len:
                best_tour_len = current_best
                
    return best_tour_len

def _compute_mst_mask(distance_matrix):
    """
    Compute a boolean mask of MST edges.
    """
    n = distance_matrix.shape[0]
    # scipy's minimum_spanning_tree expects a weighted graph
    # It returns a sparse matrix. We can convert to dense for small N.
    # Note: MST is undirected. The heuristic is directed.
    # We should boost both (i,j) and (j,i) if (i,j) is in MST.
    mst = minimum_spanning_tree(distance_matrix)
    mst_dense = mst.toarray()
    # Symmetrize the mask
    mask = (mst_dense > 0) | (mst_dense.T > 0)
    return mask

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 290.0
    
    n = distance_matrix.shape[0]
    
    # Base heuristic: inverse distance
    safe_dist = distance_matrix.copy()
    np.fill_diagonal(safe_dist, 1.0)
    safe_dist[safe_dist < 1e-9] = 1e-9
    
    base_heur = 1.0 / safe_dist
    
    # Compute MST mask
    mst_mask = _compute_mst_mask(distance_matrix)
    
    best_beta = 1.0
    best_gamma = 0.0
    best_score = float('inf')
    
    # Search grid
    # Beta: 1.0 to 3.0, step 0.2
    # Gamma: 0.0 to 1.5, step 0.3
    betas = np.arange(1.0, 3.2, 0.2)
    gammas = np.arange(0.0, 1.6, 0.3)
    
    for beta in betas:
        for gamma in gammas:
            if time.time() > deadline - 30.0:
                break
                
            # Construct heuristic
            # h = (1/d)^beta * (1 + gamma * mst_mask)
            h = (base_heur ** beta) * (1.0 + gamma * mst_mask)
            
            # Ensure validity
            h[h <= 0] = 1e-9
            h[~np.isfinite(h)] = 1e-9
            
            score = _fast_aco_simulate(distance_matrix, h, n_ants=20, n_iterations=30, seed=42)
            
            if score < best_score:
                best_score = score
                best_beta = beta
                best_gamma = gamma

    # Fine-grained search around best found
    if time.time() < deadline - 40.0:
        lower_b = max(0.5, best_beta - 0.2)
        upper_b = min(4.0, best_beta + 0.2)
        lower_g = max(0.0, best_gamma - 0.3)
        upper_g = min(2.0, best_gamma + 0.3)
        
        fine_betas = np.linspace(lower_b, upper_b, 5)
        fine_gammas = np.linspace(lower_g, upper_g, 5)
        
        for beta in fine_betas:
            for gamma in fine_gammas:
                if time.time() > deadline - 30.0:
                    break
                h = (base_heur ** beta) * (1.0 + gamma * mst_mask)
                h[h <= 0] = 1e-9
                h[~np.isfinite(h)] = 1e-9
                score = _fast_aco_simulate(distance_matrix, h, n_ants=20, n_iterations=30, seed=42)
                
                if score < best_score:
                    best_score = score
                    best_beta = beta
                    best_gamma = gamma

    # Construct final heuristic
    final_heuristic = (base_heur ** best_beta) * (1.0 + best_gamma * mst_mask)
    final_heuristic[final_heuristic <= 0] = 1e-9
    final_heuristic[~np.isfinite(final_heuristic)] = 1e-9
    
    return final_heuristic
