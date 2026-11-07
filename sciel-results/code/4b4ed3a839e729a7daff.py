"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix

def _fast_aco_sim_v2(dm, demands, capacity, H, n_ants=20, n_iterations=10, seed=42, time_limit=5.0):
    """
    Faster, higher-fidelity simulation of the evaluator's ACO.
    Matches n_ants=20.
    """
    start_time = time.time()
    n = dm.shape[0]
    tau = np.ones((n, n), dtype=np.float64)
    rng = np.random.RandomState(seed)
    best_cost = np.inf
    demands = np.asarray(demands, dtype=np.float64)
    
    for it in range(n_iterations):
        if time.time() - start_time > time_limit:
            break

        iteration_costs = []

        for ant in range(n_ants):
            if time.time() - start_time > time_limit:
                break

            current_node = 0
            remaining_cap = capacity
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            total_dist = 0.0

            while True:
                valid_mask = ~visited
                valid_mask[0] = False
                valid_mask &= (demands <= remaining_cap)

                if not np.any(valid_mask):
                    total_dist += dm[current_node, 0]
                    remaining_cap = capacity
                    current_node = 0
                    if np.all(visited):
                        break
                    continue

                row_tau = tau[current_node]
                row_eta = H[current_node]

                weights = row_tau * row_eta
                weights[~valid_mask] = 0.0

                total_w = np.sum(weights)
                if total_w <= 0:
                    valid_indices = np.where(valid_mask)[0]
                    next_node = rng.choice(valid_indices)
                else:
                    probs = weights / total_w
                    next_node = rng.choice(n, p=probs)

                total_dist += dm[current_node, next_node]
                visited[next_node] = True
                remaining_cap -= demands[next_node]
                current_node = next_node

                if np.all(visited):
                    total_dist += dm[current_node, 0]
                    break

            iteration_costs.append(total_dist)

        if iteration_costs:
            current_best = min(iteration_costs)
            if current_best < best_cost:
                best_cost = current_best
            tau *= 0.9
            tau += 1.0 / current_best

    return best_cost


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute heuristic matrix for ACO.
    Returns an edge-attractiveness matrix of the same shape as distance_matrix.
    """
    start_time = time.time()
    deadline = start_time + 290.0
    
    n = distance_matrix.shape[0]
    dm = np.asarray(distance_matrix, dtype=np.float64)
    coords = np.asarray(coordinates, dtype=np.float64)
    demands = np.asarray(demands, dtype=np.float64)
    
    dm_safe = np.where(dm > 1e-10, dm, 1e10)
    
    # Structural candidates
    H1 = 1.0 / dm_safe
    H2 = 1.0 / (dm_safe ** 2)
    
    d_demands = demands.reshape(-1, 1)
    H3 = 1.0 / (dm_safe * (1.0 + 0.1 * d_demands))
    H4 = 1.0 / ((dm_safe ** 2) * (1.0 + 0.1 * d_demands))
    
    dist_depot = dm[0, :]
    dist_depot_safe = np.where(dist_depot > 1e-10, dist_depot, 1e10)
    boost = 1.0 / (1.0 + dist_depot_safe).T
    H5 = H1 * boost
    H6 = H2 * boost
    
    candidates = []
    for h in [H1, H2, H3, H4, H5, H6]:
        h_safe = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
        candidates.append(h_safe)

    # NEW: MST-based heuristic
    try:
        # Build sparse graph for MST
        # We use the distance matrix as weights
        # scipy.sparse.csgraph.minimum_spanning_tree expects a sparse matrix
        # Convert dm to sparse (dense to sparse is O(n^2) memory but n=51 is small)
        dm_sparse = csr_matrix(dm_safe)
        mst = minimum_spanning_tree(dm_sparse)
        mst_dense = mst.toarray()
        
        # The MST has 0 for non-edges and weight for edges.
        # We want high attractiveness for MST edges.
        # Let's create a binary mask of MST edges
        mst_mask = (mst_dense > 0).astype(float)
        
        # Base heuristic: Inverse distance, but boosted for MST edges
        # H_mst = H1 * (1 + boost_factor * mst_mask)
        # Or simply: H_mst = H1 where not MST, and H1 * 2 where MST?
        # Let's try a smooth version:
        # Attractiveness is high if the edge is in the MST.
        # Since MST is a tree, it captures the global skeleton.
        
        # Candidate 7: H1 with MST boost
        H7 = H1 * (1.0 + 1.0 * mst_mask)
        
        # Candidate 8: H2 with MST boost
        H8 = H2 * (1.0 + 1.0 * mst_mask)
        
        # Candidate 9: Pure MST structure? No, needs to be positive everywhere.
        # Maybe: 1/dm_safe where edge is in MST, else 1e-9? Too sparse.
        # Let's stick to boosts.
        
        h7_safe = np.where(np.isfinite(H7) & (H7 > 0), H7, 1e-9)
        h8_safe = np.where(np.isfinite(H8) & (H8 > 0), H8, 1e-9)
        
        candidates.append(h7_safe)
        candidates.append(h8_safe)
    except Exception:
        pass

    # NEW: KNN-based heuristic (from previous version, kept)
    try:
        tree = cKDTree(coords)
        knn_dists, knn_idxs = tree.query(coords, k=6)
        
        H_nn = np.zeros_like(dm)
        for i in range(n):
            for rank in range(1, 6):
                j = knn_idxs[i, rank]
                d = knn_dists[i, rank]
                weight = (1.0 / (rank + 1)) * (1.0 / (1.0 + d))
                H_nn[i, j] = weight
                H_nn[j, i] = weight
                
        H_nn_safe = np.where(np.isfinite(H_nn) & (H_nn > 0), H_nn, 1e-9)
        candidates.append(H_nn_safe)
    except Exception:
        pass

    best_H = candidates[0]
    best_score = np.inf
    
    # Evaluate candidates
    # We have 8-9 candidates now.
    # 3 seeds each, 5s limit each -> 15s per candidate.
    # 9 * 15 = 135s. Plenty of time.
    
    for i, h in enumerate(candidates):
        if time.time() > deadline - 30.0:
            break
            
        costs = []
        for s in [0, 1, 2]:
            cost = _fast_aco_sim_v2(dm, demands, capacity, h, 
                                    n_ants=20, n_iterations=10, 
                                    seed=s, time_limit=5.0)
            if cost < np.inf:
                costs.append(cost)
                
        if costs:
            avg_cost = np.mean(costs)
            if avg_cost < best_score:
                best_score = avg_cost
                best_H = h

    # Refine with mixing
    if time.time() < deadline - 40.0:
        for alpha in [0.5, 0.7, 0.8, 0.9]:
            H_mix = alpha * best_H + (1 - alpha) * H1
            H_mix = np.where(np.isfinite(H_mix) & (H_mix > 0), H_mix, 1e-9)
            
            costs = []
            for s in [0, 1, 2]:
                cost = _fast_aco_sim_v2(dm, demands, capacity, H_mix, 
                                        n_ants=20, n_iterations=10, 
                                        seed=s, time_limit=5.0)
                if cost < np.inf:
                    costs.append(cost)
            
            if costs:
                avg_cost = np.mean(costs)
                if avg_cost < best_score:
                    best_score = avg_cost
                    best_H = H_mix

    return best_H

# EVOLVE-BLOCK-END
