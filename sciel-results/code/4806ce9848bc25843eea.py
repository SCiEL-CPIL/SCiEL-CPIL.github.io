"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline tracking
_start_time = None

def _get_start_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _start_time

def _check_deadline(safety_margin=5.0):
    """Return True if we should stop and return depot."""
    elapsed = time.time() - _get_start_time()
    return elapsed > (120.0 - safety_margin)

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    if len(unvisited_nodes) == 0:
        return 0

    # Deadline guard
    if _check_deadline():
        return 0

    dm = np.asarray(distance_matrix, dtype=np.float64)
    demands_arr = np.asarray(demands, dtype=np.float64)

    # 1. Filter feasible nodes: demand[node] <= rest_capacity
    feasible_nodes = []
    for node in unvisited_nodes:
        if demands_arr[node] <= rest_capacity:
            feasible_nodes.append(node)
            
    if len(feasible_nodes) == 0:
        return 0

    feasible_np = np.array(feasible_nodes)
    n_feasible = len(feasible_np)

    # Structural change: Cluster-density based selection with 2-step greedy lookahead.
    # For each candidate i, we estimate the cost of a short "mini-route" starting
    # from current -> i -> best_next(i) -> best_next(best_next(i)), where best_next
    # is chosen greedily as the nearest feasible node that fits remaining capacity.
    # We score each candidate by this estimated mini-route cost.
    # This is fundamentally different from savings heuristics: it directly estimates
    # local path cost rather than pairwise depot savings.

    # Precompute distances
    dist_current = dm[current_node, feasible_np]
    dm_feasible = dm[np.ix_(feasible_np, feasible_np)]
    
    # For each candidate, simulate a 2-step greedy continuation
    # Step 1: current -> candidate (cost: dist_current[i])
    # Step 2: candidate -> nearest feasible neighbor j (cost: dm_feasible[i,j])
    #         where j must satisfy demands[j] <= rest_capacity - demands[candidate]
    # Step 3: j -> nearest feasible neighbor k (cost: dm_feasible[j,k])
    #         where k must satisfy demands[k] <= rest_capacity - demands[candidate] - demands[j]
    
    # To keep it fast, limit lookahead depth to 2 additional nodes
    # and only consider top-K nearest neighbors at each step
    
    K = min(3, n_feasible - 1)  # consider top 3 nearest at each step
    
    scores = np.full(n_feasible, np.inf)
    
    for idx in range(n_feasible):
        cand = feasible_np[idx]
        rem_cap_after = rest_capacity - demands_arr[cand]
        
        # Cost so far: current -> cand
        cost_so_far = dist_current[idx]
        
        # Find nearest feasible neighbors from cand (excluding self)
        if n_feasible <= 1:
            # Only one feasible node, just take it
            scores[idx] = cost_so_far
            continue
        
        # Get distances from cand to all other feasible nodes
        dists_from_cand = dm_feasible[idx].copy()
        dists_from_cand[idx] = np.inf  # exclude self
        
        # Get top K nearest neighbors
        if n_feasible > 1:
            top_k_idx = np.argpartition(dists_from_cand, K)[:K]
        else:
            top_k_idx = np.array([])
        
        best_2step = np.inf
        
        for j_idx in top_k_idx:
            j = feasible_np[j_idx]
            if demands_arr[j] > rem_cap_after:
                continue
            
            cost_2 = cost_so_far + dists_from_cand[j_idx]
            
            # Try one more step from j
            rem_cap_after2 = rem_cap_after - demands_arr[j]
            if rem_cap_after2 < 0:
                scores[idx] = min(scores[idx], cost_2)
                continue
            
            dists_from_j = dm_feasible[j_idx].copy()
            dists_from_j[j_idx] = np.inf  # exclude j itself
            
            if n_feasible > 1:
                top_k_idx2 = np.argpartition(dists_from_j, K)[:K]
            else:
                top_k_idx2 = np.array([])
            
            for k_idx in top_k_idx2:
                k = feasible_np[k_idx]
                if demands_arr[k] > rem_cap_after2:
                    continue
                
                cost_3 = cost_2 + dists_from_j[k_idx]
                best_2step = min(best_2step, cost_3)
            
            # Also consider stopping after 2 steps (going to depot)
            cost_2_depot = cost_2 + dm[j, depot]
            best_2step = min(best_2step, cost_2_depot)
        
        # If no valid 2-step continuation found, use cost_so_far + depot return
        if np.isinf(best_2step):
            best_2step = cost_so_far + dm[cand, depot]
        
        scores[idx] = best_2step
    
    # Also compute the cost of returning to depot now
    return_cost = dm[current_node, depot]
    
    min_score = np.min(scores)
    min_idx = np.argmin(scores)
    
    # Decide: return to depot or pick the best candidate?
    # Return to depot if:
    # 1. No feasible nodes (already handled)
    # 2. The best candidate's mini-route cost is significantly worse than just returning
    #    AND capacity is very tight (risk of getting stuck)
    if rest_capacity <= 3 and min_score > return_cost * 1.5:
        return 0
    
    return int(feasible_np[min_idx])
# EVOLVE-BLOCK-END
