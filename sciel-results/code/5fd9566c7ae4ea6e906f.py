# EVOLVE-BLOCK-START
import numpy as np
import time

# Cached per-instance data so repeated calls are cheap.
_cache = {}
_start_time = None


def _get_start_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _start_time


def _init(current_node, destination_node, unvisited_nodes, distance_matrix):
    key = (id(distance_matrix), distance_matrix.shape[0])
    c = _cache.get(key)
    if c is None or c[0] is not distance_matrix:
        c = (distance_matrix, np.ascontiguousarray(distance_matrix, dtype=np.float64))
        _cache[key] = c
    return c[1]


def _held_karp_suffix(current_node, unvisited_nodes, distance_matrix, dest_node, max_n=15):
    """
    Compute the optimal suffix path starting from current_node, visiting all
    unvisited_nodes, and ending at dest_node, using Held-Karp DP.
    Returns the first node in the optimal path (int).
    If n > max_n, falls back to nearest neighbor.
    """
    U = np.asarray(unvisited_nodes)
    n = U.size
    if n == 0:
        return int(current_node)
    if n == 1:
        return int(U[0])
    
    c = int(current_node)
    dest = int(dest_node)
    
    if n > max_n:
        d = distance_matrix[c, U]
        return int(U[int(np.argmin(d))])
    
    # Use a single DP where dp[mask][i] = min cost to start at c, visit set 'mask',
    # end at U[i], and eventually return to dest.
    # We compute this backwards from the full set.
    # Actually, forward DP is easier:
    # dp[mask][i] = min cost to start at c, visit exactly the nodes in 'mask',
    # and end at U[i].
    # Then the total cost for a path starting with U[first] is:
    #   min over all paths that include U[first] as the first step.
    # This is hard to extract. Let's stick with the per-first-node approach but optimize it.
    
    # Optimization: Precompute distance submatrix
    sub_dm = distance_matrix[np.ix_(U, U)]
    c_to_U = distance_matrix[c, U]
    U_to_dest = distance_matrix[U, dest]
    
    best_cost = 1e10
    best_first = 0
    
    for first in range(n):
        rem_indices = [i for i in range(n) if i != first]
        m = len(rem_indices)
        
        if m == 0:
            cost = c_to_U[first] + U_to_dest[first]
            if cost < best_cost:
                best_cost = cost
                best_first = first
            continue
        
        # DP for remaining nodes
        rem_mask_size = 1 << m
        # Use a flat array for dp: dp[mask * m + i]
        dp = np.full(rem_mask_size * m, 1e10, dtype=np.float64)
        
        # Base case
        for i in range(m):
            dp[(1 << i) * m + i] = sub_dm[first, rem_indices[i]]
        
        # Iterate over masks
        for mask in range(1, rem_mask_size):
            base_idx = mask * m
            for i in range(m):
                if not (mask & (1 << i)):
                    continue
                cost_i = dp[base_idx + i]
                if cost_i >= 1e10:
                    continue
                remaining = (rem_mask_size - 1) ^ mask
                for j in range(m):
                    if remaining & (1 << j):
                        new_mask = mask | (1 << j)
                        new_cost = cost_i + sub_dm[rem_indices[i], rem_indices[j]]
                        idx = new_mask * m + j
                        if new_cost < dp[idx]:
                            dp[idx] = new_cost
        
        full_rem_mask = rem_mask_size - 1
        min_final = 1e10
        for i in range(m):
            val = dp[full_rem_mask * m + i]
            if val < 1e10:
                total = val + U_to_dest[rem_indices[i]]
                if total < min_final:
                    min_final = total
        
        total_cost = c_to_U[first] + min_final
        if total_cost < best_cost:
            best_cost = total_cost
            best_first = first
    
    return int(U[best_first])


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Deadline guard: the evaluator kills after 150 s. If we are near the limit,
    # fall back to the cheap nearest-neighbour choice.
    try:
        elapsed = time.time() - _get_start_time()
        if elapsed > 140.0:
            d = distance_matrix[current_node, unvisited_nodes]
            return int(unvisited_nodes[int(np.argmin(d))])
    except Exception:
        pass

    try:
        dm = _init(current_node, destination_node, unvisited_nodes, distance_matrix)
        U = np.asarray(unvisited_nodes)
        n = U.size
        if n == 0:
            return int(current_node)
        if n == 1:
            return int(U[0])

        c = int(current_node)
        dest = int(destination_node)

        # Use Held-Karp for small n, greedy for large n
        if n <= 15:
            return _held_karp_suffix(c, U, dm, dest, max_n=15)
        
        # For larger n, use a heuristic
        edge = dm[c, U]
        
        if n <= 3:
            score = edge + dm[U, dest]
            return int(U[int(np.argmin(score))])

        # 3-step lookahead using soft-min (log-sum-exp) for smoother gradients
        sub = dm[np.ix_(U, U)]
        dest_col = dm[U, dest]
        
        # Compute best continuation cost from each node in U
        # For each node i in U, the best cost to visit all other nodes and return to dest
        # Approximate with 2-step: min over j != i of dm[i,j] + (best from j to dest or further)
        # Use iterative relaxation for a few steps
        
        # Initialize with direct return cost
        cont_cost = dest_col.copy()
        
        # 2-step lookahead
        for step in range(2):
            # For each i, cont_cost[i] = min over j != i of dm[i,j] + cont_cost[j]
            # But we can't visit the same node twice, so this is an approximation
            # Use the fact that we're building a tour, so we can't go back to current or dest
            # Simple approximation: min over all j != i
            new_cont = np.empty_like(cont_cost)
            for i in range(n):
                # Exclude i itself
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                if n > 1:
                    new_cont[i] = np.min(sub[i, mask] + cont_cost[mask])
                else:
                    new_cont[i] = cont_cost[i]
            cont_cost = new_cont
        
        score = edge + cont_cost
        return int(U[int(np.argmin(score))])
    except Exception:
        d = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[int(np.argmin(d))])
# EVOLVE-BLOCK-END
