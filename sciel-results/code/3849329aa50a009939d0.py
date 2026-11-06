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


def _held_karp_best_first(c, U, dm, dest):
    """
    Compute the best first node in the optimal path from c, visiting all U, ending at dest.
    Uses Held-Karp DP for the suffix after the first node.
    Only called when len(U) is small (<= 12).
    """
    n = U.size
    if n == 1:
        return int(U[0])
    
    # Precompute sub-matrix and dest costs for U
    # U is np.array
    sub = dm[np.ix_(U, U)]
    dest_costs = dm[U, dest]
    
    best_total = float('inf')
    best_first_idx = 0
    
    for first in range(n):
        # Identify remaining nodes
        rem_indices = [i for i in range(n) if i != first]
        m = len(rem_indices)
        
        # Cost to go from c to U[first]
        cost_start = dm[c, U[first]]
        
        if m == 0:
            total_cost = cost_start + dm[U[first], dest]
            if total_cost < best_total:
                best_total = total_cost
                best_first_idx = first
            continue
        
        # DP for the remaining m nodes
        # dp[mask][i] = min cost to start at U[first], visit set 'mask' of rem_indices,
        # and end at rem_indices[i]
        size = 1 << m
        # Initialize dp array. Use a list of lists.
        # dp[mask] is a list of m floats, or None if unreachable
        dp = [None] * size
        
        # Base case: start at U[first], go to U[rem_indices[i]]
        for i in range(m):
            mask = 1 << i
            dp[mask] = [1e10] * m
            dp[mask][i] = dm[U[first], U[rem_indices[i]]]
        
        # Iterate over all masks
        for mask in range(1, size):
            row = dp[mask]
            if row is None:
                continue
            # Iterate over i in mask
            for i in range(m):
                if not (mask & (1 << i)):
                    continue
                ci = row[i]
                if ci >= 1e10:
                    continue
                
                # Try to go to j not in mask
                remaining = (size - 1) ^ mask
                for j in range(m):
                    if remaining & (1 << j):
                        new_mask = mask | (1 << j)
                        if dp[new_mask] is None:
                            dp[new_mask] = [1e10] * m
                        # Update cost
                        new_cost = ci + dm[U[rem_indices[i]], U[rem_indices[j]]]
                        if new_cost < dp[new_mask][j]:
                            dp[new_mask][j] = new_cost
        
        # Finalize: find min cost to end at any node in rem and go to dest
        full_mask = size - 1
        min_suffix_cost = 1e10
        if dp[full_mask] is not None:
            final_row = dp[full_mask]
            for i in range(m):
                if final_row[i] < 1e10:
                    total_end = final_row[i] + dm[U[rem_indices[i]], dest]
                    if total_end < min_suffix_cost:
                        min_suffix_cost = total_end
        
        total_cost = cost_start + min_suffix_cost
        if total_cost < best_total:
            best_total = total_cost
            best_first_idx = first
    
    return int(U[best_first_idx])


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

        # Use Held-Karp for small n (<= 12) to ensure optimal suffix
        # 12 is chosen to balance time (approx 1s per call) and quality.
        if n <= 12:
            return _held_karp_best_first(c, U, dm, dest)
        
        # For larger n, use the fast heuristic from the current program
        # 2-step lookahead: for each candidate u, consider the best next step
        # from u to another unvisited node v (v != u), plus eventual return to dest.
        # Cost(u) = dm[c,u] + min_{v != u} (dm[u,v] + dm[v,dest])
        
        # Build sub-matrix of unvisited-to-unvisited distances
        sub = dm[np.ix_(U, U)]

        # For each row u, find min over v != u of (sub[u,v] + dm[v,dest])
        # Add dest cost to each column of sub, then mask diagonal
        dest_col = dm[U, dest]  # shape (n,)
        extended = sub + dest_col[np.newaxis, :]  # shape (n, n): sub[u,v] + dest_cost[v]
        np.fill_diagonal(extended, 1e10)  # mask self-loops

        # For each candidate u, best continuation cost = min over v of extended[u,v]
        best_cont = extended.min(axis=1)  # shape (n,)

        # Also include direct return to dest as an option
        direct_return = dest_col  # shape (n,)

        # Take the minimum of best continuation and direct return
        lookahead = np.minimum(best_cont, direct_return)

        # Combine: immediate edge + weighted lookahead
        # Weight balances exploration vs exploitation
        alpha = 0.7  # weight on lookahead component
        score = dm[c, U] + alpha * lookahead

        return int(U[int(np.argmin(score))])
    except Exception:
        d = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[int(np.argmin(d))])
# EVOLVE-BLOCK-END
