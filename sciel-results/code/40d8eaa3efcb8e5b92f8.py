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
        # Too many nodes for exact DP, fall back to greedy
        d = distance_matrix[c, U]
        return int(U[int(np.argmin(d))])
    
    # Map unvisited nodes to indices 0..n-1
    # dp[mask][i] = min cost to start at c, visit set 'mask' of unvisited nodes,
    # and end at unvisited node i (index in U)
    # We need to visit all n nodes, so full mask is (1 << n) - 1
    
    # For n up to 15, this is 2^15 * 15 = 491520 states, manageable
    # But we only need the first step, so we can use a different approach:
    # For each candidate first node u, compute the optimal cost of visiting
    # all remaining nodes and returning to dest.
    
    # Actually, let's do full Held-Karp to find the best first node
    # dp[mask][i] where mask is subset of {0,...,n-1} and i is the last visited node
    # Initialize: dp[1<<i][i] = dm[c, U[i]]
    # Transition: dp[mask | (1<<j)][j] = min(dp[mask][i] + dm[U[i], U[j]])
    # Answer: min over i of dp[(1<<n)-1][i] + dm[U[i], dest]
    
    # To save memory and time, we'll use a list of arrays
    full_mask = (1 << n) - 1
    
    # dp[mask] is an array of size n, dp[mask][i] = min cost to reach state (mask, i)
    # Initialize with inf
    dp = [np.full(n, 1e10) for _ in range(1 << n)]
    
    # Base case: start at c, go to U[i]
    for i in range(n):
        dp[1 << i][i] = distance_matrix[c, U[i]]
    
    # Iterate over all masks
    for mask in range(1, 1 << n):
        # Skip masks with only one bit set (already initialized)
        # Actually, process all masks
        for i in range(n):
            if not (mask & (1 << i)):
                continue
            cost_i = dp[mask][i]
            if cost_i >= 1e10:
                continue
            # Try to go to j not in mask
            remaining = full_mask ^ mask
            for j in range(n):
                if remaining & (1 << j):
                    new_mask = mask | (1 << j)
                    new_cost = cost_i + distance_matrix[U[i], U[j]]
                    if new_cost < dp[new_mask][j]:
                        dp[new_mask][j] = new_cost
    
    # Find the best first node
    # The first node is U[i] where dp[1<<i][i] is part of the optimal path
    # Actually, we need to find which i gives the optimal total cost
    # Total cost for starting with U[i] is the min over all paths that start with i
    # This is tricky to extract directly from dp.
    
    # Alternative: for each candidate first node i, compute the optimal cost
    # of the remaining path (visiting all nodes except i and returning to dest)
    # starting from U[i].
    
    # Let's do a separate DP for each candidate first node
    # But that would be n times the work.
    
    # Better approach: modify the DP to track the first node
    # Or, simpler: for each candidate first node i, run a smaller DP
    # on the remaining n-1 nodes.
    
    # Since n <= 15, n-1 <= 14, so 2^14 * 14 = 229376 states per candidate
    # Total: 15 * 229376 = ~3.4M operations, which is fine.
    
    best_cost = 1e10
    best_first = 0
    
    for first in range(n):
        # Remaining nodes
        rem_indices = [i for i in range(n) if i != first]
        m = len(rem_indices)  # m = n - 1
        
        if m == 0:
            # Only one node, cost is dm[c, U[first]] + dm[U[first], dest]
            cost = distance_matrix[c, U[first]] + distance_matrix[U[first], dest]
            if cost < best_cost:
                best_cost = cost
                best_first = first
            continue
        
        # DP for remaining nodes
        # dp2[mask][i] where mask is subset of rem_indices, i is index in rem_indices
        # rem_indices maps to 0..m-1
        rem_mask_size = 1 << m
        dp2 = [np.full(m, 1e10) for _ in range(rem_mask_size)]
        
        # Base case: start from U[first], go to U[rem_indices[i]]
        for i in range(m):
            dp2[1 << i][i] = distance_matrix[U[first], U[rem_indices[i]]]
        
        # Iterate
        for mask in range(1, rem_mask_size):
            for i in range(m):
                if not (mask & (1 << i)):
                    continue
                cost_i = dp2[mask][i]
                if cost_i >= 1e10:
                    continue
                remaining = (rem_mask_size - 1) ^ mask
                for j in range(m):
                    if remaining & (1 << j):
                        new_mask = mask | (1 << j)
                        new_cost = cost_i + distance_matrix[U[rem_indices[i]], U[rem_indices[j]]]
                        if new_cost < dp2[new_mask][j]:
                            dp2[new_mask][j] = new_cost
        
        # Final cost: min over i of dp2[full][i] + dm[U[rem_indices[i]], dest]
        full_rem_mask = rem_mask_size - 1
        min_final = 1e10
        for i in range(m):
            if dp2[full_rem_mask][i] < 1e10:
                total = dp2[full_rem_mask][i] + distance_matrix[U[rem_indices[i]], dest]
                if total < min_final:
                    min_final = total
        
        total_cost = distance_matrix[c, U[first]] + min_final
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
        # Nearest neighbor with lookahead
        edge = dm[c, U]
        
        if n <= 3:
            score = edge + dm[U, dest]
            return int(U[int(np.argmin(score))])

        # 2-step lookahead
        sub = dm[np.ix_(U, U)]
        dest_col = dm[U, dest]
        extended = sub + dest_col[np.newaxis, :]
        np.fill_diagonal(extended, 1e10)
        best_cont = extended.min(axis=1)
        direct_return = dest_col
        lookahead = np.minimum(best_cont, direct_return)
        
        alpha = 0.7
        score = edge + alpha * lookahead
        return int(U[int(np.argmin(score))])
    except Exception:
        d = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[int(np.argmin(d))])
# EVOLVE-BLOCK-END
