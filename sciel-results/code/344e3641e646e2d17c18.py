"""Traveling salesman constructive heuristic with 3-step lookahead."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None
TIME_LIMIT = 148.0

def _get_time_remaining():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elapsed = time.time() - _start_time
    return max(0.0, TIME_LIMIT - elapsed)

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Time guard: fallback to nearest neighbor if time is running out
    if _get_time_remaining() < 1.0:
        if len(unvisited_nodes) == 0:
            return -1
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n = len(unvisited_nodes)
    if n == 0:
        return -1
    if n == 1:
        return int(unvisited_nodes[0])
    if n == 2:
        # With 2 nodes left, pick the one that gives shorter total path back to destination
        v0, v1 = unvisited_nodes[0], unvisited_nodes[1]
        cost0 = distance_matrix[current_node, v0] + distance_matrix[v0, v1] + distance_matrix[v1, destination_node]
        cost1 = distance_matrix[current_node, v1] + distance_matrix[v1, v0] + distance_matrix[v0, destination_node]
        return int(unvisited_nodes[0] if cost0 <= cost1 else unvisited_nodes[1])
    
    # Get relevant distance vectors
    d_cur = distance_matrix[current_node, unvisited_nodes]       # (n,)
    d_dest = distance_matrix[destination_node, unvisited_nodes]  # (n,)
    
    # Submatrix of distances between unvisited nodes
    D = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # 3-step lookahead: for each candidate i as first choice,
    # estimate the best 2-step continuation cost
    # For fixed i, we want min_{j != i} [d(i,j) + min_{k != i,j} (d(j,k) + d(k,dest))]
    # This is expensive at O(n^3), but n <= 50 so n^3 = 125000, and we do this per step (50 steps)
    # Total ~6.25M operations, which is fine.
    
    # Precompute for each pair (j,k) the cost d(j,k) + d(k,dest) for k != j
    # Then for each i, find min over j != i of [d(i,j) + min_{k != i,j}(d(j,k) + d(k,dest))]
    
    # Step 1: For each j, compute best_second_cost[j] = min_{k != j} (D[j,k] + d_dest[k])
    # Set diagonal to inf so k != j
    D_copy = D.copy()
    np.fill_diagonal(D_copy, np.inf)
    # best_from_j[j] = min_k (D[j,k] + d_dest[k]) where k != j
    # Vectorized: (D_copy + d_dest[None,:]).min(axis=1)
    best_from_j = (D_copy + d_dest[None, :]).min(axis=1)  # (n,)
    
    # Step 2: For each i, we need min_{j != i} [d_cur[i] + D[i,j] + adjusted_best_from_j[j] excluding i]
    # But best_from_j[j] might have chosen k=i, which is not allowed if i is already taken.
    # We need: for fixed i, for each j != i, compute min_{k != i, k != j} (D[j,k] + d_dest[k])
    # Then total for i = d_cur[i] + min_{j != i} (D[i,j] + min_{k != i, j} (D[j,k] + d_dest[k]))
    
    # This is O(n^3). Let's do it with a loop over i, which is n iterations,
    # and inside each we do O(n^2) work.
    
    # Alternative: use a smarter approach.
    # For each j, we have two best candidates k for minimizing D[j,k]+d_dest[k].
    # When we exclude i, if the best k was i, we use the second best.
    
    # Let's compute for each j the top-2 smallest values of (D[j,k] + d_dest[k]) and their indices.
    
    # cost_matrix[j,k] = D[j,k] + d_dest[k] for k != j
    cost_mat = D_copy + d_dest[None, :]  # (n, n), diagonal is inf
    
    # For each row j, find the two smallest values and their indices
    # We can use argpartition or sort
    # For n=50, sorting each row is fine: n * n log n = 50 * 50 * 6 = 15000 ops
    
    # Get indices of top 2
    # Use argsort and take first 2
    sorted_idx = np.argsort(cost_mat, axis=1)  # (n, n)
    top2_idx = sorted_idx[:, :2]  # (n, 2)
    top2_vals = cost_mat[np.arange(n)[:, None], top2_idx]  # (n, 2)
    
    # Now for each pair (i, j) with i != j:
    # If top2_idx[j,0] == i, then best_excluding_i[j] = top2_vals[j,1]
    # Else best_excluding_i[j] = top2_vals[j,0]
    
    # We want: for each i, min_{j != i} (D[i,j] + best_excluding_i_j)
    # Let's compute a matrix M where M[i,j] = D[i,j] + best_excluding_i(j)
    # where best_excluding_i(j) depends on i and j.
    
    # best_excluding[i,j] = top2_vals[j,0] if top2_idx[j,0] != i else top2_vals[j,1]
    
    # Create a matrix of shape (n, n): for each i (rows) and j (cols)
    # first_choice = top2_idx[:, 0]  # (n,)
    # For each i, j: if first_choice[j] == i, use second, else use first
    
    # Vectorized approach:
    # create a boolean mask: mask[i,j] = (top2_idx[j, 0] == i)
    # That is: top2_idx is (n,2), top2_idx[:,0] is (n,). 
    # mask[i,j] = (top2_idx[j,0] == i)
    # This is equivalent to: mask = (top2_idx[:,0][None, :] == np.arange(n)[:, None])
    # Shape: (n, n)
    
    first_idx = top2_idx[:, 0]  # (n,)
    first_val = top2_vals[:, 0]  # (n,)
    second_val = top2_vals[:, 1]  # (n,)
    
    # mask[i, j] = True if first_idx[j] == i
    mask = (first_idx[None, :] == np.arange(n)[:, None])  # (n, n)
    
    # best_excl[i, j] = first_val[j] if not mask[i,j] else second_val[j]
    best_excl = np.where(mask, second_val[None, :], first_val[None, :])  # (n, n)
    
    # Now M[i,j] = D[i,j] + best_excl[i,j] for i != j, and inf for i == j
    M = D + best_excl
    np.fill_diagonal(M, np.inf)  # i != j
    
    # For each i: total_cost[i] = d_cur[i] + min_j M[i,j]
    min_continuation = M.min(axis=1)  # (n,)
    total_cost = d_cur + min_continuation
    
    best_idx = np.argmin(total_cost)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
