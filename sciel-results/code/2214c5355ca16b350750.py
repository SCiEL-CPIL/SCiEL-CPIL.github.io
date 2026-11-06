import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 148.0
_SAFETY_MARGIN = 2.0

def _time_exceeded():
    return (time.time() - _START_TIME) > (_DEADLINE - _SAFETY_MARGIN)

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    if _time_exceeded():
        return depot

    if len(unvisited_nodes) == 0:
        return depot

    # Convert inputs to numpy arrays for efficient vectorization
    unvisited_arr = np.array(unvisited_nodes, dtype=int)
    demands_arr = np.array(demands, dtype=float)
    dist_mat = np.array(distance_matrix, dtype=float)

    # 1. Filter feasible nodes
    # demands is a list/array of length N+1
    # unvisited_nodes is a list/array of indices
    feasible_mask = demands_arr[unvisited_arr] <= rest_capacity
    feasible_nodes = unvisited_arr[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return depot

    # If only one feasible, return it
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])

    # 2. Score each feasible candidate using vectorized soft-min lookahead
    # Score = dist(current, j) + alpha * soft_min(dist(j, k) for k in feasible_after_j)
    
    alpha = 0.8 # Weight for lookahead, tuned to balance immediate cost vs future cost
    temperature = 1.0 # Temperature for soft-min (lower is closer to hard min)
    
    # Precompute distances from current to all feasible nodes
    d1_vec = dist_mat[current_node, feasible_nodes]
    
    # Precompute demands for feasible nodes to calculate remaining capacity
    demands_feasible = demands_arr[feasible_nodes]
    new_caps = rest_capacity - demands_feasible
    
    # We need to calculate for each j in feasible_nodes:
    # soft_min over k in (unvisited_nodes \ {j}) where demands[k] <= new_cap[j]
    # plus distance to depot
    
    # Matrix of distances from feasible_nodes (rows) to unvisited_nodes (cols)
    # shape: (num_feasible, num_unvisited)
    dist_feasible_to_unvisited = dist_mat[feasible_nodes, unvisited_arr]
    
    # Matrix of demands for unvisited nodes (cols)
    # shape: (num_unvisited,)
    demands_unvisited = demands_arr[unvisited_arr]
    
    # For each row j, we need to mask out:
    # 1. The column corresponding to j itself (to avoid self-loop)
    # 2. Columns where demand > new_cap[j]
    
    # Create a mask of valid next steps
    # valid_mask[j, k] is True if k != j and demands[k] <= new_cap[j]
    
    # demands_unvisited is (N_unv,), new_caps is (N_feas,)
    # We want (N_feas, N_unv)
    # demands_unvisited[None, :] <= new_caps[:, None]
    capacity_mask = demands_unvisited[None, :] <= new_caps[:, None]
    
    # Exclude self-loops
    # We need to map feasible_nodes indices to their positions in unvisited_arr to zero out the diagonal?
    # No, feasible_nodes is a subset of unvisited_arr.
    # We need to create a mask where dist_feasible_to_unvisited[j, k] is invalid if unvisited_arr[k] == feasible_nodes[j]
    
    # Let's find the indices of feasible_nodes within unvisited_arr
    # Since unvisited_arr is an array of indices, we can use searchsorted or a dictionary.
    # Given N=50, a simple loop or vectorized search is fine.
    
    # Create a lookup array: index_in_unvisited[node_id]
    # Initialize with -1
    lookup = np.full(max(unvisited_arr) + 1, -1, dtype=int)
    for idx, node_id in enumerate(unvisited_arr):
        lookup[node_id] = idx
        
    # Get the column indices for the feasible nodes
    self_col_indices = lookup[feasible_nodes]
    
    # Build the full validity mask
    # Start with capacity mask
    valid_mask = capacity_mask
    
    # Zero out self-loops
    # For each row j, set column self_col_indices[j] to False
    row_indices = np.arange(len(feasible_nodes))
    valid_mask[row_indices, self_col_indices] = False
    
    # Calculate soft-min for each row
    # We add depot distance as a candidate for every row
    # dist_to_depot is dist_mat[feasible_nodes, depot] shape (N_feas,)
    
    # To use log-sum-exp, we need to work in log domain or just use a stable soft-min formula
    # soft_min(x) = -T * log(sum(exp(-x/T)))
    # Let's compute the "cost" matrix including depot
    # Costs shape: (N_feas, N_unv + 1)
    # Last column is distance to depot
    
    dist_to_depot = dist_mat[feasible_nodes, depot]
    
    # Create the cost matrix
    # Invalid entries should be set to a large value (infinity) so they don't contribute to sum(exp(-x/T))
    # exp(-inf) = 0
    
    # Initialize costs with large value
    large_val = 1e9
    costs = np.full((len(feasible_nodes), len(unvisited_arr) + 1), large_val)
    
    # Fill in valid distances
    # valid_mask is (N_feas, N_unv)
    # dist_feasible_to_unvisited is (N_feas, N_unv)
    # We can use np.where(valid_mask, dist_feasible_to_unvisited, large_val)
    costs[:, :-1] = np.where(valid_mask, dist_feasible_to_unvisited, large_val)
    
    # Fill in depot distance
    costs[:, -1] = dist_to_depot
    
    # Compute soft-min
    # soft_min_j = -T * log( sum_k exp(-costs[j, k] / T) )
    # To avoid underflow/overflow, subtract max? No, we are minimizing, so we subtract min?
    # Actually, standard stable log-sum-exp:
    # log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
    # Here we have -1/T * cost. Let y = -cost/T.
    # We want log(sum(exp(y))).
    # y = -costs / T
    y = -costs / temperature
    
    # Max y for numerical stability
    max_y = np.max(y, axis=1, keepdims=True)
    # If all costs are large_val, max_y will be -large_val/T.
    # exp(y - max_y) will be exp(0) = 1 for the max, and exp(negative) for others.
    
    sum_exp = np.sum(np.exp(y - max_y), axis=1)
    log_sum_exp = max_y.flatten() + np.log(sum_exp)
    
    soft_min_costs = -temperature * log_sum_exp
    
    # Final score
    scores = d1_vec + alpha * soft_min_costs
    
    best_idx = np.argmin(scores)
    best_node = feasible_nodes[best_idx]
    
    return int(best_node)
