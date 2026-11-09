"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None
BEAM_WIDTH = 50

def _get_time_remaining():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elapsed = time.time() - _start_time
    return max(0.0, 145.0 - elapsed)

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Time guard
    if _get_time_remaining() < 2.0:
        # Fallback to simple NN
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n = len(unvisited_nodes)
    if n == 0:
        return -1
    if n == 1:
        return int(unvisited_nodes[0])
    
    # Beam Search Strategy
    # We maintain a list of states. Each state is (path_nodes, cost_so_far).
    # Since we don't have access to the full history in the function signature,
    # we must simulate the beam search internally using the available info.
    # 
    # Wait, the function signature is:
    # select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix)
    # It does NOT provide the history of visited nodes.
    # This makes a true beam search over the *path* impossible without storing state 
    # externally or assuming we can reconstruct it. 
    # 
    # However, we CAN do a "Lookahead" beam search on the *next few steps* locally.
    # We can simulate extending the path from `current_node` by 2 or 3 steps 
    # and evaluate the cost of those extensions.
    # 
    # Let's implement a 2-step Lookahead (Greedy with 2-step evaluation).
    # For each candidate v1 in unvisited:
    #   cost1 = d(current, v1)
    #   For each candidate v2 in unvisited \ {v1}:
    #     cost2 = cost1 + d(v1, v2) + heuristic(v2)
    #   best_next_for_v1 = v1 that minimizes min_{v2} (cost1 + d(v1, v2) + d(v2, dest))
    # 
    # This is O(N^2) per step. For N=50, N^2=2500. 50 steps * 2500 = 125,000 ops.
    # This is very fast in Python/NumPy if vectorized.
    
    # Vectorized 2-Step Lookahead
    # 1. Get distances from current to all unvisited
    d_cur_to_unvisited = distance_matrix[current_node, unvisited_nodes]
    
    # 2. Get distances from destination to all unvisited (heuristic)
    d_dest_to_unvisited = distance_matrix[destination_node, unvisited_nodes]
    
    # 3. We want to pick v1 that minimizes:
    #    d(cur, v1) + min_{v2 != v1} (d(v1, v2) + d(v2, dest))
    
    # Let's precompute the "best continuation cost" for each potential v1.
    # For a fixed v1, the best v2 is the one that minimizes d(v1, v2) + d(v2, dest).
    # We can compute this for all v1 simultaneously using matrix operations.
    
    # Submatrix of distances between unvisited nodes
    # D_unvisited[i, j] = d(unvisited[i], unvisited[j])
    D_unvisited = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # For each i (representing v1), we want min_j (D_unvisited[i, j] + d_dest_to_unvisited[j])
    # Note: j must be != i. We can set diagonal to inf.
    D_unvisited_diag_inf = D_unvisited.copy()
    np.fill_diagonal(D_unvisited_diag_inf, np.inf)
    
    # best_continuation[i] = min_j (D_unvisited_diag_inf[i, j] + d_dest_to_unvisited[j])
    # This is a matrix-vector addition followed by row-wise min.
    # We can do: (D_unvisited_diag_inf + d_dest_to_unvisited[None, :]).min(axis=1)
    
    best_continuation_cost = (D_unvisited_diag_inf + d_dest_to_unvisited[None, :]).min(axis=1)
    
    # Total estimated cost for picking unvisited[i] as the next node:
    # total_cost[i] = d_cur_to_unvisited[i] + best_continuation_cost[i]
    
    total_cost = d_cur_to_unvisited + best_continuation_cost
    
    # Pick the index with minimum total cost
    best_idx = np.argmin(total_cost)
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
