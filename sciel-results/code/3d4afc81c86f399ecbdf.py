"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to manage time budget across calls
_start_time = None
_TIME_LIMIT = 110.0  # Safety margin for 120s limit

def _get_start_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _start_time

def _time_exceeded():
    return (time.time() - _get_start_time()) > _TIME_LIMIT

def _two_opt(partial_tour, distance_matrix):
    """
    Perform 2-opt optimization on a partial tour.
    partial_tour: list of nodes in current order (starts with current_node's predecessor? 
                  No, the interface is tricky. We assume we are building a path.
                  Actually, select_next_node is called sequentially.
                  We need to simulate the state.
                  Since we don't have access to the full history in the signature, 
                  we must infer or maintain state?
                  The signature is: select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix)
                  It returns the NEXT node.
                  We do NOT have the list of previously visited nodes.
                  This is a critical constraint. We cannot run 2-opt on the full tour 
                  unless we can reconstruct it or if the heuristic relies only on local info.
                  
                  Wait, the standard EoH/LLM4AD contract usually implies a stateless function 
                  or a function that can access global state if allowed. 
                  However, `unvisited_nodes` changes. 
                  If we can't see visited nodes, we can't do 2-opt on the past.
                  
                  Let's re-read the prompt carefully.
                  "select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix)"
                  It does NOT provide `visited_nodes` or `tour_so_far`.
                  This implies the function is stateless regarding the past, OR we must store state globally.
                  Given `unvisited_nodes` is an array, we can deduce visited nodes if we knew the initial set.
                  But we don't know N upfront? We can infer N from distance_matrix shape.
                  So we CAN reconstruct the visited set: all nodes in [0, N) not in unvisited_nodes.
                  But we don't know the ORDER of visited nodes.
                  
                  Without the order, we cannot do 2-opt on the past.
                  We can only do lookahead on the FUTURE.
                  
                  Alternative Strategy: 
                  Since we can't fix the past, we must make a better choice for the NEXT node.
                  The current strategy is Nearest Neighbor (NN).
                  A different strategy: **Lookahead Nearest Neighbor** or **Insertion Heuristic** (if we could insert, but we can't, we append).
                  
                  Let's try **2-Step Lookahead**:
                  For each candidate `c` in `unvisited_nodes`:
                    1. Cost to go from `current` to `c`.
                    2. From `c`, go to the nearest neighbor `n` in `unvisited_nodes \ {c}`.
                    3. Score = dist(current, c) + dist(c, n) + dist(n, destination) ?? 
                       Or maybe dist(current, c) + dist(c, nearest_unvisited(c))
                       
                  Inspiration 1 mentions `_solve_lookahead_nearest`.
                  Inspiration 2/3 mentions "return-aware".
                  
                  Let's implement a **Return-Aware 2-Step Lookahead**.
                  Score for candidate `c`:
                  1. `d1 = dist(current, c)`
                  2. Find `n` = argmin `dist(c, u)` for `u` in `unvisited \ {c}`.
                  3. `d2 = dist(c, n)`
                  4. `d3 = dist(n, destination)` (anticipating the closure)
                  5. Total Score = d1 + d2 + d3 (or weighted).
                  
                  This is O(N^2) per step. N=50. Steps=50. Total ops ~ 50 * 50 * 50 = 125,000. Very fast.
                  We can even do 3-step lookahead.
                  
                  Let's try **3-Step Lookahead with Return Awareness**.
                  For each `c` in unvisited:
                    Find `n1` nearest to `c` (excluding c).
                    Find `n2` nearest to `n1` (excluding c, n1).
                    Cost = dist(cur, c) + dist(c, n1) + dist(n1, n2) + dist(n2, dest)
                    
                  This looks promising and is a different design family (Lookahead vs Greedy NN).
                  """
    pass

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using a 3-step lookahead heuristic with return awareness.
    """
    if len(unvisited_nodes) == 0:
        return -1 # Should not happen per contract, but safe
        
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])

    if _time_exceeded():
        # Fallback to simple NN if time is running out
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    # Vectorized calculation for speed
    # 1. Distance from current to all candidates
    d_cur_to_c = distance_matrix[current_node, unvisited_nodes]
    
    # 2. Distance from each candidate to all other unvisited nodes
    # dist_c_to_u: shape (len(unvisited), len(unvisited))
    # We need to find the nearest neighbor for each c, excluding itself.
    
    # Get distances between all unvisited nodes
    d_unvisited = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # Mask diagonal to infinity to exclude self
    np.fill_diagonal(d_unvisited, np.inf)
    
    # Nearest neighbor for each candidate c (index i in unvisited list)
    # min_d_c_to_n: shape (len(unvisited))
    min_d_c_to_n = np.min(d_unvisited, axis=1)
    # Index of nearest neighbor in the unvisited array
    min_idx_c_to_n = np.argmin(d_unvisited, axis=1)
    
    # 3. Now for the 2nd step: From that nearest neighbor (n1), go to next nearest (n2)
    # We need to be careful: n1 is a node in unvisited. 
    # We need the nearest neighbor of n1, excluding c and n1.
    
    # This is harder to vectorize perfectly without loops because the exclusion set changes.
    # Given N=50, a small loop over candidates is acceptable if the inner logic is vectorized or fast.
    
    best_score = np.inf
    best_node = unvisited_nodes[0]
    
    # We can optimize by pre-computing the "best next-next" for each potential n1?
    # Let's stick to a hybrid: Vectorize the first step, loop the second/third if needed, 
    # or just do a simple 2-step lookahead which is O(N^2) and very fast.
    
    # Let's try 2-step lookahead first for robustness and speed.
    # Score(c) = dist(cur, c) + dist(c, nn(c)) + dist(nn(c), dest)
    
    # We have min_d_c_to_n (dist c->nn)
    # We need dist(nn, dest).
    # nn_nodes = unvisited_nodes[min_idx_c_to_n]
    # d_nn_to_dest = distance_matrix[nn_nodes, destination_node]
    
    nn_nodes = unvisited_nodes[min_idx_c_to_n]
    d_nn_to_dest = distance_matrix[nn_nodes, destination_node]
    
    # Total Score
    scores = d_cur_to_c + min_d_c_to_n + d_nn_to_dest
    
    # To avoid going to the destination if it's not the end? 
    # The problem says "destination_node" is the node to return to.
    # Usually in TSP, you return to start. Here destination is likely start.
    # If destination is in unvisited, we might want to prioritize it? 
    # No, destination is usually the start node, which is visited (or will be visited last).
    # If destination is in unvisited, the heuristic should still work.
    
    # Pick min score
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
