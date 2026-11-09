"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# We use a module-level cache to store state across calls for the same instance.
# However, since we don't know if the evaluator resets state between instances or not,
# and the function signature is stateless, we must be careful.
# The prompt says: "scored by mean closed-tour length on eight seeded uniform 50-city instances".
# Usually, these evaluators call the function 50 times per instance (once per city added to the tour).
# If we can detect that we are in a new instance, we can reset our internal state.
# How to detect a new instance?
# 1. The distance_matrix object id might change.
# 2. The current_node might reset to 0 (or some start node).
# 3. The unvisited_nodes set might reset to all nodes except the start.

# Let's assume the standard TSP constructive heuristic loop:
# start at node 0.
# unvisited = [1..N-1]
# for step in 1..N-1:
#    next = select_next_node(current, dest, unvisited, dist)
#    current = next
#    unvisited.remove(next)
# close the tour by returning to 0 (or the destination_node if it's the start).

# We will implement a "Lookahead" or "2-Opt-like" local improvement within the selection step.
# Since we can only pick ONE node, we can't do full 2-opt.
# But we can pick the node that minimizes the sum of:
# 1. dist(current, candidate)
# 2. dist(candidate, best_next_after_candidate)  -- this is a 1-step lookahead
# Or better:
# dist(current, candidate) + dist(candidate, destination_node)  -- if destination is the final return
# But destination_node is passed in. In standard TSP, destination is often the start node (0) for the final step, or it might be the "next" node in a path.
# Given the signature: select_next_node(current_node, destination_node, ...)
# If this is a path problem, destination is fixed. If TSP, destination is likely the start node to close the loop, or just a dummy.
# Let's assume standard TSP where we return to 0 at the end.
# The "destination_node" argument is tricky. In many TSP heuristics, the "destination" is the node we ultimately want to return to.
# Let's assume destination_node is the node we must eventually return to (likely 0).

# Strategy:
# We will use a "Nearest Neighbor with Lookahead" approach.
# For each candidate in unvisited_nodes:
#   cost = dist(current, candidate)
#   # Lookahead: if this was the last node, we'd go to destination.
#   # If not, we go to the nearest neighbor of candidate among the remaining unvisited nodes (excluding candidate).
#   # This is O(N^2) per step, which is 50^2 * 50 = 125,000 ops per instance. Very fast.
#
# Let's refine the lookahead:
# score(candidate) = dist(current, candidate) + min(dist(candidate, u) for u in unvisited_nodes if u != candidate)
# If unvisited_nodes has only 1 element, the lookahead is dist(candidate, destination_node).
#
# This is a standard improvement over pure NN.

# We also need to handle the case where we are on the last step.
# If len(unvisited_nodes) == 1, we must pick that node.

# Let's also consider the "Savings Heuristic" idea, but that's hard to do incrementally in a single node selection function without global state.
# The lookahead NN is a good balance.

# To further improve, we can try a "Weighted" lookahead:
# score = alpha * dist(current, candidate) + (1-alpha) * lookahead_cost
# But simple sum is usually good.

# Let's try to detect the instance ID to cache precomputed values?
# Actually, 50 cities is small enough that O(N^2) per step is fine.
# Total steps: 49.
# Total ops: 49 * 50 * 50 ~ 120,000.
# This will run in milliseconds.

# Let's implement:
# 1. If only 1 unvisited node, return it.
# 2. For each candidate c in unvisited:
#    a. d1 = dist(current, c)
#    b. If len(unvisited) > 1:
#         remaining = unvisited without c
#         d2 = min(dist(c, u) for u in remaining)
#       Else:
#         d2 = dist(c, destination_node)
#    c. total = d1 + d2
# 3. Return argmin total.

# Is there a better heuristic?
# "Cheapest Insertion" is global.
# "2-Opt" is post-processing.
# Since we are constrained to pick one node, Lookahead NN is the strongest standard constructive heuristic.

# Let's also check if `destination_node` is actually the start node (0).
# If the problem is a TSP, the tour is closed.
# The cost of the final edge is dist(last_node, start_node).
# So for the last candidate, the lookahead should be dist(candidate, start_node).
# For intermediate candidates, the lookahead is the distance to the next nearest node.

# Let's code this.

_start_time = time.time()
_DEADLINE = 145  # 5s margin from 150s

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _start_time, _DEADLINE
    
    # Deadline guard
    if time.time() > _DEADLINE:
        # Fallback to nearest neighbor to ensure we finish
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        # Should not happen if called correctly, but return current to be safe? 
        # The contract says return int. If no nodes, maybe return current? 
        # But usually this function is called only when unvisited is not empty.
        return current_node

    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    # Pre-fetch distances from current to all unvisited
    # dist_current = distance_matrix[current_node, unvisited_nodes]
    
    # For each candidate, we need:
    # 1. dist(current, candidate)
    # 2. min(dist(candidate, other_unvisited))
    
    # We can compute the matrix of distances among unvisited nodes once?
    # unvisited_nodes is an array of indices.
    # sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    # This is 50x50 = 2500 elements. Very fast.
    
    sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    dist_current = distance_matrix[current_node, unvisited_nodes]
    
    # For each candidate i in 0..n_unvisited-1:
    # cost_i = dist_current[i] + min(sub_matrix[i, j] for j != i)
    
    # We can vectorize the min part?
    # For each row i, we need the minimum of the row excluding the diagonal element i.
    # We can set the diagonal to infinity, then take row min.
    
    # Create a copy of sub_matrix with diagonal set to inf
    # Note: sub_matrix is symmetric.
    
    # Efficient way:
    # 1. Set diagonal to inf
    # 2. Take min along axis=1
    # 3. Add dist_current
    
    # But wait, if n_unvisited is small, this is fine.
    # Let's do it.
    
    # We must be careful not to modify the original sub_matrix if it's a view?
    # np.ix_ returns a copy usually, or we can just create a new array.
    # Let's use np.fill_diagonal on a copy.
    
    min_next_costs = np.empty(n_unvisited)
    
    # Optimization: If n_unvisited is very small, loop is fine.
    # If large, vectorize.
    # 50 is small enough that a loop might be faster due to overhead of fill_diagonal?
    # Let's try vectorized first.
    
    # Copy sub_matrix to modify diagonal
    # Actually, we can just use a mask or simple logic.
    
    # For each i:
    #   row = sub_matrix[i]
    #   row[i] = inf
    #   min_next_costs[i] = np.min(row)
    
    # This is O(N^2) in Python loop, which is 2500 iterations. Fine.
    # But we can do it faster with numpy.
    
    # Let's use the fill_diagonal trick.
    # We need a temporary array.
    tmp = sub_matrix.copy()
    np.fill_diagonal(tmp, np.inf)
    min_next_costs = np.min(tmp, axis=1)
    
    total_costs = dist_current + min_next_costs
    
    best_idx = np.argmin(total_costs)
    
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
