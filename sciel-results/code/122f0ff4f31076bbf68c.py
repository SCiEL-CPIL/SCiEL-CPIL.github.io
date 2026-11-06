"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# We will use a regret-based heuristic combined with capacity awareness.
# The idea: for each feasible customer, compute the "regret" of not visiting them now.
# Regret is the difference between the best and second-best "future cost" if we skip them.
# However, a simpler and often effective proxy is: 
#   score = distance_to_depot - savings
# where savings = dist(current, i) + dist(i, depot) - dist(current, depot)
# This is essentially the Clarke-Wright savings idea but evaluated from the current node.
# Actually, let's think about the marginal cost of adding node i to the current route vs. closing the route now.
# If we close the route now, cost = dist(current, depot).
# If we go to i, cost = dist(current, i) + dist(i, depot) (assuming we close after i, which is a local approximation).
# The "extra cost" of visiting i is: dist(current, i) + dist(i, depot) - dist(current, depot).
# We want to minimize this extra cost. 
# But we also need to ensure we don't leave nodes behind that are hard to reach later.
# A common heuristic is to minimize: dist(current, i) + lambda * dist(i, depot)
# Let's try a weighted sum: score = dist(current, i) + alpha * dist(i, depot)
# And enforce capacity: demands[i] <= rest_capacity.

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    if not unvisited_nodes:
        return 0

    # Filter feasible nodes by capacity
    feasible_mask = np.array([demands[node] <= rest_capacity for node in unvisited_nodes])
    if not np.any(feasible_mask):
        # If no feasible node, must close route
        return 0
    
    feasible_nodes = unvisited_nodes[feasible_mask]
    
    # Calculate distances from current node to feasible nodes
    dists_current = distance_matrix[current_node, feasible_nodes]
    dists_depot = distance_matrix[depot, feasible_nodes]
    
    # We want to minimize a score.
    # Let's try: score = dists_current + 0.5 * dists_depot
    # The 0.5 weight balances immediate distance with the cost of returning to depot later.
    # This encourages nodes that are close to current AND close to depot (or at least not too far from depot).
    
    # Alternative: Regret-based
    # For each node i, regret = (dist(current, i) + 0.5*dist(depot, i)) - min_{j != i} (dist(current, j) + 0.5*dist(depot, j))
    # This prioritizes nodes that are uniquely good.
    
    # Let's stick to a simple weighted sum first, as regret can be noisy.
    # But the prompt suggests regret-based. Let's implement a simple regret.
    
    scores = dists_current + 0.5 * dists_depot
    
    # Calculate regret
    # Sort scores to find min and second min
    sorted_scores = np.sort(scores)
    min_score = sorted_scores[0]
    # If only one feasible node, regret is 0 (or high, doesn't matter, we must pick it)
    if len(feasible_nodes) == 1:
        best_idx = 0
    else:
        second_min_score = sorted_scores[1]
        regret = scores - second_min_score
        # We want to maximize regret? No, regret is the loss of NOT picking the best alternative.
        # Actually, standard regret heuristics pick the node with the HIGHEST regret.
        # Regret(i) = best_cost_if_i_skipped - best_cost_if_i_picked.
        # If we pick i, cost is score(i). If we skip i, the best we can do is the min of the others.
        # So Regret(i) = min_{j != i} score(j) - score(i).
        # We want to maximize Regret(i).
        # This means we pick the node where score(i) is significantly lower than the next best option.
        
        # Let's compute this properly
        # For each i, min_{j != i} score(j)
        # This is O(N^2) but N is small (<=50).
        
        regret_values = np.zeros(len(feasible_nodes))
        for idx in range(len(feasible_nodes)):
            others = np.delete(scores, idx)
            regret_values[idx] = np.min(others) - scores[idx]
        
        best_idx = np.argmax(regret_values)
        
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
