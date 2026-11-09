"""Improved heuristic for capacitated vehicle routing."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Select the next node to visit using a composite scoring heuristic.
    
    The score for each feasible candidate node combines:
    1. Immediate distance from current node (proximity)
    2. Savings relative to depot return (route compactness)
    3. Demand-weighted centrality among remaining feasible nodes
    
    Returns 0 (depot) to close the route if no feasible nodes remain
    or if returning to depot is optimal.
    """
    start_time = time.time()
    deadline = 0.5  # Safety margin within the 150s budget
    
    unvisited = np.asarray(unvisited_nodes)
    if len(unvisited) == 0:
        return depot
    
    # Filter feasible nodes by capacity
    feasible_mask = np.asarray(demands)[unvisited] <= rest_capacity
    feasible = unvisited[feasible_mask]
    
    if len(feasible) == 0:
        return depot
    
    # If only one feasible node, go there
    if len(feasible) == 1:
        return int(feasible[0])
    
    # Check if we should return to depot based on remaining capacity
    # and whether remaining nodes are far
    dist_to_depot = distance_matrix[current_node, depot]
    
    # Get distances from current to feasible nodes
    dists_to_feasible = distance_matrix[current_node, feasible]
    
    # Compute savings: going to node i instead of returning to depot
    # Savings = dist(current, depot) + dist(depot, i) - dist(current, i)
    # But we also need to consider that after visiting i, we still need to return
    # A better measure: how much does visiting i reduce the total tour length?
    # Approximate: savings_i = dist(current, depot) - dist(current, i) + dist(i, depot)
    # If savings > 0, visiting i is better than closing now
    
    dists_feasible_to_depot = distance_matrix[feasible, depot]
    
    # Composite score for each feasible node:
    # We want to minimize a cost that balances:
    # - Distance from current (lower is better)
    # - Route compactness: nodes that are close to the centroid of remaining feasible set
    # - Capacity utilization: prefer nodes that allow more nodes to be visited later
    
    # Compute demand-weighted centroid of feasible nodes
    demands_feasible = np.asarray(demands)[feasible]
    
    # Use distance matrix to compute a proxy for spatial position
    # Since we don't have coordinates, use distances to depot as a proxy
    # and distances between nodes to compute centrality
    
    # For each feasible node, compute its "centrality" among feasible nodes
    # A node is central if it has low average distance to other feasible nodes
    n_feasible = len(feasible)
    
    if n_feasible <= 20:
        # Compute pairwise distances among feasible nodes
        feasible_indices = feasible.astype(int)
        sub_matrix = distance_matrix[np.ix_(feasible_indices, feasible_indices)]
        # Average distance from each node to all other feasible nodes
        avg_dists = sub_matrix.sum(axis=1) / max(n_feasible - 1, 1)
    else:
        # For larger sets, use a simpler approximation
        avg_dists = dists_feasible_to_depot  # Proxy: distance to depot
    
    # Normalize components to [0, 1]
    dists_norm = (dists_to_feasible - dists_to_feasible.min()) / (dists_to_feasible.max() - dists_to_feasible.min() + 1e-9)
    
    savings = dist_to_depot - dists_to_feasible + dists_feasible_to_depot
    savings_min = savings.min()
    savings_max = savings.max()
    savings_norm = (savings - savings_min) / (savings_max - savings_min + 1e-9) if savings_max > savings_min else np.ones_like(savings)
    
    avg_dists_min = avg_dists.min()
    avg_dists_max = avg_dists.max()
    centrality_norm = 1.0 - (avg_dists - avg_dists_min) / (avg_dists_max - avg_dists_min + 1e-9) if avg_dists_max > avg_dists_min else np.ones_like(avg_dists)
    
    # Capacity utilization factor: prefer nodes with lower demand when many remain,
    # higher demand when few remain (to pack the route)
    remaining_capacity_after = rest_capacity - demands_feasible
    # Normalize: higher remaining capacity after visit is generally better
    cap_util_norm = remaining_capacity_after / (rest_capacity + 1e-9)
    
    # Weighted composite score (lower is better)
    # Weights: proximity, savings, centrality, capacity
    w_prox = 0.35
    w_sav = 0.30
    w_cent = 0.20
    w_cap = 0.15
    
    # For proximity and centrality, lower normalized value is better
    # For savings, higher is better, so invert
    # For capacity, higher remaining is better, so invert
    
    score = (w_prox * dists_norm 
             + w_sav * (1.0 - savings_norm) 
             + w_cent * (1.0 - centrality_norm) 
             + w_cap * (1.0 - cap_util_norm))
    
    # Check deadline
    if time.time() - start_time > deadline:
        # Fallback: nearest feasible
        return int(feasible[np.argmin(dists_to_feasible)])
    
    # Select the node with minimum composite score
    best_idx = np.argmin(score)
    best_node = int(feasible[best_idx])
    
    # Check if returning to depot might be better
    # If the best node's score is high (bad), consider closing
    # Compare: cost of continuing vs cost of closing
    # Cost of closing: dist_to_depot
    # Cost of continuing: dists_to_feasible[best_idx] + expected remaining cost
    # Simple heuristic: if savings for best node is very negative, close
    
    if savings[best_idx] < 0 and dists_to_feasible[best_idx] > dist_to_depot * 1.5:
        # Visiting this node is worse than just going back
        return depot
    
    return best_node
# EVOLVE-BLOCK-END
