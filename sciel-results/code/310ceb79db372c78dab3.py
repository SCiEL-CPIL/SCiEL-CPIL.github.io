"""
Regret-2 Constructive Heuristic for TSP.

Strategy:
1. At each step, evaluate each unvisited node 'j' based on a 2-step lookahead cost.
2. Calculate the cost of the best and second-best next steps from 'j'.
3. The 'regret' of not choosing 'j' immediately is the difference between these costs.
4. Select the node with the maximum regret. This prioritizes nodes that have a very specific, cheap connection to another unvisited node, preventing that cheap edge from being "wasted" or made unavailable in later steps.
5. If regrets are tied or zero (e.g., few nodes left), fall back to Nearest Neighbor.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_LIMIT = 140.0  # Safety margin before 150s limit

def _is_deadline_passed():
    return (time.time() - _START_TIME) > _LIMIT

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour using a Regret-2 heuristic.
    
    Args:
    current_node (int): The current node in the tour.
    destination_node (int): The starting node (to close the tour).
    unvisited_nodes (np.ndarray): Array of unvisited node indices.
    distance_matrix (np.ndarray): NxN matrix of distances.
    
    Returns:
    int: The index of the next node to visit.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Safety check for deadline
    if _is_deadline_passed():
        dists_from_current = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists_from_current)
        return int(unvisited_nodes[best_idx])
    
    # Get distances from current node to all unvisited nodes
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # If only 2 nodes, pick the one that leads to a shorter total path (lookahead)
    if n_unvisited == 2:
        j1 = unvisited_nodes[0]
        j2 = unvisited_nodes[1]
        # Path: current -> j1 -> j2 -> dest
        cost1 = distance_matrix[current_node, j1] + distance_matrix[j1, j2] + distance_matrix[j2, destination_node]
        # Path: current -> j2 -> j1 -> dest
        cost2 = distance_matrix[current_node, j2] + distance_matrix[j2, j1] + distance_matrix[j1, destination_node]
        if cost1 <= cost2:
            return int(j1)
        else:
            return int(j2)
    
    # For n_unvisited > 2, use Regret-2
    # We need to find for each j in unvisited_nodes:
    # 1. The nearest neighbor k1 in unvisited_nodes \ {j}
    # 2. The second nearest neighbor k2 in unvisited_nodes \ {j}
    
    # Vectorized approach for efficiency
    # Create a sub-matrix of distances between unvisited nodes
    # Note: distance_matrix is NxN. We need distances between elements of unvisited_nodes.
    
    # Extract sub-matrix
    # This is O(N^2) memory access, which is fine for N=50
    sub_dist = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # Set diagonal to infinity to ignore self-loops
    np.fill_diagonal(sub_dist, np.inf)
    
    # Find min and second min for each row (each candidate j)
    # np.argpartition is efficient for finding k smallest
    # But for 2 smallest, we can sort or use partition
    # Since N is small (50), sorting is fast enough and simpler
    sorted_dists = np.sort(sub_dist, axis=1)
    
    min_dist_1 = sorted_dists[:, 0]
    min_dist_2 = sorted_dists[:, 1]
    
    # The regret is the difference between the 2nd best and best next step from j
    # This represents the "ambiguity" or "alternative cost".
    # A high regret means the best next step is much better than the second.
    # We want to secure this good connection by visiting j now.
    regrets = min_dist_2 - min_dist_1
    
    # We also need to consider the cost of coming to j from current.
    # A pure regret might pick a far-away node if it has a very tight local cluster.
    # We should combine the immediate cost and the regret.
    # A common score: Score = Immediate Cost + Weight * Regret
    # Or, we can filter candidates by immediate cost first?
    
    # Let's try a composite score:
    # We want to minimize immediate cost, but prioritize high regret.
    # If we pick a node with low regret, we are "safe" but might miss a structural opportunity.
    # If we pick a node with high regret, we lock in a structure.
    
    # Let's normalize regrets to be comparable to distances?
    # Or just use a weighted sum.
    # Given the lack of fitting, let's try a simple heuristic:
    # Score = dists_from_current - lambda * regrets
    # We want to minimize Score.
    # High regret -> lower score -> more likely to be picked.
    
    # What is a good lambda?
    # Distances are in some unit. Regrets are in the same unit.
    # Let's try lambda = 1.0.
    # Or, we can rank by regret first, then break ties by distance?
    
    # Let's try: Pick the node that minimizes (dists_from_current - regrets).
    # This effectively rewards nodes that are both close AND have high regret.
    
    scores = dists_from_current - regrets
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
