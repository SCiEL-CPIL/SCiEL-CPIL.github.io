import numpy as np
import time

# We don't strictly need a deadline guard for 50 nodes with this O(N^2) per step heuristic,
# but it's good practice to ensure we don't hang if something goes wrong with the input sizes.
# However, the prompt asks to keep the signature and return type.
# The input size is fixed at 50 customers in the description.

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # unvisited_nodes is a list or array of node indices
    if len(unvisited_nodes) == 0:
        return int(depot)
    
    # Convert to numpy array for vectorization
    unvisited = np.array(unvisited_nodes)
    
    # Filter for capacity feasibility
    # demands is an array where demands[i] is the demand of node i
    feasible_mask = demands[unvisited] <= rest_capacity
    feasible_nodes = unvisited[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return int(depot)
    
    # If only one feasible node, must go there
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])
        
    # Get distances from current_node to all feasible nodes
    # distance_matrix is a 2D array
    dist_current_to_feasible = distance_matrix[current_node, feasible_nodes]
    
    # Lookahead: For each feasible node j, find the min distance to any other feasible node or the depot.
    # We want to avoid picking a node that is isolated or far from the rest of the cluster.
    
    # To do this efficiently:
    # 1. Get the submatrix of distances between feasible_nodes and (feasible_nodes U {depot})
    # 2. For each row (representing node j), find the minimum distance.
    #    Note: We must exclude j itself from the min calculation if we are looking at the matrix of feasible nodes.
    #    But since distance to self is 0, we need to be careful.
    
    # Let's construct the lookahead costs.
    # Candidate next nodes for the step after j are: all feasible_nodes except j, plus depot.
    
    # We can compute a matrix of distances between all feasible nodes.
    # D_ff[i, k] = distance(feasible_nodes[i], feasible_nodes[k])
    D_ff = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)]
    
    # Set diagonal to infinity so we don't pick the node itself
    np.fill_diagonal(D_ff, np.inf)
    
    # Distance from each feasible node to depot
    dist_to_depot = distance_matrix[feasible_nodes, depot]
    
    # For each feasible node i (corresponding to feasible_nodes[i]):
    # The best next step cost is min( min(D_ff[i, :]), dist_to_depot[i] )
    # This represents the cost of the *next* edge after visiting j.
    
    min_next_dist_from_feasible = np.min(D_ff, axis=1)
    
    # The actual next step could also be the depot if it's closer than any other feasible node
    lookahead_cost = np.minimum(min_next_dist_from_feasible, dist_to_depot)
    
    # Total score for picking feasible_nodes[i] is:
    # dist(current, feasible_nodes[i]) + lookahead_cost[i]
    # We use a weight of 1.0 for the lookahead term. This is a standard 2-step greedy.
    # Sometimes a weight < 1.0 (like 0.5) works better to prioritize immediate proximity,
    # but 1.0 is a strong baseline for "avoiding dead ends".
    
    scores = dist_current_to_feasible + lookahead_cost
    
    # Find the index of the minimum score
    best_idx = np.argmin(scores)
    
    return int(feasible_nodes[best_idx])
