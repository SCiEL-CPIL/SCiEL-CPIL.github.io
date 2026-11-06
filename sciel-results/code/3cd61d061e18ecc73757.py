"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for deadline tracking
_start_time = None
_deadline = 145.0  # 5s buffer before 150s limit

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    global _start_time
    
    # Initialize deadline tracking on first call
    if _start_time is None:
        _start_time = time.time()
    
    # Safety margin: stop if we've exceeded deadline
    elapsed = time.time() - _start_time
    if elapsed > _deadline:
        return depot
    
    if len(unvisited_nodes) == 0:
        return depot
    
    # Check feasible nodes (demand <= rest_capacity)
    unvisited_arr = np.array(unvisited_nodes)
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    feasible_nodes = unvisited_arr[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return depot

    # Strategy: Nearest Neighbor with 2-Opt Lookahead
    # 1. Identify the nearest feasible neighbor (NN)
    # 2. Check if taking the second-nearest (2NN) allows a better local 2-Opt move 
    #    or if NN leads to a "trap" (e.g., immediate return to depot is forced due to capacity 
    #    or geometry, whereas 2NN keeps the route open).
    
    # Get distances from current to all feasible nodes
    dists = distance_matrix[current_node, feasible_nodes]
    
    # Sort feasible nodes by distance
    sorted_indices = np.argsort(dists)
    
    # Candidate 1: Nearest Neighbor
    nn_idx = sorted_indices[0]
    nn_node = feasible_nodes[nn_idx]
    nn_dist = dists[nn_idx]
    
    # Candidate 2: Second Nearest Neighbor
    if len(sorted_indices) > 1:
        snn_idx = sorted_indices[1]
        snn_node = feasible_nodes[snn_idx]
        snn_dist = dists[snn_idx]
    else:
        snn_node = nn_node
        snn_dist = nn_dist

    # Heuristic Check:
    # If the nearest neighbor is very close, it's usually good.
    # However, if the nearest neighbor is "isolated" (far from other unvisited nodes 
    # relative to its distance from current), it might be better to go to a node 
    # that is part of a cluster.
    
    # Simplified 2-Opt / Cluster detection:
    # Calculate the "density" or "centrality" of the candidate among remaining unvisited nodes.
    # A node that is close to many other unvisited nodes is a good hub.
    # We approximate this by checking the distance to the centroid of remaining unvisited nodes.
    
    # To keep it fast, we only do this extra check if there are > 5 nodes left.
    if len(unvisited_nodes) > 5:
        # Calculate centroid of all unvisited nodes (not just feasible, to see the "cloud")
        # Using coordinates is not available, only distance matrix.
        # So we use a proxy: Average distance to other unvisited nodes.
        # Nodes with lower average distance to others are more central.
        
        # Vectorized average distance calculation for candidates
        # dist_matrix is n x n. 
        # For a candidate c, avg_dist = sum(d(c, u) for u in unvisited) / len(unvisited)
        
        unvisited_set = set(unvisited_nodes)
        
        # We only need to compare NN and SNN.
        # Compute avg distance for NN
        nn_dists_to_others = distance_matrix[nn_node, unvisited_arr]
        nn_avg_dist = np.mean(nn_dists_to_others)
        
        # Compute avg distance for SNN
        snn_dists_to_others = distance_matrix[snn_node, unvisited_arr]
        snn_avg_dist = np.mean(snn_dists_to_others)
        
        # If SNN is significantly more central (lower avg dist) and not too far from current,
        # pick SNN.
        # Threshold: If SNN is within 1.2x distance of NN and is 10% more central, pick SNN.
        if snn_dist < nn_dist * 1.2 and snn_avg_dist < nn_avg_dist * 0.9:
            return int(snn_node)
            
    # Default to Nearest Neighbor
    return int(nn_node)

# EVOLVE-BLOCK-END
