"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 115.0  # Safety margin below 120s

def _check_deadline():
    return (time.time() - _START_TIME) > _TIME_LIMIT

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using a 2-step lookahead heuristic.
    For each candidate in unvisited_nodes, we estimate the cost of moving to it
    and then to its nearest unvisited neighbor (or destination if it's the last one).
    We choose the candidate that minimizes this 2-step cost.
    """
    # Check deadline for safety
    if _check_deadline():
        # Fallback to nearest neighbor if time is running out
        if len(unvisited_nodes) == 0:
            return -1
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1
    
    # If only one node left, it must be the destination (or just the last one)
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # If two nodes left, we must pick the one that leads to the other, 
    # or rather, the one that minimizes dist(current, c1) + dist(c1, c2) vs dist(current, c2) + dist(c2, c1).
    # Actually, if 2 nodes left, the tour closes from the last one to destination.
    # The problem statement implies destination_node is the start node for closing the loop.
    # Let's assume standard TSP where we start at dest, visit all, return to dest.
    # The function signature includes destination_node.
    
    # General case: 2-step lookahead
    # For each candidate c in unvisited_nodes:
    #   Cost to go current -> c is dist(current, c)
    #   Next step from c:
    #     If c is the last unvisited node (i.e., n_unvisited == 1, handled above), go to destination.
    #     Else, go to the nearest unvisited node from c (excluding c itself).
    #   Total 2-step cost = dist(current, c) + min_{u in unvisited \ {c}} dist(c, u)
    
    # Optimization: Precompute distances from current to all unvisited
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    best_cost = np.inf
    best_node = unvisited_nodes[0]
    
    # We iterate over candidates
    # To make this faster, we can use vectorized operations where possible, 
    # but the "next nearest" part depends on the candidate.
    # For N=50, a simple loop is fast enough.
    
    # Create a copy of unvisited nodes for easy indexing
    unvisited_list = unvisited_nodes
    
    for i, c in enumerate(unvisited_list):
        # Cost to reach c
        cost_to_c = dists_from_current[i]
        
        # Find the best next step from c
        # Exclude c from the unvisited nodes for the next step
        # We need the min distance from c to any other node in unvisited_nodes
        # dist_matrix[c, unvisited_nodes] gives distances to all unvisited
        # We set the distance to c itself to inf to exclude it
        
        # Vectorized approach for finding min next step
        dists_from_c = distance_matrix[c, unvisited_nodes]
        # Mask out c itself
        # Since unvisited_nodes is an array, we can't easily mask by value in a simple slice without creating a new array.
        # dists_from_c[i] corresponds to distance c->c which is 0.
        # We want min of dists_from_c excluding index i.
        
        # Efficient way:
        # If n_unvisited is small, just compute min of the rest.
        # dists_from_c has shape (n_unvisited,)
        # We can do:
        # temp = dists_from_c.copy()
        # temp[i] = np.inf
        # next_cost = np.min(temp)
        
        # Even faster without copy:
        if n_unvisited == 2:
            # The other node is the one at index 1-i
            other_idx = 1 - i
            next_cost = dists_from_c[other_idx]
        else:
            # Use np.min on a view or just exclude the element
            # np.min on a slice is not directly possible if we want to exclude one element without copy in older numpy, 
            # but we can do:
            # next_cost = np.min(np.delete(dists_from_c, i))
            # np.delete is O(N), which is fine for N=50.
            next_cost = np.min(np.delete(dists_from_c, i))
            
        total_cost = cost_to_c + next_cost
        
        if total_cost < best_cost:
            best_cost = total_cost
            best_node = c
            
    return int(best_node)
# EVOLVE-BLOCK-END
