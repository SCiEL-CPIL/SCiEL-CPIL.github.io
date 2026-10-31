"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for deadline guard
_start_time = time.time()
_TIME_LIMIT = 140.0  # Safety margin before 150s limit

def _compute_mst_weight(nodes, distance_matrix):
    """
    Compute the weight of the Minimum Spanning Tree for the subgraph induced by 'nodes'.
    Uses Prim's algorithm with O(N^2) complexity, which is efficient for N <= 50.
    """
    n = len(nodes)
    if n <= 1:
        return 0.0
    
    # Extract the sub-matrix for the nodes
    # nodes is an array of indices
    sub_dist = distance_matrix[np.ix_(nodes, nodes)]
    
    in_mst = np.zeros(n, dtype=bool)
    min_dist = np.full(n, np.inf)
    min_dist[0] = 0.0
    total_weight = 0.0
    
    for _ in range(n):
        # Find the node with the minimum distance that is not yet in MST
        # Use argmin on the masked array
        # To avoid issues with inf, we can just find the global min among unvisited
        # Since min_dist for visited nodes is not set to inf, we need to be careful.
        # Standard Prim: pick u = argmin(min_dist[i] for i not in MST)
        
        # Mask visited nodes with inf
        min_dist_masked = min_dist.copy()
        min_dist_masked[in_mst] = np.inf
        
        u = np.argmin(min_dist_masked)
        
        if min_dist[u] == np.inf:
            # Disconnected graph? Should not happen in TSP context with Euclidean distances
            break
            
        total_weight += min_dist[u]
        in_mst[u] = True
        
        # Update min_dist for all other nodes
        # min_dist[v] = min(min_dist[v], dist(u, v))
        new_dists = sub_dist[u]
        # Update only for nodes not in MST? No, updating all is fine, in_mst handles exclusion later
        # But we can optimize by only updating unvisited
        # min_dist = np.minimum(min_dist, new_dists)
        # This is O(N) per step, total O(N^2)
        
        # We must ensure we don't update the MST node itself or already in MST?
        # The mask in the next iteration handles it.
        min_dist = np.minimum(min_dist, new_dists)
        
    return total_weight

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using an MST-based heuristic.
    
    Strategy:
    1. Check time budget. If tight, fall back to nearest neighbor.
    2. For each candidate node `c` in `unvisited_nodes`:
       a. Calculate the cost of connecting `current_node` to `c`.
       b. Calculate the MST weight of the remaining unvisited nodes (excluding `c`).
       c. The total estimated cost is `dist(current, c) + MST_weight(remaining)`.
    3. Select the candidate `c` that minimizes this total cost.
    
    This approach considers the global structure of the remaining problem,
    avoiding choices that are locally cheap but lead to expensive residual tours.
    """
    # Deadline guard
    if time.time() - _start_time > _TIME_LIMIT:
        # Fallback: nearest neighbor
        if len(unvisited_nodes) == 0:
            return -1
        if len(unvisited_nodes) == 1:
            return int(unvisited_nodes[0])
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])

    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
        
    # If very few nodes, just use nearest neighbor to save time?
    # MST is O(N^2). For N=50, it's 2500 ops. 50 calls * 50 candidates * 2500 ops = 6.25M ops.
    # This is well within the time limit.
    
    # Pre-compute distances from current node to all unvisited nodes
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    best_score = np.inf
    best_node = -1
    
    # Iterate through each candidate
    for i, c in enumerate(unvisited_nodes):
        # Cost to connect current to candidate
        connect_cost = dists_from_current[i]
        
        # Remaining nodes after visiting c
        # We need to compute MST of unvisited_nodes excluding c
        # Create a list/array of remaining indices
        # To avoid creating a new array every time, we can use a mask or list comprehension
        # But np.ix_ is fast enough.
        
        # Mask for remaining nodes
        # remaining_indices = np.delete(unvisited_nodes, i)
        
        # Calculate MST weight of the remaining subgraph
        # Note: If remaining is empty, MST weight is 0.
        if n_unvisited == 2:
            mst_weight = 0.0
        else:
            # Construct the array of remaining nodes
            # Using np.delete is O(N), which is fine.
            remaining_nodes = np.delete(unvisited_nodes, i)
            mst_weight = _compute_mst_weight(remaining_nodes, distance_matrix)
        
        total_score = connect_cost + mst_weight
        
        if total_score < best_score:
            best_score = total_score
            best_node = c
            
    return int(best_node)

# EVOLVE-BLOCK-END
