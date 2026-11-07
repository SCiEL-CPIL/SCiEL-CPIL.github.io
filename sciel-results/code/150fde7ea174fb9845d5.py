"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_GLOBAL_START_TIME = time.time()
_WALL_CLOCK_LIMIT = 325.0  # Safety margin before 330s kill

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    # Check time limit - return original distances if we're running out of time
    if time.time() - _GLOBAL_START_TIME > _WALL_CLOCK_LIMIT:
        return edge_distance.copy()
    
    # Create a copy of the true distance matrix to modify
    modified_dist = edge_distance.copy()
    
    n = edge_distance.shape[0]
    
    # Identify edges in the current local optimum tour
    # local_opt_tour is a permutation of [0, n-1]
    # Edges are (tour[i], tour[i+1]) for i in 0..n-2 and (tour[n-1], tour[0])
    
    # We want to penalize these specific edges based on how often they have been used.
    # edge_n_used is a symmetric count matrix.
    
    # Extract the usage counts for the edges in the current tour
    # For each edge (u, v) in the tour, we add a penalty to modified_dist[u, v] and modified_dist[v, u]
    
    # Vectorized approach for speed:
    # Create indices for the tour edges
    i_idx = np.arange(n)
    j_idx = (i_idx + 1) % n
    
    u = local_opt_tour[i_idx]
    v = local_opt_tour[j_idx]
    
    # Get the usage counts for these specific directed pairs (u, v)
    # Since edge_n_used is symmetric, edge_n_used[u, v] is the count for that edge.
    usage_counts = edge_n_used[u, v]
    
    # Apply additive penalty
    # The penalty should be significant enough to break the local optimum but not so large
    # that it completely destroys the geometry for 2-opt.
    # A common heuristic in Guided Local Search is alpha * d(i,j) or just a constant.
    # Using usage_counts directly as a weight for an additive penalty on the edge length.
    
    # Let's scale the penalty. 
    # If an edge has been used 5 times, we add 5 * scale to its distance.
    # A scale of 1.0 might be too strong or too weak depending on the magnitude of distances.
    # Distances are in [0, sqrt(2)]. 
    # Let's use a penalty proportional to the edge's own length to maintain relative scale?
    # Or just a fixed additive penalty?
    # Standard GLS often adds alpha * d(i,j). Let's try that.
    
    # Get the true lengths of these edges
    edge_lengths = edge_distance[u, v]
    
    # Penalty = alpha * usage_count * edge_length
    # Let's choose alpha = 0.5. If an edge is used 10 times, it gets a penalty of 5 * length.
    # This effectively doubles the cost of that edge.
    alpha = 0.5
    penalties = alpha * usage_counts * edge_lengths
    
    # Apply to the matrix
    # modified_dist[u, v] += penalties
    # modified_dist[v, u] += penalties
    
    modified_dist[u, v] += penalties
    modified_dist[v, u] += penalties
    
    return modified_dist
# EVOLVE-BLOCK-END
