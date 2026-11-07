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
    
    # Strategy: Global Repulsive Field
    # Instead of penalizing only the current tour edges, we penalize ALL edges
    # based on their cumulative usage count.
    # 
    # Rationale:
    # The standard GLS approach penalizes only the edges in the current local optimum.
    # This can sometimes lead to a "symmetric" escape where the search just swaps
    # edges back and forth. By penalizing ALL previously used edges globally, we
    # create a pressure that pushes the 2-opt to find edges that have been used
    # less frequently in the history of the search.
    #
    # We use a non-linear scaling to make the penalty stronger for more frequent edges.
    # penalty = alpha * (usage_count)^beta * edge_length
    #
    # alpha controls the overall strength.
    # beta > 1 makes the penalty super-linear, strongly discouraging very common edges.
    #
    # We also ensure the penalty doesn't make edges infinite, just very costly.
    
    # Get the maximum usage count to normalize if necessary, but raw counts often work well
    # with a small alpha.
    # edge_n_used is a symmetric matrix.
    
    # To avoid penalizing the diagonal (which is 0 distance and 0 usage usually), 
    # we just apply the formula. Since edge_distance[i,i] is 0, the penalty will be 0.
    
    # Use a power of 1.5 to create a strong non-linear effect
    # Usage counts are integers. 
    # If an edge was used 10 times, penalty factor is 10^1.5 = 31.6
    # If used 1 time, penalty factor is 1.
    
    # We need to be careful not to make the matrix too distorted.
    # Let's use a smaller alpha since we are applying it to ALL edges, not just ~100 out of 4950.
    
    alpha = 0.05
    beta = 1.5
    
    # Calculate penalties for all edges
    # np.power handles 0^1.5 = 0 correctly.
    usage_counts = edge_n_used
    edge_lengths = edge_distance
    
    # Apply penalty: modified_dist = edge_distance + alpha * (usage_counts ** beta) * edge_distance
    # This is a multiplicative penalty on the distance, scaled by usage.
    # dist_new = dist * (1 + alpha * usage^beta)
    
    # Vectorized operation
    penalty_factor = alpha * np.power(usage_counts, beta)
    modified_dist = edge_distance * (1.0 + penalty_factor)
    
    # Ensure symmetry (though edge_n_used and edge_distance are symmetric, 
    # numerical noise might creep in, but usually not an issue with direct ops)
    # modified_dist = (modified_dist + modified_dist.T) / 2.0 # Optional, might be slow
    
    # Ensure finite values
    # If any values are inf or nan, replace them with a large finite number
    # But with alpha=0.05 and max usage ~20, max factor is 0.05 * 20^1.5 * 1.414 ~ 4.
    # So values should be well within float range.
    
    return modified_dist
# EVOLVE-BLOCK-END
