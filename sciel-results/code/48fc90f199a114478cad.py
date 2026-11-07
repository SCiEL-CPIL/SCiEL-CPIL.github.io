import numpy as np
import time

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search penalty for TSP using rank-based, distance-normalized additive penalties.
    
    Structural changes from the seed:
    1. Scale-invariance: Penalties are normalized by the median edge distance of the instance.
    2. Rank-based weighting: Instead of using raw counts (which can have large variance/outliers),
       we use the rank of the count among all edges. This ensures a consistent relative penalty
       structure regardless of the absolute scale of usage counts.
    3. Additive penalty: Adds a constant cost (in distance units) to penalized edges,
       which is more direct for 2-opt's delta calculations than multiplicative scaling.
    """
    start_time = time.time()
    n = edge_distance.shape[0]
    
    # Work on a copy
    d_out = edge_distance.copy()
    
    if n <= 1:
        return d_out
    
    # 1. Compute scale factor: median of non-zero, off-diagonal distances
    #    This normalizes the penalty to be meaningful relative to the instance's geometry.
    mask = ~np.eye(n, dtype=bool)
    valid_distances = edge_distance[mask]
    valid_distances = valid_distances[valid_distances > 0]
    
    if len(valid_distances) == 0:
        return d_out
    
    median_dist = np.median(valid_distances)
    if median_dist <= 0:
        return d_out
    
    # 2. Prepare usage counts
    #    We need to assign a penalty based on how "frequently used" an edge is.
    #    Using ranks makes this robust to the specific accumulation history.
    
    # Get all unique edge pairs (i, j) with i < j to avoid double counting in ranking
    # But edge_n_used is symmetric. We can just work on the upper triangle.
    iu = np.triu_indices(n, k=1)
    usage_counts = edge_n_used[iu]
    
    # Get ranks of the usage counts.
    # Higher usage -> Higher rank -> Higher penalty.
    # We want edges with high usage to be penalized more.
    # rankdata: 1 for smallest, N for largest.
    # We want the penalty to be proportional to the rank.
    # Let's map ranks to [0, 1] range.
    
    if len(usage_counts) > 0:
        # Handle ties by averaging ranks (standard competition ranking is fine too, but average is smoother)
        ranks = np.argsort(np.argsort(usage_counts))
        # Normalize ranks to [0, 1]
        # Max rank is len(usage_counts) - 1
        # Min rank is 0
        normalized_ranks = ranks / max(len(usage_counts) - 1, 1)
        
        # Create a penalty matrix based on these normalized ranks
        # We want to penalize edges that are frequently used.
        # Let's say the max penalty for a heavily used edge is a fraction of the median distance.
        # This ensures the perturbation is significant but not overwhelming.
        max_penalty = 0.2 * median_dist
        
        # Construct the full penalty matrix
        penalty_matrix = np.zeros((n, n), dtype=float)
        penalty_matrix[iu] = normalized_ranks * max_penalty
        penalty_matrix = penalty_matrix + penalty_matrix.T
        
        # 3. Additional penalty for edges in the current local optimum tour
        #    These are the edges that led to the current local optimum.
        #    We want to push 2-opt to break these specific edges to escape the basin.
        tour = np.array(local_opt_tour)
        if len(tour) < 2:
            return d_out
            
        # Identify tour edges
        # Add a slightly higher penalty to these specific edges to encourage breaking the current tour structure
        tour_penalty = 0.1 * median_dist
        
        for i in range(len(tour) - 1):
            a, b = tour[i], tour[i + 1]
            penalty_matrix[a, b] += tour_penalty
            penalty_matrix[b, a] += tour_penalty
        # Close the tour
        a, b = tour[-1], tour[0]
        penalty_matrix[a, b] += tour_penalty
        penalty_matrix[b, a] += tour_penalty
        
        # 4. Apply additive penalty
        #    d_out = d_true + penalty
        #    This directly increases the cost of using penalized edges.
        d_out = d_out + penalty_matrix
        
        # Ensure diagonal is 0
        np.fill_diagonal(d_out, 0.0)
        
        # Ensure symmetry (should be, but enforce)
        d_out = (d_out + d_out.T) / 2.0
        
        # Ensure finite and non-negative
        d_out = np.nan_to_num(d_out, nan=0.0, posinf=1e10, neginf=1e10)
        d_out = np.maximum(d_out, 0.0)
        
        # Deadline guard
        elapsed = time.time() - start_time
        if elapsed > 280.0:
            return d_out
            
        return d_out
    else:
        return d_out
