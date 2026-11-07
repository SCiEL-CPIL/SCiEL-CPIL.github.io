import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search penalty for TSP using adaptive additive penalties.
    
    Strategy:
    1. Compute a scale factor based on the current tour length relative to 
       the median edge length. This normalizes the penalty magnitude so that 
       the perturbation is consistent in terms of "tour cost units" rather than 
       raw distance units, which varies wildly between instances.
    2. Apply an additive penalty to edges based on their cumulative usage 
       (edge_n_used) and their presence in the current local optimum tour.
    3. The penalty is capped to avoid infinite distances, ensuring 2-opt 
       remains effective.
    
    Returns a perturbed distance matrix.
    """
    start_time = time.time()
    n = edge_distance.shape[0]
    
    # Work on a copy to avoid modifying the input
    d_out = edge_distance.copy()
    
    if n <= 1:
        return d_out
    
    # Get edges in the current local optimum tour
    tour = np.array(local_opt_tour)
    if len(tour) < 2:
        return d_out
    
    # Identify edges in the current tour
    tour_edges = np.zeros((n, n), dtype=bool)
    for i in range(len(tour) - 1):
        a, b = tour[i], tour[i + 1]
        tour_edges[a, b] = True
        tour_edges[b, a] = True
    # Close the tour
    a, b = tour[-1], tour[0]
    tour_edges[a, b] = True
    tour_edges[b, a] = True
    
    # Calculate current tour length
    tour_length = 0.0
    for i in range(len(tour) - 1):
        tour_length += edge_distance[tour[i], tour[i + 1]]
    tour_length += edge_distance[tour[-1], tour[0]]
    
    # Estimate a lower bound / typical scale
    # Using median of off-diagonal distances as a proxy for typical edge length
    # This helps normalize the penalty so it's a consistent fraction of tour cost
    mask = ~np.eye(n, dtype=bool)
    median_edge = np.median(edge_distance[mask])
    if median_edge <= 0:
        median_edge = 1.0
    
    # Scale factor: How much "distance" should one unit of penalty add?
    # We want the penalty to be significant but not overwhelming.
    # A typical tour has n edges. If we penalize an edge by X, it affects tour length by X.
    # Let's aim for a perturbation that is roughly 1-5% of the total tour length 
    # for the most penalized edges.
    # max_possible_penalty_per_edge ~ 0.05 * tour_length
    # But we distribute this based on usage counts.
    
    # Adaptive scaling:
    # If the tour is "long" (high tour_length / n / median_edge), we can afford larger penalties.
    # If the tour is "tight", we need smaller penalties to avoid breaking good structures.
    # Let's define a base penalty unit.
    base_unit = 0.01 * median_edge * n  # ~1% of a typical tour length (n * median_edge)
    
    # Normalize usage counts to [0, 1] range for scaling
    max_usage = np.max(edge_n_used)
    if max_usage > 0:
        normalized_usage = edge_n_used / max_usage
    else:
        normalized_usage = edge_n_used
    
    # Additive penalty calculation:
    # Penalty = base_unit * (alpha_usage * normalized_usage + alpha_tour * tour_edge_indicator)
    # We use linear scaling for usage to be more aggressive with frequent edges,
    # but capped by the base_unit scaling.
    
    alpha_usage = 1.0
    alpha_tour = 0.5
    
    penalty_matrix = np.zeros((n, n), dtype=float)
    penalty_matrix += alpha_usage * normalized_usage * base_unit
    
    # Additional immediate penalty for edges in the current tour
    penalty_matrix[tour_edges] += alpha_tour * base_unit
    
    # Ensure diagonal is 0
    np.fill_diagonal(penalty_matrix, 0.0)
    
    # Apply additive penalty
    d_out = d_out + penalty_matrix
    
    # Ensure symmetry
    d_out = (d_out + d_out.T) / 2.0
    
    # Ensure finite values
    d_out = np.nan_to_num(d_out, nan=0.0, posinf=1e10, neginf=1e10)
    
    # Ensure non-negative distances
    d_out = np.maximum(d_out, 0.0)
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 280.0:
        return d_out
    
    return d_out
