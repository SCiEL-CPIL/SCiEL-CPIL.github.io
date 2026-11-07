import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Targeted Perturbation Penalty for Guided Local Search.
    
    Strategy:
    Instead of penalizing all edges in the current tour equally (or proportionally to total usage),
    we identify the 'most stuck' edges—those with the highest usage counts—and apply a strong
    additive penalty only to those edges. This creates a significant gap in the cost landscape
    specifically for the edges that the search has been repeatedly returning to, forcing the
    subsequent 2-opt to find a substantially different path through that region of the tour,
    rather than just making minor adjustments.
    
    This approach aims to break the "local loop" where the search oscillates between a few
    similar tours by making the most common sub-structures prohibitively expensive.
    """
    # Safety check
    if edge_n_used.size == 0:
        return edge_distance.copy()
    
    max_used = np.max(edge_n_used)
    if max_used == 0:
        return edge_distance.copy()
    
    # Extract the edges currently in the local_opt_tour
    # local_opt_tour is a list of nodes, e.g., [0, 1, 2, ..., 99, 0] or [0, 1, ..., 99]
    # We need to handle the closed tour. The problem states "closed-tour length".
    # The input `local_opt_tour` is likely a sequence of city indices.
    # Let's assume it's a list of length N representing the order.
    # The edges are (tour[i], tour[(i+1)%N]).
    
    n = len(local_opt_tour)
    if n < 2:
        return edge_distance.copy()
    
    # Identify the edges in the current tour
    # We will collect the usage counts of these specific edges
    tour_edges = []
    tour_edge_indices = [] # (i, j) pairs
    
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        tour_edges.append(edge_n_used[u, v])
        tour_edge_indices.append((u, v))
    
    tour_usage_counts = np.array(tour_edges)
    
    # Determine which edges to penalize.
    # We only penalize edges that are "significantly" used.
    # Threshold: If an edge is in the top quartile of usage counts among the current tour's edges,
    # or if its usage is above a certain absolute threshold (e.g., 2 or 3 times), we penalize it.
    # Given the context of "stuck" edges, let's target the edges with usage > median usage of the tour.
    
    if np.max(tour_usage_counts) == 0:
        return edge_distance.copy()
    
    # Calculate the median usage of the edges in the current tour
    median_usage = np.median(tour_usage_counts)
    
    # We want to penalize edges that are more frequent than the median.
    # This ensures we target the "core" of the current local optimum.
    # If all edges have the same usage, we penalize none? Or all?
    # If they are all equal, it's not "stuck" in a specific way, it's just a standard tour.
    # However, in GLS, usually all tour edges get penalized.
    # But our strategy is *targeted*.
    # Let's penalize edges where usage > median_usage.
    # If usage counts are [1, 1, 1, 10], median is 1. Only the 10 gets penalized.
    # If usage counts are [2, 2, 2, 2], median is 2. None get penalized (strictly >).
    # This might be too weak. Let's use >= median if there is variance, or just top 50%.
    
    # Better: Penalize the top 50% of edges in the tour by usage count.
    # This guarantees that we always perturb the most "repeated" parts of the current solution.
    
    # Sort edges by usage count
    sorted_indices = np.argsort(tour_usage_counts)
    num_to_penalize = max(1, n // 2)
    
    # The indices of the edges in the tour list that should be penalized
    # We take the top half (highest usage)
    indices_to_penalize = sorted_indices[-num_to_penalize:]
    
    # Create the penalty matrix
    # We use an additive penalty.
    # The magnitude should be significant enough to change the 2-opt outcome.
    # Typical TSP distances in unit square for 100 points:
    # Optimal length ~ sqrt(100/pi) * 1.0 ~ 5.6? No, expected length is approx sqrt(N/pi).
    # For N=100, sqrt(100/3.14) ~ sqrt(31.8) ~ 5.6.
    # Edge lengths are roughly 1/sqrt(N) ~ 0.1.
    # So a penalty of 0.5 or 1.0 is huge.
    # Let's scale the penalty based on the usage count relative to the max.
    
    penalty_matrix = np.zeros_like(edge_distance)
    
    # Base penalty magnitude
    # We want to make the "stuck" edges effectively unselectable by 2-opt.
    # Adding a value larger than the typical edge length difference will force a swap.
    # Let's use a penalty proportional to the usage count itself, but scaled.
    # If max_used is 20 (last round), and we penalize the top edges.
    
    for idx in indices_to_penalize:
        u, v = tour_edge_indices[idx]
        usage = tour_usage_counts[idx]
        
        # Penalty = alpha * usage
        # If alpha is too small, it won't break the local optimum.
        # If alpha is too large, it might create artificial long paths.
        # Let's try a fixed strong penalty for the targeted edges.
        # Since we only pick the top 50%, we can use a uniform strong penalty for simplicity and robustness.
        # Or scale it: penalty = 0.1 * usage.
        # If usage is 1, penalty 0.1. If usage is 10, penalty 1.0.
        # Given edge lengths ~0.1, a penalty of 0.5 is 5x the edge length.
        
        # Let's use a penalty that ensures the edge becomes "expensive".
        # penalty = 0.2 * usage
        pen_val = 0.2 * usage
        
        # Apply to symmetric matrix
        penalty_matrix[u, v] += pen_val
        penalty_matrix[v, u] += pen_val
        
    return edge_distance + penalty_matrix
