"""Guided local search penalty for TSP: Nearest Neighbor Bias Disruption."""
import numpy as np
import time

_start_time = time.perf_counter()
_DEADLINE = _start_time + 315.0  # Safety margin before 330s kill


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    # Deadline guard
    if time.perf_counter() > _DEADLINE:
        return np.asarray(edge_distance, dtype=float).copy()

    n = edge_distance.shape[0]
    if n <= 2:
        return np.asarray(edge_distance, dtype=float).copy()

    # True distances
    d = np.asarray(edge_distance, dtype=float)

    # Build the edge list of the current tour (closed cycle)
    tour = np.asarray(local_opt_tour, dtype=int)
    if tour.shape[0] != n:
        tour = np.arange(n)

    # Edge endpoints for the closed tour
    i = tour[:-1]
    j = tour[1:]
    i = np.append(i, tour[0])
    j = np.append(j, tour[-1])

    # Identify K-Nearest Neighbors for each node
    # K=5 is a standard heuristic for TSP candidate edges
    K = 5
    nn_mask = np.zeros((n, n), dtype=bool)
    
    # Efficiently compute K-NN using argsort on rows
    # d has inf on diagonal? No, d is true distances. We should ignore self.
    d_sorted_idx = np.argsort(d, axis=1)
    
    # For each node, mark its K nearest neighbors
    for u in range(n):
        # Skip self if it appears in the first K (it shouldn't if d[u,u]=0 and others > 0)
        # But to be safe, we take the first K indices that are not u
        neighbors = d_sorted_idx[u, :K+1] # Take K+1 to allow for self-skip if needed
        count = 0
        for v in neighbors:
            if v == u:
                continue
            nn_mask[u, v] = True
            count += 1
            if count >= K:
                break

    # Symmetrize the NN mask: if v is in K-NN of u, or u is in K-NN of v
    # This defines the "Good Edge" set
    is_candidate = nn_mask | nn_mask.T

    # Build penalty matrix
    penalty = np.zeros((n, n), dtype=float)
    
    # Usage-based penalty: penalize edges that have been used many times
    usage = np.asarray(edge_n_used, dtype=float)
    usage = np.maximum(usage, 0.0)
    u_max = usage.max()
    
    # Base usage penalty: mild, to prevent cycling
    if u_max > 0:
        # Log-scaled usage
        usage_scaled = np.log1p(usage) / np.log1p(u_max)
        # Apply mild global usage penalty
        penalty += 0.1 * usage_scaled
    else:
        penalty += np.zeros_like(usage)

    # Structural Disruption:
    # Penalize edges in the current tour that are NOT candidate edges (not K-NN for either endpoint)
    # This forces the tour to shed "structurally unlikely" edges.
    
    tour_edge_indices = np.arange(n)
    
    # Vectorized check for candidate status of tour edges
    # is_candidate[i, j] is True if (i,j) is a candidate edge
    tour_is_candidate = is_candidate[i, j]
    
    # We want to penalize edges where tour_is_candidate is False
    # Penalty magnitude: proportional to edge length and usage
    
    tour_len = d[i, j]
    
    # Create a mask for non-candidate tour edges
    non_candidate_mask = ~tour_is_candidate
    
    # Apply penalty to non-candidate edges
    # We use a strong multiplier to ensure 2-opt breaks these edges
    # Penalty = alpha * length + beta * usage
    alpha = 0.5
    beta = 0.5
    
    # We only apply this to the specific edges in the tour
    # We can't just add to the matrix globally because we only want to disrupt the *current* bad edges
    # However, GLS usually adds penalty to the matrix for all edges, and the search finds a new tour.
    # If we only penalize the current tour's bad edges, the search might just swap them for other bad edges.
    # Better strategy: Penalize ALL non-candidate edges that are currently in the tour heavily.
    # Also, penalize non-candidate edges that have high usage (even if not in tour) to discourage future use.
    
    # Adaptive Topological Disruption:
    # Penalize the k longest edges in the current tour.
    # k scales with the average usage of tour edges to increase disruption when stuck.
    
    # Calculate average usage of current tour edges
    tour_usage = usage[i, j]
    avg_tour_usage = np.mean(tour_usage)
    
    # Determine k: number of edges to penalize
    # Base k is 5% of n, scaled up by usage (capped at 20% of n)
    base_k = max(1, int(0.05 * n))
    scale_factor = 1.0 + min(avg_tour_usage / 5.0, 1.0)  # Scale up to 2x based on usage
    k = min(int(base_k * scale_factor), int(0.2 * n))
    k = max(k, 1)  # At least 1 edge
    
    # Get indices of the k longest tour edges
    if k < n:
        # Find the k largest values in tour_len
        # argsort gives indices that would sort the array, so we take the last k
        top_k_indices = np.argsort(tour_len)[-k:]
    else:
        top_k_indices = np.arange(n)
    
    # Apply strong penalty to these edges
    # Penalty magnitude: 1.0 * length + 0.5 * usage
    for idx in top_k_indices:
        a, b = i[idx], j[idx]
        mag = 1.0 * tour_len[idx] + 0.5 * usage[a, b]
        penalty[a, b] += mag
        penalty[b, a] += mag

    # Symmetrize
    penalty = 0.5 * (penalty + penalty.T)

    # Result: true distance + penalty
    result = d + penalty

    # Ensure finiteness
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=0.0)

    return result
