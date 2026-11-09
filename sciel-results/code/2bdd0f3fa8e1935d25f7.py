"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _two_opt_vectorized(tour, dist, max_passes=10):
    """Improve a tour using vectorized 2-opt. Returns optimized tour as list."""
    n = len(tour)
    tour = np.array(tour, dtype=int)
    
    for _ in range(max_passes):
        improved = False
        for i in range(n - 1):
            a = tour[i]
            b = tour[i + 1]
            # For all j > i+1, compute the 2-opt delta
            c_arr = tour[i + 2:]
            if len(c_arr) == 0:
                break
            d_arr = np.empty_like(c_arr)
            d_arr[:-1] = tour[i + 3:]
            d_arr[-1] = tour[0]
            
            old_cost = dist[a, b] + dist[c_arr, d_arr]
            new_cost = dist[a, c_arr] + dist[b, d_arr]
            deltas = new_cost - old_cost
            best_j = np.argmin(deltas)
            if deltas[best_j] < -1e-12:
                j = i + 2 + best_j
                tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                improved = True
        if not improved:
            break
    
    return tour.tolist()


def _nearest_neighbor_tour(dist, start=0):
    """Generate a tour using nearest neighbor heuristic."""
    n = dist.shape[0]
    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    tour = [start]
    current = start
    for _ in range(n - 1):
        dists = dist[current, :].copy()
        dists[visited] = np.inf
        next_node = np.argmin(dists)
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
    return tour


def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 330s limit
    n = distance_matrix.shape[0]
    
    # Generate multiple diverse reference tours and optimize them
    reference_tours = []
    
    # Nearest neighbor from multiple starts, then optimize with fast 2-opt
    for start in [0, 1, n // 2, n // 3, n - 1]:
        if time.time() > deadline - 20:
            break
        nn_tour = _nearest_neighbor_tour(distance_matrix, start)
        optimized = _two_opt_vectorized(nn_tour, distance_matrix)
        reference_tours.append(optimized)
    
    # Compute tour lengths and weights
    tour_lengths = []
    for tour in reference_tours:
        tour_len = sum(distance_matrix[tour[i], tour[(i + 1) % n]] for i in range(n))
        tour_lengths.append(tour_len)
    
    min_len = min(tour_lengths)
    max_len = max(tour_lengths)
    if max_len > min_len + 1e-12:
        # Exponential weighting: better tours get much higher weight
        tour_weights = [np.exp(5.0 * (max_len - l) / (max_len - min_len)) for l in tour_lengths]
    else:
        tour_weights = [1.0] * len(tour_lengths)
    
    # Signal 1: Edge frequency in reference tours (weighted by tour quality)
    edge_freq = np.zeros((n, n))
    for tour, weight in zip(reference_tours, tour_weights):
        for i in range(n):
            u = tour[i]
            v = tour[(i + 1) % n]
            edge_freq[u, v] += weight
    
    # Signal 2: k-nearest-neighbor edge signal
    # For each node, rank edges by distance; top-k edges get high attractiveness
    k = 4  # number of nearest neighbors
    knn_signal = np.zeros((n, n))
    for i in range(n):
        # Get distances from node i to all others (excluding self)
        dists = distance_matrix[i, :].copy()
        dists[i] = np.inf
        # Get top-k nearest
        top_k = np.argpartition(dists, k)[:k]
        top_k_dists = dists[top_k]
        # Normalize: closer = higher attractiveness
        if len(top_k_dists) > 0:
            min_d = np.min(top_k_dists)
            max_d = np.max(top_k_dists)
            for idx, node in enumerate(top_k):
                if max_d > min_d + 1e-12:
                    norm = 1.0 - (dists[node] - min_d) / (max_d - min_d)
                else:
                    norm = 1.0
                knn_signal[i, node] = norm
    
    # Scale signals to comparable range
    ef_max = np.max(edge_freq)
    if ef_max > 0:
        edge_freq_scaled = edge_freq / ef_max
    else:
        edge_freq_scaled = edge_freq
    
    knn_max = np.max(knn_signal)
    if knn_max > 0:
        knn_scaled = knn_signal / knn_max
    else:
        knn_scaled = knn_signal
    
    # Combine: 70% tour frequency, 30% kNN signal
    alpha = 0.7
    combined = alpha * edge_freq_scaled + (1 - alpha) * knn_scaled
    
    # Set diagonal to 0
    np.fill_diagonal(combined, 0.0)
    
    # Apply aggressive power law to amplify differences
    mask = ~np.eye(n, dtype=bool)
    min_val = np.min(combined[mask])
    max_val = np.max(combined[mask])
    
    if max_val > min_val + 1e-12:
        normalized = (combined - min_val) / (max_val - min_val + 1e-12)
        # Power 2.0 for strong amplification of top edges
        powered = normalized ** 2.0
        edge_attractiveness_final = 1.0 + 50.0 * powered
    else:
        edge_attractiveness_final = 1.0 + 50.0 * normalized
    
    # Ensure diagonal is 0
    np.fill_diagonal(edge_attractiveness_final, 0.0)
    
    # Ensure all off-diagonal entries are positive
    edge_attractiveness_final[mask] = np.maximum(edge_attractiveness_final[mask], 1e-9)
    
    # Final sanity check
    if not np.all(np.isfinite(edge_attractiveness_final)):
        fallback = 1.0 / (distance_matrix + 1e-9)
        np.fill_diagonal(fallback, 0.0)
        fallback[mask] = np.maximum(fallback[mask], 1e-9)
        return fallback
    
    return edge_attractiveness_final
# EVOLVE-BLOCK-END
