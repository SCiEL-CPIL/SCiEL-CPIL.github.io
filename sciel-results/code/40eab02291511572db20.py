"""Heuristic matrix for ant colony optimization of TSP."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _nearest_neighbor_tour(dm, start=0, randomize=False, seed=None):
    """Generate a tour using the nearest neighbor heuristic, optionally randomized."""
    n = dm.shape[0]
    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    tour = [start]
    current = start
    
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random
    
    for _ in range(n - 1):
        dists = dm[current].copy()
        dists[visited] = np.inf
        
        if randomize:
            # Add noise to distances to diversify the tour
            # Noise scale relative to mean distance
            mean_dist = np.mean(dm[~np.eye(n, dtype=bool)])
            noise = rng.gauss(0, 0.1 * mean_dist)
            dists += noise
            # Ensure we don't pick visited nodes (already set to inf)
            # But noise might make inf not the max? No, inf + noise is still inf.
            # However, we must ensure we don't pick a node with negative effective dist if noise is large?
            # dists are positive, noise is small.
        
        next_node = int(np.argmin(dists))
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
    return tour

def _tour_length(tour, dm):
    """Calculate the length of a closed tour."""
    n = len(tour)
    total = 0.0
    for i in range(n):
        total += dm[tour[i], tour[(i + 1) % n]]
    return total

def _two_opt(tour, dm, max_iter=500):
    """Improve a tour using 2-opt local search."""
    n = len(tour)
    tour = list(tour)
    best_len = _tour_length(tour, dm)
    improved = True
    iters = 0
    while improved and iters < max_iter:
        improved = False
        iters += 1
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % n]
                delta = (dm[a, c] + dm[b, d]) - (dm[a, b] + dm[c, d])
                if delta < -1e-10:
                    tour[i + 1:j + 1] = reversed(tour[i + 1:j + 1])
                    best_len += delta
                    improved = True
    return tour, best_len

def _or_opt(tour, dm):
    """Improve a tour using Or-opt (relocating single nodes)."""
    n = len(tour)
    tour = list(tour)
    best_len = _tour_length(tour, dm)
    improved = True
    max_passes = 50
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(n):
            node = tour[i]
            a = tour[i - 1]
            b = tour[(i + 1) % n]
            tour.pop(i)
            n_minus_1 = n - 1
            best_delta = 0.0
            best_pos = -1
            for j in range(n_minus_1):
                if j == i or j == (i - 1) % n_minus_1:
                    continue
                c = tour[j - 1]
                d = tour[j]
                delta = (dm[c, node] + dm[node, d]) - (dm[c, d]) + (dm[a, b])
                if delta < best_delta - 1e-10:
                    best_delta = delta
                    best_pos = j
            if best_pos >= 0:
                tour.insert(best_pos, node)
                best_len += best_delta
                improved = True
            else:
                tour.insert(i, node)
    return tour, best_len

def heuristics(distance_matrix):
    """
    Returns an edge-attractiveness matrix for ACO.
    Strategy: Consensus Heuristic.
    1. Generate multiple diverse tours using randomized Nearest Neighbor + Local Search.
    2. Count the frequency of each directed edge across these tours.
    3. Combine inverse distance with edge frequency to create the heuristic.
    """
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    dm = np.asarray(distance_matrix, dtype=float)
    n = dm.shape[0]
    
    if n == 0:
        return dm.copy()
    
    # Base heuristic: inverse distance
    base_heuristic = 1.0 / (dm + 1e-9)
    
    # Edge frequency matrix
    edge_counts = np.zeros((n, n), dtype=float)
    
    # Number of diverse tours to generate
    # We have ~280s. Each tour generation (NN + 2-Opt + Or-Opt) for n=50 is fast (<0.1s).
    # Let's aim for 50-100 diverse tours.
    num_tours = 100
    
    # Use different seeds and randomization levels for diversity
    for k in range(num_tours):
        if time.time() > deadline:
            break
            
        # Vary the randomization noise and start node
        start_node = (k * 7) % n  # Pseudo-random spread of starts
        noise_level = 0.05 + 0.1 * (k % 5) / 5.0  # Vary noise
        
        # Generate tour
        tour = _nearest_neighbor_tour(dm, start=start_node, randomize=True, seed=k)
        
        # Improve
        tour, _ = _two_opt(tour, dm, max_iter=200)
        tour, _ = _or_opt(tour, dm)
        tour, _ = _two_opt(tour, dm, max_iter=200)
        
        # Update edge counts
        for i in range(n):
            u = tour[i]
            v = tour[(i + 1) % n]
            edge_counts[u, v] += 1.0
            
        # Check time
        if time.time() > deadline - 10:
            break
    
    # If no tours were generated (shouldn't happen), fallback
    if not np.any(edge_counts > 0):
        return np.maximum(base_heuristic, 1e-9)
    
    # Normalize edge counts
    # Max count is at most num_tours
    # We want edges that appear in many good tours to be attractive.
    # Use a sigmoid or simple scaling.
    # Let's scale counts to [0, 1] range relative to the max observed count.
    max_count = np.max(edge_counts)
    if max_count > 0:
        normalized_counts = edge_counts / max_count
    else:
        normalized_counts = np.zeros_like(edge_counts)
    
    # Combine base heuristic and consensus
    # Heuristic = base * (1 + alpha * normalized_counts)
    # If an edge is in all tours, boost is (1 + alpha).
    # If an edge is in no tours, boost is 1.
    # We want a strong boost for consensus edges.
    alpha = 5.0
    
    heuristic = base_heuristic * (1.0 + alpha * normalized_counts)
    
    # Ensure no values are not positive or non-finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
