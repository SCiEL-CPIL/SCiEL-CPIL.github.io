"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Assign a priority score to each vertex for a greedy Maximum Independent Set decoder.
    
    The greedy decoder picks vertices in descending score order, skipping neighbors
    of already-picked vertices. We want scores that lead to large independent sets.
    
    Strategy: Use a combination of structural heuristics with a fast vectorized simulation
    to select the best scoring strategy from a set of candidates.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    adj = np.asarray(adjacency)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([], dtype=float)
    
    # Convert to boolean for efficient operations
    adj_bool = adj.astype(bool)
    
    # Compute degree
    degree = adj_bool.sum(axis=1).astype(float)
    
    # Compute for each vertex: the sum of degrees of its neighbors
    neighbor_degree_sum = (adj_bool @ degree).astype(float)
    
    # Compute for each vertex: the number of low-degree neighbors (degree <= median)
    median_degree = np.median(degree)
    low_degree_neighbors = (adj_bool @ (degree <= median_degree).astype(float)).astype(float)
    
    # Compute for each vertex: the number of high-degree neighbors (degree > median)
    high_degree_neighbors = (adj_bool @ (degree > median_degree).astype(float)).astype(float)
    
    # Compute for each vertex: the max degree among its neighbors
    # Use a trick: for each vertex, find the max degree of its neighbors
    # We can do this by iterating, but for efficiency, let's use a simpler proxy
    # Actually, let's compute the average degree of neighbors
    avg_neighbor_degree = np.zeros(n, dtype=float)
    for i in range(n):
        if degree[i] > 0:
            avg_neighbor_degree[i] = neighbor_degree_sum[i] / degree[i]
    
    # Compute for each vertex: the ratio of its degree to the average degree
    avg_deg = degree.mean() if degree.mean() > 0 else 1.0
    degree_ratio = degree / avg_deg
    
    # Normalize the components to comparable scales
    max_degree = degree.max() if degree.max() > 0 else 1.0
    max_neighbor_degree_sum = neighbor_degree_sum.max() if neighbor_degree_sum.max() > 0 else 1.0
    max_low_degree_neighbors = low_degree_neighbors.max() if low_degree_neighbors.max() > 0 else 1.0
    max_high_degree_neighbors = high_degree_neighbors.max() if high_degree_neighbors.max() > 0 else 1.0
    max_avg_neighbor_degree = avg_neighbor_degree.max() if avg_neighbor_degree.max() > 0 else 1.0
    max_degree_ratio = degree_ratio.max() if degree_ratio.max() > 0 else 1.0
    
    # Check if we're still within time budget
    if time.time() > deadline:
        return -degree  # Fallback to simple degree-based
    
    # Precompute neighbor lists for faster simulation
    neighbor_lists = [np.where(adj_bool[i])[0] for i in range(n)]
    
    def simulate_greedy_fast(scores):
        """Simulate the greedy decoder using precomputed neighbor lists."""
        selected = 0
        available = np.ones(n, dtype=bool)
        
        # Get vertices in order of descending score
        order = np.argsort(-scores, kind='stable')
        
        for v in order:
            if not available[v]:
                continue
            # Select this vertex
            selected += 1
            # Remove it and its neighbors from available
            available[v] = False
            neighbors = neighbor_lists[v]
            if len(neighbors) > 0:
                available[neighbors] = False
            
            # Early exit optimization: if no more available vertices
            if not available.any():
                break
        
        return selected
    
    # Generate multiple alternative scoring strategies
    strategies = []
    
    # Strategy 1: Pure degree-based (lower degree = higher priority)
    strategies.append(-degree)
    
    # Strategy 2: Inverse degree
    strategies.append(1.0 / (degree + 1.0))
    
    # Strategy 3: Degree with neighbor degree adjustment
    strategies.append(-degree + neighbor_degree_sum / (max_neighbor_degree_sum + 1e-10))
    
    # Strategy 4: Degree with low-degree neighbor penalty
    strategies.append(-degree - low_degree_neighbors / (max_low_degree_neighbors + 1e-10))
    
    # Strategy 5: Degree with high-degree neighbor bonus
    strategies.append(-degree + high_degree_neighbors / (max_high_degree_neighbors + 1e-10))
    
    # Strategy 6: Average neighbor degree based
    strategies.append(-avg_neighbor_degree / (max_avg_neighbor_degree + 1e-10))
    
    # Strategy 7: Degree ratio based
    strategies.append(-degree_ratio / (max_degree_ratio + 1e-10))
    
    # Strategy 8: Combined: low degree + high neighbor degree
    strategies.append(-degree / max_degree + neighbor_degree_sum / max_neighbor_degree_sum)
    
    # Strategy 9: Low degree with tie-break by neighbor degree
    strategies.append(-degree - 0.1 * neighbor_degree_sum / max_neighbor_degree_sum)
    
    # Strategy 10: Pure neighbor degree sum (prefer vertices next to hubs)
    strategies.append(neighbor_degree_sum)
    
    # Strategy 11: Composite score
    score = (-degree / max_degree) + (neighbor_degree_sum / max_neighbor_degree_sum) - (low_degree_neighbors / max_low_degree_neighbors)
    strategies.append(score)
    
    # Strategy 12: Low degree + high-degree neighbors
    strategies.append(-degree / max_degree + high_degree_neighbors / max_high_degree_neighbors)
    
    # Strategy 13: Inverse degree with neighbor adjustment
    strategies.append(1.0 / (degree + 1.0) + neighbor_degree_sum / (max_neighbor_degree_sum + 1e-10))
    
    # Strategy 14: Degree-based with quadratic penalty
    strategies.append(-degree - 0.5 * (degree / max_degree) ** 2)
    
    # Strategy 15: Log degree based
    strategies.append(-np.log(degree + 1.0))
    
    # Strategy 16: Degree with clustering proxy (using low-degree neighbors as proxy)
    strategies.append(-degree - 0.3 * low_degree_neighbors / (max_low_degree_neighbors + 1e-10) + 0.3 * high_degree_neighbors / (max_high_degree_neighbors + 1e-10))
    
    # Strategy 17: Priority based on (degree - avg_neighbor_degree)
    strategies.append(-(degree - avg_neighbor_degree))
    
    # Strategy 18: Priority based on (neighbor_degree_sum / degree)
    strategies.append(-(neighbor_degree_sum / (degree + 1e-10)))
    
    # New strategies:
    # Strategy 19: Prefer vertices with many low-degree neighbors (they can "protect" more vertices)
    strategies.append(low_degree_neighbors / (max_low_degree_neighbors + 1e-10) - degree / max_degree)
    
    # Strategy 20: Prefer vertices with few high-degree neighbors (less likely to be blocked by hubs)
    strategies.append(-high_degree_neighbors / (max_high_degree_neighbors + 1e-10) - degree / max_degree)
    
    # Strategy 21: Balanced: prefer low degree and moderate neighbor diversity
    strategies.append(-degree / max_degree + 0.5 * low_degree_neighbors / (max_low_degree_neighbors + 1e-10) - 0.5 * high_degree_neighbors / (max_high_degree_neighbors + 1e-10))
    
    # Strategy 22: Prefer vertices that are "central" among low-degree vertices
    strategies.append(low_degree_neighbors / (degree + 1e-10))
    
    # Strategy 23: Prefer vertices with high degree but low average neighbor degree (peripheral hubs)
    strategies.append(degree / max_degree - avg_neighbor_degree / (max_avg_neighbor_degree + 1e-10))
    
    # Strategy 24: Prefer vertices with low degree and low neighbor degree sum
    strategies.append(-degree / max_degree - neighbor_degree_sum / (max_neighbor_degree_sum + 1e-10))
    
    # Strategy 25: Prefer vertices that maximize the number of vertices they can "cover" (degree + low-degree neighbors)
    strategies.append(degree / max_degree + low_degree_neighbors / (max_low_degree_neighbors + 1e-10))
    
    # Strategy 26: Prefer vertices with low degree and high ratio of low-degree neighbors
    strategies.append(-degree / max_degree + low_degree_neighbors / (degree + 1e-10))
    
    # Strategy 27: Prefer vertices with low degree and low ratio of high-degree neighbors
    strategies.append(-degree / max_degree - high_degree_neighbors / (degree + 1e-10))
    
    # Strategy 28: Prefer vertices that are "isolated" from hubs
    strategies.append(-high_degree_neighbors / (max_high_degree_neighbors + 1e-10) + low_degree_neighbors / (max_low_degree_neighbors + 1e-10))
    
    # Strategy 29: Prefer vertices with low degree and moderate connectivity
    strategies.append(-degree / max_degree + 0.3 * (low_degree_neighbors - high_degree_neighbors) / (max_low_degree_neighbors + max_high_degree_neighbors + 1e-10))
    
    # Strategy 30: Prefer vertices that minimize the "damage" when selected (low degree, low high-degree neighbors)
    strategies.append(-degree / max_degree - 0.2 * high_degree_neighbors / (max_high_degree_neighbors + 1e-10))
    
    # Evaluate each strategy and pick the best
    best_score = None
    best_size = -1
    
    for strat in strategies:
        if time.time() > deadline:
            break
        size = simulate_greedy_fast(strat)
        if size > best_size:
            best_size = size
            best_score = strat
    
    if best_score is not None:
        return best_score
    else:
        return -degree
# EVOLVE-BLOCK-END
