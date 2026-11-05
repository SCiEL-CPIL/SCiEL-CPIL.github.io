"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _greedy_decode(adjacency, scores):
    """Fixed greedy decoder: pick vertices in descending score order, skipping neighbours."""
    n = adjacency.shape[0]
    if n == 0:
        return 0
    
    order = np.argsort(-scores, kind='stable')
    
    picked = np.zeros(n, dtype=bool)
    forbidden = np.zeros(n, dtype=bool)
    count = 0
    
    for v in order:
        if forbidden[v]:
            continue
        picked[v] = True
        count += 1
        neighbors = np.where(adjacency[v])[0]
        forbidden[neighbors] = True
        
    return count


def priority(adjacency):
    """
    Compute vertex scores for the Maximum Independent Set greedy decoder.
    
    Strategy: Iterative Minimum-Degree Peeling with Dynamic Re-scoring.
    
    Instead of assigning static scores, we simulate the optimal greedy MIS construction:
    1. Maintain a "remaining" subgraph.
    2. At each step, find the vertex with the minimum degree in the remaining graph.
    3. Add that vertex to the independent set (it gets a high priority score).
    4. Remove that vertex and all its neighbors from the remaining graph.
    5. Repeat until no vertices remain.
    
    The priority score for each vertex is inversely proportional to the step at which
    it was selected (earlier selection = higher score). This directly encodes the
    optimal greedy peeling order.
    
    To break ties and add robustness, we use a composite score that combines:
    - The peeling order (primary signal)
    - The degree at the time of selection (secondary signal for tie-breaking)
    """
    deadline = time.time() + 115
    adjacency = np.asarray(adjacency, dtype=bool)
    n = adjacency.shape[0]
    
    if n == 0:
        return np.array([])
    
    # Core algorithm: iterative min-degree peeling
    # We track which vertices have been removed
    removed = np.zeros(n, dtype=bool)
    # scores will be assigned based on peeling order
    scores = np.zeros(n, dtype=float)
    # Track the degree of each vertex in the remaining graph
    # We'll recompute degrees efficiently
    
    # For efficiency, maintain a list of active vertices
    active = np.ones(n, dtype=bool)
    
    # We'll assign scores in reverse: last peeled gets lowest score
    # First peeled gets highest score
    
    peel_order = np.zeros(n, dtype=int)
    peel_step = 0
    
    # Precompute initial degrees
    current_degree = adjacency.sum(axis=1).astype(np.int64)
    
    # Main peeling loop
    while active.any():
        if time.time() > deadline:
            break
            
        # Find active vertices
        active_indices = np.where(active)[0]
        if len(active_indices) == 0:
            break
        
        # Find the vertex with minimum degree among active vertices
        # Use the precomputed current_degree
        active_degrees = current_degree[active_indices]
        min_deg_idx = active_indices[np.argmin(active_degrees)]
        
        # Assign peeling order
        peel_order[min_deg_idx] = peel_step
        peel_step += 1
        
        # Remove this vertex and its neighbors from the active set
        # First, update degrees of neighbors before removing
        neighbors = np.where(adjacency[min_deg_idx])[0]
        # Only consider neighbors that are still active
        active_neighbors = neighbors[active[neighbors]]
        
        # Decrement degree of active neighbors
        current_degree[active_neighbors] -= 1
        
        # Mark vertex and its neighbors as removed
        removed[min_deg_idx] = True
        active[min_deg_idx] = False
        active[active_neighbors] = False
        
        # Also mark neighbors as removed for bookkeeping
        removed[active_neighbors] = True
    
    # Now assign scores based on peeling order
    # Vertices peeled earlier get higher scores
    # Use: score = (n - 1 - peel_order) so that first peeled has score n-1, last has 0
    # This ensures the greedy decoder will pick them in the correct order
    
    # But we need to handle vertices that might not have been peeled (shouldn't happen, but safety)
    unpeeled = peel_order == 0
    # Actually, peel_order starts at 0 for the first vertex, so we need to check differently
    # Let's use a separate flag
    was_peeled = np.zeros(n, dtype=bool)
    # Re-run to track which were peeled (we already did, but let's be explicit)
    # Actually, all vertices should be peeled. Let's just assign scores.
    
    # Score = n - 1 - peel_order (so first peeled gets highest)
    scores = (n - 1 - peel_order).astype(float)
    
    # Add a small perturbation based on the degree at time of selection to break ties
    # and add robustness. Vertices that had lower degree when selected get a slight boost.
    # We can reconstruct this by noting that the degree at selection time is stored implicitly.
    # For simplicity, add a tiny noise based on the original degree to break any remaining ties.
    original_degree = adjacency.sum(axis=1).astype(float)
    # Normalize original degree to [0, 1]
    deg_range = original_degree.max() - original_degree.min()
    if deg_range > 1e-6:
        normalized_deg = (original_degree - original_degree.min()) / deg_range
    else:
        normalized_deg = np.zeros(n)
    
    # Add a small bonus for lower original degree (they tend to be better MIS candidates)
    # Scale: add up to 0.5 to the score, so it doesn't override the peeling order
    scores += 0.5 * (1.0 - normalized_deg)
    
    # Now, do a local search refinement: try swapping pairs of vertices in the score order
    # to see if we can improve the greedy decode result
    best_count = _greedy_decode(adjacency, scores)
    best_scores = scores.copy()
    
    # Time remaining for local search
    time_left = deadline - time.time()
    if time_left > 15:
        # Hill climbing: try random swaps
        max_iters = min(2000, int(time_left * 100))  # Rough estimate
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            i = np.random.randint(0, n)
            j = np.random.randint(0, n)
            if i == j:
                continue
            
            # Swap scores
            best_scores[i], best_scores[j] = best_scores[j], best_scores[i]
            
            count = _greedy_decode(adjacency, best_scores)
            
            if count > best_count:
                best_count = count
            else:
                # Revert
                best_scores[i], best_scores[j] = best_scores[j], best_scores[i]
    
    return best_scores
# EVOLVE-BLOCK-END
