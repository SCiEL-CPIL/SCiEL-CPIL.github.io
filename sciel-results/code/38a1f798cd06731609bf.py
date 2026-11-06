import numpy as np
import time

def priority(adjacency):
    """
    Computes vertex scores for the greedy MIS decoder.
    Uses a value-based approach: for each vertex, estimate how much
    independent set size we gain by selecting it vs. not.
    Also uses a simulated greedy pass to identify "valuable" vertices.
    """
    adjacency = np.asarray(adjacency, dtype=float)
    n = adjacency.shape[0]
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    if n == 0:
        return np.array([])
    
    degrees = adjacency.sum(axis=1).astype(int)
    
    # Simple baseline: prefer low degree
    scores = -degrees.astype(float).copy()
    
    # Strategy 1: Simulate the greedy decoder multiple times with different
    # tie-breaking to identify which vertices are consistently selected.
    # Vertices that are selected in many simulations are "valuable".
    
    # Also use a local improvement heuristic:
    # For each vertex v, compute the "marginal gain" of selecting v:
    # gain(v) = 1 - (number of neighbors that would be blocked)
    # But this is circular. Instead, use iterative refinement.
    
    # Approach: Use a weighted degree-based score with neighbor competition.
    # The key insight: a vertex is valuable if it has low degree (few neighbors to block)
    # and if its neighbors are not themselves very valuable.
    
    # Let's use a fixed-point iteration with a different formulation:
    # score[v] = base[v] - penalty from neighbors
    # where base[v] = -degree[v] (prefer low degree)
    # and penalty = sum over neighbors of their "attractiveness"
    
    # But more importantly, let's simulate the actual greedy process
    # and use that to refine scores.
    
    # First, do a few iterations of score refinement
    alpha = 0.5
    beta = 2.0
    iterations = 30
    
    for it in range(iterations):
        if time.time() > deadline - 20:
            break
        
        new_scores = np.zeros(n)
        for v in range(n):
            neighbors = np.where(adjacency[v] > 0)[0]
            base = -degrees[v]
            if len(neighbors) > 0:
                # Penalty: sum of neighbor scores (if neighbors are good, v is less good)
                # Use a soft sum to avoid too much sensitivity
                neighbor_sum = np.sum(scores[neighbors])
                penalty = beta * neighbor_sum / (len(neighbors) + 1)
            else:
                penalty = 0.0
            target = base - penalty
            new_scores[v] = alpha * scores[v] + (1 - alpha) * target
        
        scores = new_scores
        scores -= np.mean(scores)
    
    # Now, simulate the greedy decoder to see which vertices get picked
    # and use that to boost scores.
    
    # Run greedy with current scores
    selected = set()
    order = np.argsort(-scores)  # descending score
    for v in order:
        if v in selected:
            continue
        # Check if any neighbor is already selected
        neighbors = np.where(adjacency[v] > 0)[0]
        conflict = False
        for u in neighbors:
            if u in selected:
                conflict = True
                break
        if not conflict:
            selected.add(v)
    
    # Vertices that were selected get a boost
    # Vertices that were blocked by high-scoring neighbors get a penalty
    # But actually, we want to identify vertices that are "key" - 
    # selecting them blocks many low-value neighbors
    
    # Alternative: use the greedy simulation to identify "wasted" selections
    # A vertex is wasted if it blocks many neighbors that could have been selected instead
    
    # Let's try a different approach: compute for each vertex the number of
    # neighbors it would block, weighted by their scores.
    # High-value vertices block low-value neighbors -> good
    # Low-value vertices block high-value neighbors -> bad
    
    # Final score: combine structural score with blocking value
    blocking_value = np.zeros(n)
    for v in range(n):
        neighbors = np.where(adjacency[v] > 0)[0]
        if len(neighbors) > 0:
            # How much value do we lose by blocking these neighbors?
            # If neighbors have low scores, blocking them is cheap
            # If neighbors have high scores, blocking them is expensive
            neighbor_avg_score = np.mean(scores[neighbors])
            # We want to select v if its own score is high AND neighbors are low
            blocking_cost = neighbor_avg_score
            blocking_value[v] = scores[v] - blocking_cost
        else:
            blocking_value[v] = scores[v]
    
    # Combine
    final_scores = 0.5 * scores + 0.5 * blocking_value
    final_scores -= np.mean(final_scores)
    
    # Boost isolated vertices
    isolated = degrees == 0
    if np.any(isolated):
        final_scores[isolated] += 5.0
    
    # Ensure isolated vertices are highest priority
    if np.any(isolated):
        final_scores[isolated] = np.max(final_scores) + 1
    
    return final_scores
