"""Combined strategy: Local Search Optimization with fast vectorized simulation."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    """
    Priority score for greedy maximum independent set.
    
    Strategy: Local Search Optimization (2-Opt Swap) with vectorized simulation.
    1. Initialize scores based on a strong static heuristic (Low Degree + Hub Penalty).
    2. Simulate the greedy decoding to get a baseline Independent Set size.
    3. Iteratively attempt to improve the set size by swapping the priority scores of pairs of vertices.
       - Use a fast, vectorized simulation of the greedy decoder.
       - Accept swaps that strictly improve the independent set size.
       - If stuck, perform a random restart or perturbation.
    4. Return the final scores.
    
    This approach directly optimizes the outcome of the fixed greedy decoder by
    adjusting the vertex ordering, rather than relying solely on static structural features.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit (10s buffer)
    
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])
    
    # 1. Initial Heuristic: Low Degree + Hub Penalty
    # We want low degree vertices to have high priority (picked first).
    # Penalize vertices connected to high-degree hubs.
    deg = adj.sum(axis=1).astype(float)
    neighbor_deg_sum = (adj @ deg)
    safe_deg = deg + 1.0
    hub_penalty = neighbor_deg_sum / safe_deg
    
    # Initial score: primarily -degree, with a small penalty for connecting to hubs
    scores = -deg - 0.1 * hub_penalty
    
    # Pre-compute neighbor lists for potential use, though vectorized sim might be faster
    # Let's stick to vectorized simulation for speed.
    
    def simulate_fast(s):
        """
        Simulate the greedy decoder using vectorized operations.
        Returns the size of the independent set.
        """
        order = np.argsort(-s, kind='stable')
        active = np.ones(n, dtype=bool)
        count = 0
        
        for v in order:
            if active[v]:
                count += 1
                # Mark v and its neighbors as inactive
                active[v] = False
                # Vectorized neighbor update
                # adj[v] is a boolean array of neighbors
                # We need to set active[neighbors] = False
                # This is O(N) per selection, so O(N^2) total.
                # With N=200, this is 40,000 ops per simulation.
                # In Python/NumPy, this should be fast enough.
                active[adj[v]] = False
        
        return count

    # 2. Baseline
    best_val = simulate_fast(scores)
    
    # 3. Local Search: Swap pairs
    # We will iterate until the deadline.
    # To avoid getting stuck in local optima, we can use a simple restart strategy
    # or just continue with random swaps.
    
    rng = np.random.default_rng(42)
    
    # Keep track of the best scores found
    best_scores = scores.copy()
    
    no_improve_count = 0
    max_no_improve = 5000  # If no improvement for 5000 swaps, restart with perturbation
    
    while time.time() < deadline:
        # Pick two random vertices to swap
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
            
        # Swap scores
        scores[i], scores[j] = scores[j], scores[i]
        
        # Evaluate
        new_val = simulate_fast(scores)
        
        if new_val > best_val:
            best_val = new_val
            best_scores = scores.copy()
            no_improve_count = 0
        else:
            # Revert
            scores[i], scores[j] = scores[j], scores[i]
            no_improve_count += 1
            
            # If stuck for a while, perturb the scores to escape local optimum
            if no_improve_count > max_no_improve:
                # Randomly perturb a few scores
                num_perturb = min(10, n // 10)
                if num_perturb > 0:
                    perturb_indices = rng.choice(n, size=num_perturb, replace=False)
                    scores[perturb_indices] += rng.normal(0, 1.0, size=num_perturb)
                    # Check if perturbation helps
                    perturbed_val = simulate_fast(scores)
                    if perturbed_val > best_val:
                        best_val = perturbed_val
                        best_scores = scores.copy()
                        no_improve_count = 0
                    else:
                        # Revert perturbation
                        scores = best_scores.copy()
                        no_improve_count = 0 # Reset counter after restart attempt
        
    return best_scores
# EVOLVE-BLOCK-END
