"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _greedy_decode(adjacency, scores):
    """Fixed greedy decoder: pick vertices in descending score order, skipping neighbours."""
    n = adjacency.shape[0]
    if n == 0:
        return 0
    
    # Use argsort for the order
    order = np.argsort(-scores, kind='stable')
    
    picked = np.zeros(n, dtype=bool)
    forbidden = np.zeros(n, dtype=bool)
    count = 0
    
    for v in order:
        if forbidden[v]:
            continue
        picked[v] = True
        count += 1
        # Mark neighbours as forbidden
        neighbors = np.where(adjacency[v])[0]
        forbidden[neighbors] = True
        
    return count


def priority(adjacency):
    """
    Compute vertex scores for the Maximum Independent Set greedy decoder.
    
    Strategy: Monte Carlo Stochastic Game Value Estimation.
    We estimate each vertex's 'value' by simulating the greedy process multiple times
    with slightly perturbed scores (noise). Vertices that are selected in many simulations
    are considered 'robust' and are assigned higher priority. This captures non-linear
    interference effects better than static centrality because it observes the actual
    outcome of the greedy interaction.
    """
    deadline = time.time() + 115  # Safety margin before 120s kill
    adjacency = np.asarray(adjacency, dtype=bool)
    n = adjacency.shape[0]

    if n == 0:
        return np.array([])
    
    degree = adjacency.sum(axis=1).astype(float)
    
    # Base scores: Start with low degree (good for MIS) and small random noise to break ties
    # We use a base that is already decent, then refine with MC.
    # Low degree is a strong prior.
    base_scores = -degree
    
    # We will run multiple Monte Carlo simulations.
    # In each simulation, we add Gaussian noise to the base scores.
    # The magnitude of noise is tuned to explore the "neighborhood" of the greedy solution.
    # If a vertex is picked often despite the noise, it is a "core" vertex.
    
    # Count how many times each vertex is picked in the simulations
    pick_counts = np.zeros(n, dtype=float)
    
    # Number of simulations: balance between time and statistical accuracy.
    # 200 nodes, 120s limit. _greedy_decode is O(N^2) worst case (200^2 = 40k ops).
    # 40k ops is very fast in NumPy/Python loop hybrid.
    # Let's aim for 500-1000 simulations if time permits.
    
    max_sims = 1000
    noise_std = 0.5  # Relative to degree scale? Degrees are ~0-100. Let's scale noise.
    # Actually, if base is -degree, range is ~100. Noise of 0.5 is tiny.
    # We want noise large enough to swap close vertices but not reorder hubs completely.
    # Let's use noise proportional to the spread of degrees.
    deg_std = degree.std()
    if deg_std < 1e-6:
        deg_std = 1.0
    
    # We want to explore local perturbations.
    # Let's use a noise std that is ~10% of the degree standard deviation?
    # Or just a fixed small value relative to the range.
    # Let's try noise_std = 1.0 for degrees around 10.
    # If degrees are 0-50, std is ~5. Noise 1.0 is reasonable.
    
    current_noise_std = max(0.5, deg_std * 0.1)
    
    # Adaptive noise: Start with larger noise to explore, then shrink?
    # Or just fixed. Fixed is simpler and less prone to getting stuck in local noise patterns.
    # Let's use a fixed noise_std derived from the graph.
    
    # To make it more robust, let's use a few different noise levels?
    # No, keep it simple. One noise level.
    
    # However, the "Stochastic Game" idea suggests that the *order* matters.
    # By adding noise, we are effectively sampling different greedy orders.
    # The frequency of selection is a proxy for "how much this vertex wants to be in the MIS
    # given the competition from its neighbors".
    
    # Optimization: Vectorize the simulation loop?
    # Hard to vectorize the sequential greedy step easily without custom C code.
    # But N=200 is small. A Python loop of 200 steps is ~200 operations.
    # 1000 sims * 200 steps = 200,000 steps.
    # Each step involves numpy ops on arrays of size 200.
    # This should be well within 120s.
    
    # Let's run the simulations.
    # We need to track the count of picks.
    
    # To speed up, we can pre-compute neighbor lists?
    # adjacency[v] is already a boolean vector. np.where is fast.
    
    # Let's add a time check inside the loop to ensure we don't exceed deadline.
    
    for sim in range(max_sims):
        if time.time() > deadline:
            break
        
        # Generate noisy scores
        # Use a different random seed for each sim?
        # Yes, for independence.
        # We don't need to store the noisy scores, just pass them to decoder.
        # But _greedy_decode takes scores.
        
        # To avoid regenerating the full array if possible?
        # scores = base_scores + noise
        noise = np.random.normal(0, current_noise_std, n)
        noisy_scores = base_scores + noise
        
        # Run greedy decode
        # We only need the count? No, we need to know WHICH vertices were picked to update pick_counts.
        # So we need a modified decoder that returns the picked mask or list.
        
        # Let's write an inline fast decoder that returns the picked mask.
        # Or modify _greedy_decode? No, keep it pure.
        # Inline implementation for speed and to return mask.
        
        order = np.argsort(-noisy_scores, kind='stable')
        forbidden = np.zeros(n, dtype=bool)
        
        for v in order:
            if forbidden[v]:
                continue
            pick_counts[v] += 1
            # Mark neighbors
            neighbors = np.where(adjacency[v])[0]
            forbidden[neighbors] = true_mask = np.ones(len(neighbors), dtype=bool) # Just use direct assignment
            forbidden[neighbors] = True
            
    # Normalize pick_counts to 0..1
    total_sims = sim + 1 if max_sims > 0 else 1
    if total_sims == 0:
        return base_scores
    
    survival_prob = pick_counts / total_sims
    
    # The score for the final greedy decoder should reflect this "survival probability".
    # Vertices with high survival probability should be picked first.
    # Why? Because they are robustly part of the MIS.
    # If we pick them first in the *final* deterministic run, we secure them.
    # Their neighbors (which have lower survival probs) will be blocked.
    
    # We can combine this with the original degree heuristic?
    # The survival probability already encodes the degree effects (low degree -> high survival).
    # But adding a small tiebreaker based on degree might help if probabilities are close.
    
    # Let's just use the survival probability as the score.
    # Higher score = picked earlier.
    final_scores = survival_prob
    
    # One potential issue: If the graph is very sparse, many vertices have similar high survival.
    # If dense, low survival.
    # The ranking is what matters.
    
    # Let's verify with a quick decode on the final scores to ensure validity?
    # The evaluator does this. We just return the scores.
    
    # Optional: Polish the final scores with a few local swaps to see if we can squeeze more out?
    # The MC estimate might have noise. A local search on the *final* ordering
    # using the *true* greedy decoder (no noise) could refine the result.
    
    # Let's try a few iterations of local swap optimization on the final scores.
    best_count = _greedy_decode(adjacency, final_scores)
    best_scores = final_scores.copy()
    
    # Time remaining?
    if time.time() < deadline - 10:
        # Do local search
        # We can try swapping pairs in the score vector.
        # Since n=200, n^2/2 = 20,000 pairs.
        # Each swap requires a full decode (O(N^2) ~ 40k ops).
        # 20k * 40k = 800M ops. Might be too slow in Python.
        # Let's limit the number of swaps or use a smarter heuristic.
        
        # Instead of all pairs, try swapping a random subset or just the top/bottom?
        # Or use the "hill climbing" approach:
        # Pick a random pair, swap, if better keep, else revert.
        
        max_local_iters = 500
        for _ in range(max_local_iters):
            if time.time() > deadline:
                break
            
            # Pick two random vertices to swap
            i = np.random.randint(0, n)
            j = np.random.randint(0, n)
            if i == j:
                continue
            
            # Swap
            best_scores[i], best_scores[j] = best_scores[j], best_scores[i]
            
            # Evaluate
            count = _greedy_decode(adjacency, best_scores)
            
            if count > best_count:
                best_count = count
                # Keep swap
            else:
                # Revert
                best_scores[i], best_scores[j] = best_scores[j], best_scores[i]
                
    return best_scores
# EVOLVE-BLOCK-END
