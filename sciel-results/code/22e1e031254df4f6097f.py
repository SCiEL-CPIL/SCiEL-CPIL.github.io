"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multiple-constraint knapsack.
    
    Strategy: Forward-Looking Local Search (Greedy Hill-Climbing) Frequency.
    
    1. Problem Framing:
       The previous strategy used "regret" (marginal loss of exclusion). 
       This new strategy uses "selection frequency" in a simulated greedy packing.
       
       Intuition:
       - An ACO ant builds a solution sequentially. Items that are frequently part 
         of high-quality greedy solutions are likely to be part of high-quality 
         ACO solutions.
       - By simulating multiple greedy packings with slightly different priority 
         weights, we identify a "core set" of items that are robustly good choices 
         regardless of minor perturbations in the priority ordering.
       - This is forward-looking: it answers "what would a smart greedy agent pick?" 
         rather than "how much did we lose by removing this?"

    2. Method:
       a. Define a base priority score for each item. We use a combination of 
          prize and inverse weight consumption to create a robust density metric.
          priority_i = prizes_i / (1 + sum(weights_i))
          
       b. Perform K simulations of a greedy packing algorithm.
          In each simulation k:
          - Add small random noise to the priority scores to break ties and explore 
            slightly different orderings.
          - Sort items by perturbed priority.
          - Iterate through sorted items, adding them to the knapsack if they fit 
            all constraints.
          - Record which items were selected.
          
       c. The heuristic score for each item is its selection frequency across the 
          K simulations, normalized.
          
       d. To ensure the heuristic is strictly positive (as required by ACO), we add 
          a small epsilon.

    3. Complexity:
       - K simulations.
       - Each simulation involves sorting N items (O(N log N)) and a linear scan 
         with O(N * C) feasibility checks.
       - With N=100, C=5, and K=50, this is 50 * (100*7 + 100*5) ~ 60,000 operations.
       - This is extremely fast (< 10ms in NumPy/Python).
       
       We can increase K to 200 or 500 to get a more stable frequency estimate 
       without significantly impacting runtime.
    """
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Handle shapes: prizes (5, 100) -> (100,), weights (5, 100)
    if prizes.ndim == 2:
        prizes = prizes.sum(axis=0)
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
    
    n_items = weights.shape[1]
    n_constraints = weights.shape[0]
    capacity = 25.0
    
    # Time budget guard
    start_time = time.time()
    time_limit = 10.0  # Should be much faster
    
    # Base priority: Prize / (1 + total weight consumption)
    # This penalizes items that use a lot of capacity across all constraints.
    total_weights = np.sum(weights, axis=0)
    base_priority = prizes / (1.0 + total_weights)
    
    # Normalize base priority to [0, 1] to control noise scale
    max_base = np.max(base_priority)
    if max_base > 0:
        base_priority_norm = base_priority / max_base
    else:
        base_priority_norm = np.ones(n_items)
    
    # Selection frequency counter
    freq = np.zeros(n_items)
    
    # Number of simulations
    # We can afford many simulations. Let's do 100.
    n_sims = 100
    
    # Pre-allocate used weights array for reuse? No, we need to reset it each sim.
    # But we can optimize the inner loop.
    
    for k in range(n_sims):
        if time.time() - start_time > time_limit:
            break
        
        # Add Gaussian noise to priority to explore different orderings
        # Scale noise relative to the priority range
        noise_scale = 0.1
        perturbed_priority = base_priority_norm + np.random.normal(0, noise_scale, n_items)
        
        # Sort indices by perturbed priority descending
        sorted_indices = np.argsort(-perturbed_priority)
        
        # Greedy packing
        used_weights = np.zeros(n_constraints)
        
        for idx in sorted_indices:
            # Check if item fits
            # weights[:, idx] has shape (C,)
            # Vectorized check: all(used_weights + weights[:, idx] <= capacity)
            if np.all(used_weights + weights[:, idx] <= capacity):
                used_weights += weights[:, idx]
                freq[idx] += 1.0
        
    # Normalize frequency to [0, 1]
    max_freq = np.max(freq)
    if max_freq > 0:
        normalized_freq = freq / max_freq
    else:
        # Fallback if no items were selected (should not happen)
        normalized_freq = base_priority_norm
    
    # Construct heuristic scores
    # We want high frequency items to have high scores.
    # A linear mapping is sufficient, but a slight power can sharpen the focus 
    # on the most critical items.
    # Let's use a power of 1.0 (linear) to be safe, or 1.5 to emphasize core.
    # Given ACO multiplies pheromone * heuristic, and pheromone starts at 1,
    # a stronger heuristic for core items helps them get picked early.
    
    scores = 1e-9 + (normalized_freq ** 1.5)
    
    # Ensure positive and finite
    scores = np.where(scores > 0, scores, 1e-9)
    scores = np.where(np.isfinite(scores), scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END
