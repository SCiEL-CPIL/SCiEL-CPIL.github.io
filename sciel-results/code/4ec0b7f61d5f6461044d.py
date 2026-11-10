"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, weights):
    """
    Computes attractiveness scores for items in a Multiple-Constraint Knapsack Problem (MCKP).
    
    Strategy: Context-Aware Marginal Value Estimation.
    1. We simulate a greedy packing of all items except the candidate item.
    2. This gives us a 'residual capacity' vector that represents the 'tightness' of the knapsack 
       without that item.
    3. The score for the item is its value divided by the maximum resource it would consume 
       relative to that specific residual state.
    
    This approach differentiates items based on how 'expensive' they are in terms of the 
    remaining slack of a near-optimal packing, rather than just their static density.
    """
    
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin for the 300s limit
    
    n_constraints, n_items = weights.shape
    capacity = 25.0
    
    # Total prize per item (sum across constraints, assuming total prize is sum of components)
    item_values = prizes.sum(axis=0)
    
    # Pre-compute base attributes for a fast initial greedy pass to estimate "typical" packing
    # We use a simple value-density ratio to guide the baseline greedy simulation
    # This is just to get a "reference" packing state quickly.
    base_density = item_values / (weights.max(axis=0) + 1e-9)
    
    # We will estimate the marginal value for each item.
    # To do this efficiently, we first run a single "global" greedy packing to see which items are 
    # typically selected and what the final residual capacity looks like.
    # However, the prompt suggests: "simulating a greedy packing of the remaining items (excluding the candidate)"
    # Doing this for every item is O(N^2 * C) per simulation.
    # N=100. 100 items * (100 items * 5 constraints) = 50,000 ops per full set of evaluations.
    # This is very fast. We can afford to do this for every item, potentially multiple times if we add noise.
    
    # Let's refine the strategy:
    # For each item j:
    #   1. Create a list of all other items.
    #   2. Run a greedy algorithm on this list to fill the knapsack as much as possible.
    #      Greedy criterion: Pick the item with the highest value/weight ratio that fits.
    #      (Or simply: highest value density).
    #   3. Record the final residual capacity vector `rem_cap_j`.
    #   4. Calculate the "marginal density" of item j given `rem_cap_j`:
    #      If item j fits in `rem_cap_j`, its marginal cost is its weight.
    #      If it doesn't fit, we can still score it based on how close it is to fitting, 
    #      or penalize it heavily.
    #      
    # Actually, a simpler and more robust interpretation of "marginal value" in this context:
    # The value of an item is its contribution to the total prize.
    # The "cost" is the reduction in the number of *other* items we could have added.
    # 
    # Let's stick to the prompt's specific hint: 
    # "score as the ratio of its prize to the maximum resource it would consume given that specific residual state"
    # 
    # Residual state R_j is the capacity left after packing all items except j greedily.
    # If we add item j, the "effective weight" might be interpreted as the weight vector w_j.
    # But if w_j > R_j, it doesn't fit.
    # 
    # Let's try a different angle that is computationally similar but perhaps more aligned with "attractiveness":
    # 1. Run a greedy packing of ALL items to get a set S and residual R.
    # 2. For items in S, their marginal value is high because they are part of the solution.
    # 3. For items not in S, their marginal value is lower.
    # 
    # But we need a score for ALL items.
    # 
    # Let's implement the "Leave-One-Out" greedy simulation as described in the directive.
    # For each item k in 0..N-1:
    #   remaining_items = [i for i in range(N) if i != k]
    #   rem_cap = np.full(C, 25.0)
    #   # Greedy fill remaining_items
    #   # Sort remaining_items by value density (pre-computed) for efficiency?
    #   # Or just iterate? Sorting 99 items is fast.
    #   
    #   # To make it stochastic and robust, we can run this greedy multiple times with slight noise
    #   # or just use a deterministic greedy based on value density.
    #   # Given N=100, deterministic greedy is very fast.
    
    scores = np.zeros(n_items)
    
    # Pre-sort indices by base density descending for fast greedy access
    # We will use this order to greedily pick items.
    order_by_density = np.argsort(base_density)[::-1]
    
    # We will run this leave-one-out simulation.
    # To add some robustness against the specific greedy order, we can add a small random perturbation
    # to the densities for each simulation, or just rely on the density order.
    # Let's try a hybrid: 50% deterministic density order, 50% random order?
    # No, let's stick to density order for consistency, but maybe run a few variations.
    
    # Optimization: 
    # Instead of re-sorting every time, we pre-compute the density order.
    # For each k, we iterate through order_by_density, skipping k.
    
    for k in range(n_items):
        if time.time() > deadline:
            # Fallback to static density if we run out of time
            break
            
        rem_cap = np.full(n_constraints, capacity)
        
        # Greedily pack all items except k
        # We iterate through the pre-sorted list of indices
        for idx in order_by_density:
            if idx == k:
                continue
            # Check if idx fits
            # weights[:, idx] is the weight vector for item idx
            if np.all(weights[:, idx] <= rem_cap):
                rem_cap -= weights[:, idx]
                
        # Now we have the residual capacity rem_cap after packing everything except k.
        # We want to score item k.
        # The "maximum resource it would consume" is simply its weight vector w_k.
        # But how do we define the ratio?
        # If w_k fits in rem_cap, it can be added. Its "cost" is w_k.
        # If w_k doesn't fit, it can't be added in this state.
        
        # A good heuristic for "attractiveness" in ACO is often value / cost.
        # Here, cost is the weight.
        # However, if the item is very large, it consumes a lot of the *remaining* slack.
        
        # Let's define the score as:
        # value_k / (sum of weights of k relative to the tightest constraint's residual)
        # Or simpler: value_k / max(weights[:, k])
        # But this is static.
        
        # The prompt says: "ratio of its prize to the maximum resource it would consume given that specific residual state"
        # This phrasing is slightly ambiguous. "Maximum resource it would consume" is just its weight.
        # "Given that specific residual state" implies the denominator should depend on rem_cap.
        
        # Perhaps: value_k / (w_k / rem_cap).max() ? i.e. value / (fraction of capacity used)
        # If rem_cap is small, the fraction is large, score is low.
        # If rem_cap is large, fraction is small, score is high.
        # This rewards items that fit easily into the "leftover" space.
        
        # Let's try: score_k = value_k / np.max(weights[:, k] / (rem_cap + 1e-9))
        # This normalizes the weight by the available slack.
        
        # If rem_cap is close to 0, the denominator is huge -> score ~ 0.
        # If rem_cap is large, denominator is small -> score is high.
        
        # This seems to favor items that are "small" relative to the leftover space.
        # This is a valid heuristic for "filling the gaps".
        
        # Let's also consider that if the item *doesn't* fit, the score should be low.
        # The formula above handles this: if w_k > rem_cap, w_k/rem_cap > 1, score is lower.
        
        # To avoid division by zero or extremely high values, we clamp.
        ratios = weights[:, k] / (rem_cap + 1e-9)
        max_ratio = np.max(ratios)
        
        # If max_ratio is very large, it means the item is much bigger than the slack.
        # We want the score to be low.
        # score = value / max_ratio
        
        scores[k] = item_values[k] / (max_ratio + 1e-9)
        
    # If we didn't finish all items, fill the rest with a fallback static heuristic
    if time.time() > deadline:
        # Fallback: simple density
        fallback = item_values / (weights.max(axis=0) + 1e-9)
        # Check which items were not scored (scores are 0)
        # Actually, if we broke out, some scores are 0.
        # We should probably just return the fallback for all if we ran out of time early?
        # Or just fill the 0s.
        zero_mask = scores < 1e-6
        if np.any(zero_mask):
            scores[zero_mask] = fallback[zero_mask]
            
    # Ensure all scores are positive and finite
    scores = np.where(np.isfinite(scores), scores, 1e-9)
    scores = np.maximum(scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END
