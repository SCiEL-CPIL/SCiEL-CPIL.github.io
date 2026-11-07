"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, weights):
    """
    Computes attractiveness scores using a Probabilistic Greedy Simulation.
    
    Strategy:
    Instead of a static Lagrangian relaxation which assumes linear penalties,
    we simulate the competitive selection process. We estimate how likely an item
    is to be 'selected' by a greedy agent that balances prize against resource
    consumption. 
    
    1. We define a 'greedy score' for each item: s_i = total_prize_i / (1 + total_weight_i).
       This approximates the value density while penalizing heavy items.
       
    2. We perform a 'soft' greedy selection simulation. We don't just pick the top K, 
       but estimate the probability that each item survives a competition where 
       resources are finite. 
       
    3. The heuristic value is proportional to the estimated marginal contribution 
       of the item, which is higher for items that are 'efficient' and likely to be 
       picked early in the greedy process, and lower for items that are dominated 
       by more efficient alternatives.
       
    Specifically, we use a temperature-based softmax over the greedy scores to 
    estimate the 'weight' of each item in an optimal-like subset. We then scale 
    this by the item's own prize to ensure high-value items remain attractive, 
    but penalized if they are not efficient.
    
    This approach captures the non-linear interactions of the constraints better 
    than linear Lagrangian multipliers because it respects the 'knapsack' nature 
    (discrete selection) more closely via the simulation logic.
    """
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Safety check for shape
    if prizes.shape != (5, 100) or weights.shape != (5, 100):
        # Fallback to simple ratio
        total_prizes = np.sum(prizes, axis=0)
        total_weights = np.sum(weights, axis=0)
        heuristic = total_prizes / (total_weights + 1e-9)
        return np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)

    # 1. Calculate Basic Metrics
    total_prizes_per_item = np.sum(prizes, axis=0) # (100,)
    total_weights_per_item = np.sum(weights, axis=0) # (100,)
    
    # 2. Greedy Score: Value Density with Regularization
    # We use 1 + weight to avoid division by zero and to naturally penalize weight.
    # A higher score means this item is 'better' per unit of resource consumption.
    greedy_scores = total_prizes_per_item / (1.0 + total_weights_per_item)
    
    # 3. Probabilistic Selection Simulation
    # We simulate a 'soft' selection. 
    # If we were to pick items greedily, the order is determined by greedy_scores.
    # However, the actual selection depends on remaining capacity.
    # 
    # Heuristic Insight: Items with high greedy scores are more likely to be included 
    # in the final solution. Items with low greedy scores are 'sacrificed'.
    # We want the heuristic to reflect this 'survival' likelihood.
    #
    # We use a softmax with a temperature T to create a probability distribution 
    # over items. A lower T makes the distribution more peaked (greedy-like), 
    # a higher T makes it flatter.
    #
    # Given 100 items and capacity 25 (avg weight 0.5 -> ~50 items fit if no constraint, 
    # but multi-constraint reduces this), we expect roughly 20-30 items to be selected 
    # in a good solution.
    #
    # Let's estimate the 'effective number' of selected items.
    # With 5 constraints, cap 25, avg weight 0.5, the bottleneck is likely the tightest 
    # constraint. 
    # Let's assume an effective selection size K ~ 25.
    # 
    # We can approximate the 'importance' of an item by its rank in the greedy order, 
    # but smoothed. 
    # 
    # Alternative: Use the softmax probabilities directly as the heuristic, 
    # scaled by the prize. This ensures that items that are 'probable winners' 
    # in a greedy contest get high scores.
    
    # Normalize greedy scores for numerical stability in softmax
    # We want to differentiate between 'good' and 'bad' items.
    # If we use raw greedy scores, the differences might be small.
    # Let's scale them.
    
    # A robust way: Rank the items.
    # Rank 1 is the best greedy item. Rank 100 is the worst.
    # We want items with better ranks to have higher heuristics.
    # But we also want to account for the actual prize magnitude.
    
    # Let's try a combination:
    # 1. Compute softmax probabilities based on greedy scores.
    # 2. Multiply by total prize.
    
    # To make the softmax meaningful, we need to set a temperature.
    # If T is too high, all items look the same. If T is too low, it's pure greedy.
    # The ACO pheromone will handle the 'memory' part. The heuristic should provide 
    # the 'greedy' bias.
    
    # Let's set T such that the top ~25 items have significantly higher probability.
    # Standard deviation of greedy scores?
    gs_mean = np.mean(greedy_scores)
    gs_std = np.std(greedy_scores)
    
    if gs_std < 1e-9:
        # All items are identical in greedy score. Fall back to prize.
        heuristic = total_prizes_per_item
    else:
        # Temperature: We want the distribution to be somewhat sharp.
        # Let's use T = gs_std / 2. This makes the top items ~2 std devs above mean 
        # have significantly higher weight.
        T = gs_std / 2.0
        
        # Softmax
        # exp((score - max) / T)
        shifted_scores = (greedy_scores - np.max(greedy_scores)) / T
        # Clip to avoid underflow/overflow issues, though exp of large negative is 0
        shifted_scores = np.clip(shifted_scores, -50, 0)
        
        probs = np.exp(shifted_scores)
        sum_probs = np.sum(probs)
        
        if sum_probs < 1e-9:
            # Fallback
            heuristic = total_prizes_per_item
        else:
            probs = probs / sum_probs
            
            # The heuristic should be high for items that are both:
            # a) Likely to be selected (high prob)
            # b) High value (high prize)
            # 
            # If we just use probs, we lose the absolute value scale.
            # If we just use prize, we lose the efficiency signal.
            # 
            # Let's use: heuristic_i = prize_i * (1 + probs_i * K)
            # Where K is a scaling factor. 
            # Actually, simpler: heuristic_i = prize_i * probs_i^alpha
            # This boosts efficient items.
            
            # Let's try: heuristic = total_prizes_per_item * (1.0 + 10.0 * probs)
            # This ensures that even if an item has a high prize, if it's inefficient 
            # (low prob), its score is dampened relative to an efficient item with 
            # slightly lower prize.
            
            heuristic = total_prizes_per_item * (1.0 + 10.0 * probs)
            
    # Ensure positivity and finiteness
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Safety check for time
    if time.time() - start_time > 290:
        total_prizes = np.sum(prizes, axis=0)
        total_weights = np.sum(weights, axis=0)
        heuristic = total_prizes / (total_weights + 1e-9)
        return np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
        
    return heuristic
# EVOLVE-BLOCK-END
