"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multiple-constraint knapsack.
    
    Strategy: Greedy-Based Regret Score.
    
    1. Problem Framing:
       The current strategy uses Lagrangian duality, which provides a global price
       for resources but may miss local combinatorial blocking effects. We propose
       a discrete combinatorial signal: "regret".
       
       The regret of an item is defined as the loss in total prize of a baseline
       greedy solution when that item is forced to be excluded.
       
       Intuition:
       - If an item is part of the baseline greedy solution, excluding it forces the
         greedy algorithm to pick the next best items. The difference in total prize
         approximates the item's marginal contribution to the "greedy-optimal" set.
       - If an item is NOT in the baseline solution, excluding it might not change the
         solution (regret 0) or might allow a better item to be picked (regret negative,
         which we clip to 0 or a small epsilon). This identifies items that are
         "blocking" better items in the greedy order.
       
       By using regret, we guide the ACO ants towards items that are critical to the
       high-value core of the knapsack and away from items that are either redundant
       or actively harmful to the greedy packing process.

    2. Method:
       a. Compute a baseline greedy solution. We use a "value density" ratio
          prize / sum(weights) as the primary sorting key, with a tie-breaker on prize.
          Actually, for multiple constraints, a simple sum of weights might be misleading.
          A better greedy key is prize / max_c(weight_ic) or prize / (sum_c weight_ic / n_c).
          Let's use a robust greedy: sort by prize / (1 + sum(weights_i)) or similar.
          To be safe and fast, we will compute a "greedy score" for sorting: 
          g_i = prize_i / (1 + np.sum(weights[i]))
          This penalizes items that consume a lot of total capacity.
          
       b. Run the greedy algorithm to get the baseline set S_base and total prize P_base.
       
       c. For each item j, compute the regret:
          - Create a modified set of items excluding j.
          - Run the greedy algorithm on this modified set.
          - Get the total prize P_j.
          - Regret_j = P_base - P_j.
          
       d. The heuristic score is the regret.
          - High regret means the item is essential to the baseline solution.
          - Low/Zero regret means the item is not essential (or was not in the baseline).
          
       e. To handle items not in the baseline: 
          If an item is not in S_base, P_base - P_j might be 0 (if greedy picks same items) 
          or negative (if greedy picks different, better items without j).
          We clip regret to be non-negative: max(0, P_base - P_j).
          
       f. Normalize and sharpen the scores.

    3. Complexity:
       - Baseline greedy: O(N log N) for sorting + O(N * C) for feasibility checks.
       - Regret computation: For each of N items, run greedy O(N log N).
       - Total: O(N^2 log N). With N=100, this is 10,000 * 10 = 100,000 operations per greedy run?
         Wait, N=100. Sorting 100 items is fast. 100 runs * 100 log 100 ~ 70,000 comparisons.
         Feasibility check is O(C) per item added. Max 100 items added. So 100 * 100 * 5 = 50,000 ops per greedy.
         Total 100 * 50,000 = 5,000,000 ops. This is well within 300s in Python/NumPy.
         We can optimize by pre-sorting and just skipping the excluded item in the loop.
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
    time_limit = 20.0  # Should be much faster, but be safe
    
    # Pre-compute greedy sorting keys
    # Key: prize / (1 + sum of weights). This is a proxy for density.
    total_weights = np.sum(weights, axis=0)
    greedy_keys = prizes / (1.0 + total_weights)
    
    # Sort indices by greedy key descending
    sorted_indices = np.argsort(-greedy_keys)
    
    def run_greedy(exclude_idx):
        """
        Run greedy algorithm excluding a specific item index.
        Returns total prize of the selected items.
        """
        used_weights = np.zeros(n_constraints)
        total_prize = 0.0
        
        for idx in sorted_indices:
            if idx == exclude_idx:
                continue
            
            # Check if item fits
            # weights[:, idx] has shape (C,)
            if np.all(used_weights + weights[:, idx] <= capacity):
                used_weights += weights[:, idx]
                total_prize += prizes[idx]
        
        return total_prize

    # 1. Compute baseline greedy solution (no exclusion)
    baseline_prize = run_greedy(-1)  # -1 is never a valid index
    
    # 2. Compute regret for each item
    regrets = np.zeros(n_items)
    
    for j in range(n_items):
        if time.time() - start_time > time_limit:
            # If we run out of time, fill remaining with 0 or average
            # But 100 items * fast greedy should be < 1s
            break
        
        prize_without_j = run_greedy(j)
        regret = baseline_prize - prize_without_j
        # Clip to non-negative
        if regret < 0:
            regret = 0.0
        regrets[j] = regret
    
    # 3. Construct heuristic scores
    # The regret is a discrete signal. We want to map it to a continuous score.
    # Items with high regret are "essential".
    # Items with zero regret are "optional" or "blocking".
    
    # Normalize regrets to [0, 1]
    max_regret = np.max(regrets)
    if max_regret > 0:
        normalized_regrets = regrets / max_regret
    else:
        # If all regrets are 0, fall back to a simple density-based heuristic
        normalized_regrets = greedy_keys / np.max(greedy_keys) if np.max(greedy_keys) > 0 else np.ones(n_items)
    
    # To help ACO, we might want to combine this with the original prize or density
    # to avoid zero scores for items that are good but not in the specific greedy path.
    # However, the directive says to use the regret value as the heuristic.
    # Let's stick to pure regret but ensure it's positive.
    
    # Apply a power to sharpen the distribution?
    # Regrets are often sparse (many zeros). 
    # If we square it, zeros stay zero.
    # Let's add a small base to ensure all items have some chance, 
    # but weight the high regret items much more heavily.
    
    # H_i = 1e-9 + normalized_regret^p
    # If p=1, it's linear.
    # If p=2, it emphasizes the top items more.
    # Given ACO uses P * H, and P starts at 1, a higher H for key items is good.
    
    scores = 1e-9 + (normalized_regrets ** 2.0)
    
    # Ensure positive and finite
    scores = np.where(scores > 0, scores, 1e-9)
    scores = np.where(np.isfinite(scores), scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END
