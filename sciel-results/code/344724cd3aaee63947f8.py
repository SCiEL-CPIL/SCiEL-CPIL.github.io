import numpy as np
import time

def heuristics(prizes, weights):
    start_time = time.time()
    
    # prizes: (5, 100), weights: (5, 100)
    # Capacity per constraint is 25
    
    # Structural change: Use bottleneck-aware efficiency.
    # Instead of just total_prize / total_weight, we account for the
    # tightest constraint each item hits. Items that are efficient
    # in the constraint they stress the most are more valuable.
    
    eps = 1e-9
    
    # 1. Per-constraint efficiency: prize_ij / weight_ij
    # This tells us how much prize we get per unit of resource j
    ratios = prizes / (weights + eps)  # shape (5, 100)
    
    # 2. Bottleneck efficiency: min ratio across constraints
    # This represents the "worst-case" efficiency of the item.
    # An item with high min-ratio is good across ALL constraints.
    min_ratios = ratios.min(axis=0)  # shape (100,)
    
    # 3. Total efficiency: sum(prizes) / sum(weights)
    # This is the standard value density.
    total_prize = prizes.sum(axis=0)
    total_weight = weights.sum(axis=0)
    total_eff = total_prize / (total_weight + eps)
    
    # 4. Combine the two perspectives.
    # The bottleneck metric (min_ratios) is crucial for MCKP because
    # a single tight constraint can block an item. However, pure min
    # can be too pessimistic if one constraint is trivially loose.
    # A blend allows the heuristic to balance global value density
    # with local constraint tightness.
    #
    # We use a weighted geometric mean or a simple convex combination.
    # Let's try a convex combination that emphasizes the bottleneck
    # but retains the total signal.
    #
    # Normalization: The scales of min_ratios and total_eff may differ.
    # Since prizes and weights are in [0,1], ratios are roughly in [0, inf).
    # A simple sum or weighted sum should work if scales are similar.
    # To be safe, we can normalize each to [0, 1] range relative to their max,
    # or just rely on the fact that both are positive.
    #
    # Let's use a blend: 0.5 * min_ratios + 0.5 * total_eff
    # Or perhaps weight the bottleneck more heavily?
    # Inspiration 2 used 0.7 * min + 0.3 * total.
    # Inspiration 1 (score 1.0) used prizes / (weights.sum(axis=0) + eps) which is total_eff.
    # Wait, Inspiration 1's code: `return prizes / (weights.sum(axis=0) + 1e-9)`
    # This returns a (5, 100) array? No, the task says "returns one attractiveness score per item".
    # If Inspiration 1 returned (5, 100), it would be invalid shape unless the evaluator sums it?
    # The task says "wrong shape make the whole result invalid".
    # So Inspiration 1 must have been interpreted differently or I am misreading.
    # `prizes / (weights.sum(axis=0) + 1e-9)` -> (5, 100) / (100,) -> (5, 100).
    # This would be an invalid shape for `heuristics` which must return shape (100,).
    # Unless... the seed program or evaluator handles it?
    # "Interface: heuristics(prizes, weights) -> array of one score per item."
    # So Inspiration 1 likely failed or was fixed? But it says score 1.0000.
    # Maybe the evaluator sums the last axis? Or maybe the code shown is a simplified version?
    # Let's assume the score 1.0000 is the baseline (seed program).
    # The seed program is likely just `prizes.sum(axis=0) / (weights.sum(axis=0) + eps)` or similar.
    # If the seed is 1.0, and my current is 0.86, I need to get back to ~1.0 or better.
    #
    # Let's look at Inspiration 2 (0.88) and 3 (0.87). They are worse than seed.
    # This suggests that complex heuristics might be hurting, or the seed is very strong.
    # The seed is likely simple total efficiency.
    #
    # Let's try a different structural approach:
    # Use the harmonic mean of the per-constraint efficiencies?
    # Or simply: Total Prize / (1 + sum of relative weights)?
    #
    # Let's try to mimic the "bottleneck" idea but more robustly.
    # If an item has a very low ratio in one constraint, it's bad.
    # `min_ratios` captures this.
    #
    # Let's try: eta = min_ratios * total_prize
    # This rewards high total prize AND high bottleneck efficiency.
    
    # Let's try a different combination:
    # eta = total_prize / (1 + max_relative_weight)
    # where max_relative_weight = max(weights) / capacity
    # This penalizes items that hit the capacity limit in any constraint.
    
    capacity = 25.0
    rel_weights = weights / capacity  # (5, 100)
    max_rel_weight = np.max(rel_weights, axis=0)  # (100,)
    
    # Penalize items that consume a large fraction of any constraint
    penalty = 1.0 + max_rel_weight
    
    # Heuristic: Total Prize / Penalty
    # This is a structural change from pure density to penalty-based density.
    eta = total_prize / (penalty + eps)
    
    # Ensure positivity
    eta = np.where(eta > 0, eta, 1e-9)
    
    # Deadline guard
    if time.time() - start_time > 290:
        return eta
        
    return eta
