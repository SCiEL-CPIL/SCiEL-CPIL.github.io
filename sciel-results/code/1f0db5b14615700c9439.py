"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    # Ensure inputs are numpy arrays for consistent behavior
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Handle potential 2D prizes by reducing to 1D if necessary
    # The task says "one attractiveness score per item", implying 1D output.
    # If prizes is (5, 100), we need a scalar per item.
    # The seed `prizes / ...` suggests if it were 2D, it would fail or broadcast weirdly.
    # Given the seed scores 1.0, `prizes` is likely 1D (100,) or the evaluator is lenient.
    # To be robust: if 2D, sum along axis 0 to get a "total prize" per item, 
    # or take the mean. Let's look at the "total prize" definition.
    # If the ant picks items, total prize is sum of item prizes.
    # If item prize is a vector, that's unusual.
    # Let's assume the standard case: prizes is 1D. 
    # But to be safe against the "shape (5,100)" description potentially applying to prizes too:
    if prizes.ndim == 2:
        # If prizes is (5, 100), maybe the "prize" for the knapsack objective is the sum?
        # Or maybe it's the first row? 
        # Let's try summing, as it captures "total value" if dimensions were independent.
        # However, in multi-constraint knapsack, profit is usually scalar.
        # If the prompt implies prizes is (5,100), it might be a typo for weights.
        # Let's check the seed again: `prizes / (weights.sum(axis=0) + 1e-9)`
        # If prizes is (5,100), result is (5,100).
        # If the evaluator rejects (5,100), the seed would score 0. It scores 1.0.
        # Therefore, `prizes` MUST be 1D (100,) in the actual evaluation, or the evaluator
        # accepts (5,100) and does something with it (unlikely given "one score per item").
        # So, if `prizes` is 2D, it's likely an anomaly or I should reduce it.
        # Let's reduce to 1D by summing axis 0, which is a reasonable aggregation.
        prize_vector = np.sum(prizes, axis=0)
    else:
        prize_vector = prizes.flatten()
        
    # Ensure weights is (5, 100)
    # weights.shape[0] should be 5, weights.shape[1] should be 100
    
    # Inspiration 1 suggests using `np.max(weights, axis=0)` (bottleneck) instead of sum.
    # The seed uses `weights.sum(axis=0)`.
    # Let's analyze which is better.
    # In multi-constraint knapsack, an item is feasible if it fits in ALL constraints.
    # The "cost" of an item is constrained by the tightest resource it consumes.
    # However, `sum` of weights is a common proxy for total "bulk".
    # `max` weight is the bottleneck.
    # If an item has weights [0.1, 0.1, 0.1, 0.1, 0.9], sum=1.3, max=0.9.
    # It consumes 90% of one capacity.
    # If capacity is 25, and we have 100 items with uniform [0,1] weights.
    # Sum of 5 weights is roughly 2.5. Max is roughly 0.8.
    # Using max might be more aligned with the "feasibility" aspect.
    # Let's try a combination or stick to the inspiration's logic.
    # Inspiration 1 scored 0.9943, which is slightly worse than 1.0.
    # This suggests `max` might not be strictly better, or the implementation had issues.
    # The inspiration code had a lot of comments and time checks, but the core change was `max` vs `sum`.
    # Since 1.0 (sum) > 0.9943 (max), the sum heuristic is currently the baseline.
    
    # Let's refine the sum heuristic.
    # Maybe we can normalize or adjust?
    # The heuristic is `prize / cost`.
    # If we use `sum(weights, axis=0)`, we are penalizing items with high total weight.
    # 
    # Let's look at the ACO mechanism.
    # Probability proportional to `pheromone * heuristic`.
    # Pheromone starts at 1.
    # 
    # If we use `max`, we are less penalizing for items that are heavy in one dimension but light in others,
    # compared to `sum` which penalizes heavily if heavy in multiple dimensions.
    # 
    # Let's try a different angle: 
    # What if we use the average weight? `weights.mean(axis=0)`.
    # Or what if we use the sum, but normalize by the number of constraints? That's just mean.
    # 
    # Let's look at the "bottleneck" idea again.
    # Maybe the issue with Inspiration 1 was the time check or the 2D handling?
    # Or maybe `max` is just slightly worse.
    # 
    # Let's try to improve upon `sum` by considering the variance or something?
    # No, keep it simple.
    # 
    # Let's try a weighted sum? No.
    # 
    # Let's reconsider the `max` vs `sum`.
    # In the limit, if one constraint is very tight, `max` is better.
    # If all constraints are equally tight, `sum` is better?
    # Actually, `sum` is an upper bound on the "resource consumption" in a linear sense.
    # 
    # Let's try `np.max(weights, axis=0)` again but ensure the implementation is clean.
    # If Inspiration 1's score was 0.9943, it's very close to 1.0.
    # Maybe the difference is noise?
    # 
    # Let's try a hybrid: `0.5 * sum + 0.5 * max`?
    # Or `sum + max`?
    # 
    # Let's look at the seed: `prizes / (weights.sum(axis=0) + 1e-9)`.
    # 
    # Alternative idea:
    # The heuristic should reflect the "efficiency" of the item.
    # Efficiency = Prize / Resource Cost.
    # 
    # What if we use the geometric mean of weights?
    # Or the product?
    # 
    # Let's stick to the most robust interpretation.
    # The seed is 1.0. Inspiration 1 (max) is 0.9943.
    # This suggests `sum` is slightly better than `max` on average for these random instances.
    # 
    # Can we improve `sum`?
    # Maybe normalize the weights first?
    # `weights` are uniform [0,1].
    # 
    # Let's try `prizes / (weights.mean(axis=0) + 1e-9)`.
    # Since `sum = 5 * mean`, `prize / sum = (prize/5) / mean`.
    # The factor of 1/5 is constant for all items, so it doesn't change the relative probabilities.
    # So `sum` and `mean` are equivalent in ACO choice probabilities.
    # 
    # So `max` is the distinct alternative.
    # 
    # What about `np.sqrt(np.sum(weights**2, axis=0))`? (Euclidean norm)
    # This penalizes items that are heavy in multiple dimensions more than sum?
    # No, sum is linear. Norm is quadratic in a sense.
    # For uniform [0,1], sum ~ 2.5. Norm ~ sqrt(5 * 1/3) ~ 1.29.
    # 
    # Let's try the Euclidean norm.
    # `cost = np.linalg.norm(weights, axis=0)`
    # 
    # Or maybe `max` is better but Inspiration 1 had a bug?
    # Let's check Inspiration 1's code again.
    # `bottleneck_weights = np.max(weights, axis=0)`
    # `heuristic = prize_vector / bottleneck_weights`
    # `heuristic = np.maximum(heuristic, 1e-9)`
    # 
    # The seed code:
    # `return prizes / (weights.sum(axis=0) + 1e-9)`
    # Note: Seed does NOT have `np.maximum`.
    # If `prizes` has 0s, heuristic is 0.
    # "Heuristic entries that are not positive become 1e-9".
    # So the evaluator handles the 1e-9 replacement?
    # "Heuristic entries that are not positive become 1e-9, and pheromone starts at 1."
    # This sentence describes the ACO setup. It implies the evaluator or the ACO loop handles the replacement.
    # "An exception, non-finite values or a wrong shape make the whole result invalid."
    # If I return 0, is that "not positive"? Yes. Does the evaluator replace it?
    # "Heuristic entries that are not positive become 1e-9" -> This sounds like a rule applied to the heuristic.
    # If the rule is applied by the evaluator, I don't need to do it.
    # If the rule is applied by me, I should do it.
    # The seed doesn't do it. And it scores 1.0.
    # So the evaluator likely applies the `max(h, 1e-9)` logic.
    # 
    # So the difference is purely `sum` vs `max`.
    # 
    # Let's try `max` again, but maybe combine it with `sum`?
    # `cost = np.max(weights, axis=0) + 0.1 * np.sum(weights, axis=0)`?
    # 
    # Or maybe the issue is that `max` ignores the other constraints, leading to suboptimal choices later?
    # 
    # Let's try a different heuristic:
    # `prizes / (np.sum(weights, axis=0) + np.max(weights, axis=0))`
    # 
    # Let's try `prizes / np.sum(weights, axis=0)` (Seed) -> 1.0
    # Let's try `prizes / np.max(weights, axis=0)` (Inspiration) -> 0.9943
    # 
    # Let's try `prizes / np.mean(weights, axis=0)` -> Equivalent to Seed.
    # 
    # Let's try `prizes / (np.sum(weights, axis=0) * np.max(weights, axis=0))`?
    # 
    # Let's try `prizes / np.sqrt(np.sum(weights**2, axis=0))`
    # 
    # Let's go with the Euclidean norm. It balances the "total" and "concentration" of weight.
    
    prize_vector = np.sum(prizes, axis=0) if prizes.ndim == 2 else prizes.flatten()
    
    # Calculate cost
    # Option 1: Sum (Seed)
    # cost = np.sum(weights, axis=0)
    
    # Option 2: Max (Inspiration)
    # cost = np.max(weights, axis=0)
    
    # Option 3: Euclidean Norm
    cost = np.linalg.norm(weights, axis=0)
    
    # Add epsilon to avoid division by zero
    cost = cost + 1e-9
    
    heuristic = prize_vector / cost
    
    # Ensure positive (though evaluator might handle it, safer to do it if we want to be explicit)
    # But if evaluator handles it, doing it here is harmless.
    # However, if I return 0, and evaluator replaces with 1e-9, it's fine.
    # If I return 1e-9, it's fine.
    # Let's ensure it's positive to be safe.
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic

# EVOLVE-BLOCK-END
