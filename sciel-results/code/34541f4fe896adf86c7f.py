import numpy as np

def heuristics(prizes, weights):
        # prizes: (5, 100) - wait, task says prizes and weights are (5, 100).
        # "prizes and weights, of shape (5, 100)"
        # This implies 5 constraints, 100 items.
        # prizes is likely the prize for each item? No, "Item scores...".
        # "prizes and weights, of shape (5, 100)".
        # Usually prize is a scalar per item. Why shape (5, 100)?
        # Ah, "prizes ... uniform in [0,1]".
        # If prizes is (5, 100), maybe it's 5 different prize vectors?
        # No, "total prize" is mentioned.
        # Let's re-read carefully: "prizes and weights, of shape (5, 100)".
        # "Every constraint has capacity 25".
        # "prizes ... uniform in [0, 1]".
        # If prizes is (5, 100), does it mean there are 5 prize dimensions?
        # "highest total prize found". This suggests a single total prize.
        # Perhaps `prizes` is (1, 100) or (100,) in standard MCKP, but here it says (5, 100).
        # Maybe the prize is the sum of the 5 rows? Or the mean?
        # "total prize" usually implies $\sum p_i x_i$.
        # If `prizes` is (5, 100), maybe the prize for item $j$ is $\sum_{i=0}^4 prizes[i, j]$?
        # Or maybe the prize is just one of them?
        # Let's assume the "prize" of an item is the sum of the values in the `prizes` array for that column, or the mean.
        # Given "uniform in [0,1]", summing 5 values gives [0,5].
        # Let's check the seed: `return prizes / (weights.sum(axis=0) + 1e-9)`.
        # If `prizes` is (5, 100), then `prizes / weights_sum` results in (5, 100).
        # But the function must return "one attractiveness score per item".
        # So the return shape must be (100,).
        # The seed code `return prizes / ...` would return (5, 100) if `prizes` is (5, 100).
        # This would violate "wrong shape ... makes the whole result invalid".
        # UNLESS `prizes` is actually (100,) and the description "shape (5, 100)" applies to `weights` only?
        # "prizes and weights, of shape (5, 100)". Grammatically, both.
        # But if the seed returns (5, 100) and scores 1.0, then the evaluator must accept (5, 100) or the seed is wrong?
        # "The heuristic is computed once... An exception, non-finite values or a wrong shape make the whole result invalid."
        # "Interface: heuristics(prizes, weights) -> array of one score per item."
        # This strongly implies the return must be (100,).
        # If the seed `return prizes / ...` returns (5, 100), it should be invalid.
        # But it scores 1.0.
        # Hypothesis 1: `prizes` is actually (100,) in the actual execution, despite the description.
        # Hypothesis 2: The evaluator sums the rows of the returned array? Unlikely.
        # Hypothesis 3: The description "shape (5, 100)" is for `weights`, and `prizes` is (100,).
        # Let's look at `weights.sum(axis=0)`. If weights is (5, 100), sum is (100,).
        # If `prizes` is (100,), `prizes / weights.sum(axis=0)` is (100,). This works.
        # If `prizes` is (5, 100), `prizes / weights.sum(axis=0)` broadcasts to (5, 100).
        # Given the seed works, I will assume `prizes` is effectively a 1D array of length 100, or I need to reduce it to 1D.
        # To be safe, I will handle both. If `prizes` is 2D, I'll sum or mean it to get a scalar per item.
        # "total prize" suggests summing the prize components if there are multiple.
        # Let's assume `item_prize = prizes.sum(axis=0)` if `prizes.ndim == 2`, else `prizes`.
        
        # Let's refine the heuristic.
        # We established `sum` weights was baseline 1.0.
        # `max` weights was worse.
        # Let's try `sqrt(sum(weights^2))` (L2 norm) or `sum(weights)`.
        # Let's try normalizing the heuristic to have mean 1.
        
        item_prizes = prizes.sum(axis=0) if prizes.ndim > 1 else prizes
        weight_sums = weights.sum(axis=0)
        
        # Avoid division by zero
        weight_sums_safe = weight_sums + 1e-9
        
        # Ratio
        ratios = item_prizes / weight_sums_safe
        
        # Normalization: Scale so that the mean is 1.
        # This helps balance the influence of pheromone (starts at 1) and heuristic.
        mean_ratio = np.mean(ratios)
        if mean_ratio > 0:
            normalized_ratios = ratios / mean_ratio
        else:
            normalized_ratios = ratios
            
        # Optional: Power to sharpen
        # Let's try power 1.5 or 2.
        # Sharpness helps find the best items quickly, but might reduce diversity.
        # With 30 iterations, diversity is maintained by pheromone decay.
        # Let's try power 1.0 first (just normalization) to see if scaling matters.
        # Actually, the seed didn't normalize.
        # If I normalize, the magnitudes change.
        # Let's try: eta = (ratios / max(ratios)) ** 2 ?
        # Or just normalized ratios.
        
        # Let's go with Normalized Ratios.
        return normalized_ratios
