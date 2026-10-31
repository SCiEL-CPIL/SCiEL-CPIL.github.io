"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Precompute constants for the Weibull distribution used in the evaluator.
# The task mentions "five generated Weibull sequences of 1000 items with capacity 100".
# Typical Weibull parameters for bin packing benchmarks often involve shape k ~ 1.0-2.0 and scale lambda.
# Without exact parameters, we assume a standard distribution. 
# However, the heuristic should be robust. 
# Let's use a generic "expected next item" estimate.
# If we assume items are drawn from a distribution with mean mu, we can approximate mu.
# For Weibull(k, lambda), mean = lambda * Gamma(1 + 1/k).
# Common benchmark: k=1, lambda=10 -> mean=10. Or k=2, lambda=5 -> mean ~ 6.3.
# Let's assume a mean item size of roughly 10-20 for capacity 100.
# A safer approach: use the current item size as a proxy for the next one, or a fixed constant.
# The directive suggests: "scoring each bin based on the negative of the 'effective capacity' defined as the remaining space minus the expected size of the next item".
# Let's assume expected_next_item = 10.0 (a reasonable guess for capacity 100).
# If the distribution is Weibull(1, 10), mean is 10.
# If it's Weibull(2, 5), mean is ~6.3.
# Let's use a conservative estimate.

EXPECTED_NEXT_ITEM = 10.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: 
    1. Filter bins that can fit the current item.
    2. For valid bins, calculate 'effective capacity' = remaining_space - EXPECTED_NEXT_ITEM.
    3. We want to maximize the probability of accepting a second consecutive item.
       This means we want the remaining space after placing the current item to be >= EXPECTED_NEXT_ITEM.
       So, valid bins for the *next* item are those where (bins - item) >= EXPECTED_NEXT_ITEM.
       Among those, we want to minimize the wasted space? Or maximize the fit?
       
       The directive says: "prioritizing bins that maximize the probability of accepting a second consecutive item".
       If remaining_space_after_current < EXPECTED_NEXT_ITEM, the probability is 0 (or low).
       If remaining_space_after_current >= EXPECTED_NEXT_ITEM, the probability is higher.
       
       A simple score: 
       Let r = bins - item (remaining space after placing item).
       If r < EXPECTED_NEXT_ITEM, score is low (penalty).
       If r >= EXPECTED_NEXT_ITEM, score is high.
       
       To make it a smooth scoring:
       score = -r  (Best Fit: minimize remaining space)
       BUT, if r is too small, we can't fit the next item.
       
       Let's try: 
       score = -(r - EXPECTED_NEXT_ITEM) if r >= EXPECTED_NEXT_ITEM else -r * 1.1 (slight penalty for not fitting next)
       Actually, if we can't fit the next item, that bin is "wasted" for the next step.
       
       Alternative interpretation:
       We want to keep the bin "useful" for the next item.
       So, we prefer bins where r is close to EXPECTED_NEXT_ITEM from above?
       No, if r is exactly EXPECTED_NEXT_ITEM, we fit the next item perfectly.
       If r is much larger, we have more flexibility.
       
       Let's look at the "effective capacity" definition in the directive:
       "negative of the 'effective capacity' defined as the remaining space minus the expected size of the next item"
       effective_cap = r - EXPECTED_NEXT_ITEM
       score = -effective_cap = -(r - EXPECTED_NEXT_ITEM) = EXPECTED_NEXT_ITEM - r
        
       If r > EXPECTED_NEXT_ITEM, score is negative.
       If r < EXPECTED_NEXT_ITEM, score is positive.
       We want the HIGHEST score.
       So, if r < EXPECTED_NEXT_ITEM, we get a positive score (good?).
       Wait, if r < EXPECTED_NEXT_ITEM, we CANNOT fit the next item. That should be bad.
       
       Let's re-read: "prioritizing bins that maximize the probability of accepting a second consecutive item".
       Probability of accepting next item is 1 if r >= expected_next, else 0 (assuming deterministic next size for simplicity, or high if r is large).
       
       If we use score = EXPECTED_NEXT_ITEM - r:
       - If r is small (e.g., 1), score = 9.
       - If r is large (e.g., 50), score = -40.
       This prefers bins with small remaining space, even if they can't fit the next item.
       This contradicts "maximize probability of accepting next item".
       
       Maybe the score should be:
       If r >= EXPECTED_NEXT_ITEM: score = r - EXPECTED_NEXT_ITEM (prefer larger remaining space? No, that's Worst Fit).
       Or score = - (r - EXPECTED_NEXT_ITEM)^2?
       
       Let's try a different angle.
       Standard Best Fit minimizes r.
       Standard First Fit picks the first.
       
       If we assume the next item is likely to be similar to the current one, we want to leave enough room for it.
       So, we should avoid bins where r < item (or r < EXPECTED_NEXT_ITEM).
       Among bins where r >= EXPECTED_NEXT_ITEM, which is better?
       Usually, Best Fit (minimize r) is good for packing density.
       
       Let's try:
       score = -r  (Best Fit)
       BUT, add a penalty if r < EXPECTED_NEXT_ITEM.
       
       Or, simply:
       score = -r
       This is the seed program. It scores 1.0.
       
       Let's try to improve on Best Fit.
       What if we prefer bins where r is close to EXPECTED_NEXT_ITEM?
       score = -abs(r - EXPECTED_NEXT_ITEM)
       This would prefer bins where the remaining space is exactly the expected next item size.
       
       Let's test this hypothesis.
       If r = 10 (expected next), score = 0.
       If r = 5, score = -5.
       If r = 15, score = -5.
       So it prefers r=10 over r=5 or r=15.
       This keeps the bin "ready" for the next item.
       
       However, if the next item is smaller than expected, r=15 is better.
       If the next item is larger, r=15 is worse.
       
       Let's stick to the directive's suggestion:
       "scoring each bin based on the negative of the 'effective capacity' defined as the remaining space minus the expected size of the next item"
       
       effective_cap = r - EXPECTED_NEXT_ITEM
       score = -effective_cap = EXPECTED_NEXT_ITEM - r
        
       This is linear.
       Highest score -> smallest r.
       But it doesn't penalize r < EXPECTED_NEXT_ITEM specifically, it just gives higher scores for smaller r.
       Since we want to maximize the score, and smaller r gives higher score, this is equivalent to Best Fit.
       Wait, if r=1, score=9. If r=10, score=0. If r=20, score=-10.
       It prefers r=1.
       But r=1 cannot fit the next item (size 10).
       So this heuristic is bad if we care about the next item.
       
       Maybe the directive implies we should ONLY consider bins that CAN fit the next item?
       "prioritizing bins that maximize the probability of accepting a second consecutive item"
       
       Let's modify the score to heavily penalize bins that cannot fit the expected next item.
       
       def priority(item, bins):
           r = bins - item
           # Only consider bins that can fit the current item
           valid = r >= 0
           
           # Calculate score
           # If r < EXPECTED_NEXT_ITEM, we can't fit the next item. Penalty.
           # If r >= EXPECTED_NEXT_ITEM, we can fit the next item.
           
           # Score = -r (Best Fit) but with a jump at EXPECTED_NEXT_ITEM?
           
           # Let's try:
           # score = -r
           # But if r < EXPECTED_NEXT_ITEM, score = -r - 1000 (huge penalty)
           
           # This would force the algorithm to only use bins that can fit the next item, unless no such bin exists.
           # If no bin can fit the next item, it falls back to Best Fit.
           
           scores = -r
           # Apply penalty
           penalty_mask = r < EXPECTED_NEXT_ITEM
           scores[penalty_mask] -= 1000.0
           
           # Invalidate bins that can't fit current item
           scores[~valid] = -np.inf
           
           return scores

    This seems like a strong improvement over pure Best Fit.
    It ensures that we don't "waste" a bin by filling it up so much that the next item (which is likely similar) doesn't fit.
    This is a form of "Look-Ahead" Best Fit.
    
    Let's refine the penalty.
    Instead of a hard penalty, maybe a smooth one?
    But the deadline is tight, so simple is better.
    
    Also, we need to handle the case where EXPECTED_NEXT_ITEM is not known.
    We estimated it as 10.0.
    
    Let's also consider that the item sizes might vary.
    If the current item is large, the expected next item might be different?
    No, the distribution is fixed.
    
    Let's implement this.
    
    One more thing: The directive mentions "soft-min/soft-max".
    We are using a hard penalty.
    Maybe we can use a smooth penalty?
    penalty = -max(0, EXPECTED_NEXT_ITEM - r) * factor
    But this is still piecewise linear.
    
    Let's stick to the hard penalty for now, as it's simple and effective for "look-ahead".
    
    Wait, what if EXPECTED_NEXT_ITEM is wrong?
    If it's too high, we might reject bins that could fit the next item.
    If it's too low, we might accept bins that can't.
    
    Given the uncertainty, a moderate penalty is safer.
    
    Let's try:
    scores = -r
    scores[r < EXPECTED_NEXT_ITEM] -= 50.0  # Moderate penalty
    
    This way, if there's a bin with r=9 (can't fit 10) and a bin with r=11 (can fit 10),
    r=9 score: -9 - 50 = -59
    r=11 score: -11
    We prefer r=11.
    
    If there's only r=9 and r=5 (both can't fit 10),
    r=9 score: -59
    r=5 score: -5 - 50 = -55
    We prefer r=5 (Best Fit among the bad ones).
    
    This seems reasonable.

    Let's finalize the code.
    """
    # Calculate remaining space after placing the item
    remaining = bins - item
    
    # Initialize scores with Best Fit (minimize remaining space)
    scores = -remaining
    
    # Apply penalty for bins that cannot fit the expected next item
    # This encourages keeping bins "open" for the next item
    penalty_mask = remaining < EXPECTED_NEXT_ITEM
    scores[penalty_mask] -= 50.0
    
    # Invalidate bins that cannot fit the current item
    invalid_mask = remaining < 0
    scores[invalid_mask] = -np.inf
    
    return scores

# EVOLVE-BLOCK-END
