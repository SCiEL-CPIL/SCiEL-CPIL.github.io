"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Weibull parameters for the generated sequences (assumed standard for this task)
# The mode of a Weibull distribution is ( (k-1)/lambda )^(1/k) for k > 1
# Common Weibull parameters in bin packing benchmarks are shape k=1.5 or 2.0, scale lambda=1.0
# Let's assume k=1.5, lambda=1.0 for the "mode" target.
# Mode = ((1.5-1)/1.0)^(1/1.5) = (0.5)^(2/3) ≈ 0.63
# If k=2.0, lambda=1.0, Mode = ((2-1)/1.0)^(1/2) = 1.0
# Given capacity 100, items are likely scaled. 
# The prompt mentions "five generated Weibull sequences". 
# Typical online bin packing benchmarks (e.g. from FunSearch papers) often use Weibull(1.5, 1.0) or similar, 
# but scaled to fit capacity. 
# However, we don't know the exact scale. 
# A robust heuristic is to target the *median* or *mode* of the item distribution.
# If we assume the items are drawn from Weibull(k, lambda) and scaled to have mean ~10 or so?
# Actually, the prompt says "capacity 100". 
# Let's look at the "Inspiration 3" which used _weibull_cdf. 
# The strategy directive specifically asks for "squared deviation between the resulting remainder and the mode of the Weibull item size distribution".
# We need to estimate the mode. Without knowing the exact parameters, we can assume a standard distribution.
# Common in these tasks: Weibull(1.5, 1.0) scaled? Or just raw?
# If items are small (e.g., 1-10), mode is small. If items are large (e.g., 20-80), mode is larger.
# Let's assume the items are distributed such that their mode is around 1/3 to 1/2 of the capacity?
# No, let's look at the "cell findings": "probabilistic gap penalty that avoids creating fragments smaller than the expected next Weibull item".
# The directive is specific: "steering remainders toward the most probable future item size".
# Let's assume a Weibull distribution with shape k=1.5 and scale lambda=1.0, but scaled to the problem.
# If the mean item size is not given, we have to guess. 
# However, often in these "FunSearch" tasks, the items are generated from a Weibull(1.5, 1.0) distribution multiplied by a constant, 
# or simply Weibull(1.5, 1.0) with capacity adjusted. 
# Wait, capacity is 100. If items are ~10, then mode ~6.3. 
# Let's assume the items are from Weibull(1.5, 1.0) scaled by 10? Or just Weibull(1.5, 1.0) with capacity 10? 
# The prompt says "capacity 100". 
# Let's assume the item sizes are drawn from a Weibull distribution with shape 1.5 and scale 1.0, 
# but perhaps scaled to have a mean of ~10? 
# Actually, a safer bet is to use a target remainder that is a fraction of the capacity, 
# but the directive says "mode of the Weibull item size distribution".
# Let's assume the distribution is Weibull(k=1.5, lambda=1.0) and the items are scaled by 10. 
# Then Mode = 0.63 * 10 = 6.3.
# Or if items are Weibull(1.5, 1.0) and capacity is 10? No, capacity is 100.
# Let's try a generic target. If we don't know the scale, we can't compute the exact mode.
# However, we can infer from the "Inspiration 3" that it used _weibull_cdf, implying it might have tried to fit or use a known distribution.
# Let's assume the standard Weibull(1.5, 1.0) distribution for the *shape* of the sizes, 
# and assume a scale factor such that the mean is around 10-20.
# Let's pick a target remainder of 10.0 as a proxy for the mode (assuming items ~10).
# Or better, let's use the "next item" expectation. 
# The directive says "mode of the Weibull item size distribution".
# I will define a constant MODE_ITEM_SIZE. 
# If I assume Weibull(1.5, 1.0) scaled by 10, mode is ~6.3.
# If I assume Weibull(2.0, 1.0) scaled by 10, mode is 10.0.
# Let's use 10.0 as a reasonable estimate for the mode of items in a capacity 100 problem.

MODE_ITEM_SIZE = 10.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score bins by how close the remaining capacity is to the mode of the item distribution.
    This steers the packing towards states where the remaining space is most likely to be filled by a future item.
    """
    n = len(bins)
    scores = np.full(n, -1e18)
    
    # Calculate remaining capacity after placement
    remaining = bins - item
    
    # Only consider bins where remaining >= 0
    valid_mask = remaining >= 0
    if not np.any(valid_mask):
        return scores
    
    valid_indices = np.where(valid_mask)[0]
    valid_remaining = remaining[valid_mask]
    
    # The score is the negative squared deviation from the target mode.
    # We want to minimize (remaining - MODE_ITEM_SIZE)^2
    # So we score by - (remaining - MODE_ITEM_SIZE)^2
    
    # To avoid overwhelming the "best fit" aspect if MODE is far off, 
    # we can combine with a small best-fit term, but the directive says 
    # "score for a bin is the negative of the squared deviation".
    # I will stick to the directive strictly.
    
    deviation = valid_remaining - MODE_ITEM_SIZE
    scores[valid_indices] = -deviation**2
    
    return scores
# EVOLVE-BLOCK-END
