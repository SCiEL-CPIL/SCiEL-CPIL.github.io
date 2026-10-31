"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 120.0  # Safety margin before the 150s kill switch

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Higher score is preferred.
    
    Strategy: Thermodynamic Equilibrium Heuristic
    - Models the bin packing process as a thermodynamic system.
    - The goal is to minimize the 'free energy' of the gap distribution.
    - We assume an optimal equilibrium state for the remaining capacity.
    - For Weibull-distributed items (shape k~1-2, scale lambda~20-30), the 
      optimal 'residual' space to keep in a bin after placing an item is 
      not zero (which causes fragmentation) nor large (which wastes space).
    - Instead, we target a specific 'equilibrium gap' G* that balances 
      fragmentation risk against wasted capacity.
    - The score is the negative of the Kullback-Leibler-like divergence 
      (squared error) between the resulting remainder and G*.
    
    Key structural change:
    - Current: Purely local, non-linear penalty on remainder (quadratic/linear).
    - New: Targets a specific non-zero remainder value (G*) derived from 
      statistical properties of the item distribution.
    """
    # Deadline guard
    if time.time() - _start_time > _DEADLINE:
        # Return a valid but arbitrary score to avoid timeout
        n = len(bins)
        scores = np.full(n, -1e18)
        valid_mask = bins >= item
        if np.any(valid_mask):
            scores[valid_mask] = 0.0
        return scores

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
    
    # --- Thermodynamic Model ---
    # Estimate the 'equilibrium gap' G*.
    # For Weibull items with capacity 100, the mean item size is roughly 20-30.
    # If we pack too tightly (remainder ~0), we create many tiny gaps that 
    # cannot accommodate future items, leading to wasted bins.
    # If we pack loosely (remainder ~50), we waste space.
    # A good heuristic for G* is related to the expected size of the *next* item 
    # that might fit, or a fraction of the capacity.
    # Let's estimate G* based on the item size itself and the bin's current state.
    
    # A simple model: The optimal remainder is proportional to the item size,
    # but capped. Smaller items should leave more relative space? 
    # Actually, for small items, we want to pack them tightly to save space for 
    # larger items. For large items, we want to avoid leaving large gaps.
    
    # Let's use a dynamic target:
    # G* = alpha * item + beta
    # If item is small, G* is small (pack tight).
    # If item is large, G* is larger (don't overpack).
    
    # However, a more robust thermodynamic approach uses the entropy of the 
    # gap distribution. Minimizing the variance of the gaps is often a good 
    # proxy. But we are choosing ONE bin.
    
    # Let's try targeting a specific remainder that is optimal for the 
    # Weibull distribution.
    # The mode of a Weibull(k, lambda) is lambda * ((k-1)/k)^(1/k).
    # Assuming k=2, lambda=25 (mean ~20), mode ~25 * (0.5)^(0.5) ~ 17.6.
    # So, leaving a gap of ~15-20 might be optimal.
    
    # Let's define G* as a function of the item size to adapt dynamically.
    # If item is very small, we want to pack it into a nearly full bin 
    # (G* close to 0).
    # If item is large, we want to put it in a bin where the remainder is 
    # moderate (G* ~ 10-20).
    
    # G* = max(0, 20 - 0.5 * item)
    # If item=40, G*=0.
    # If item=10, G*=15.
    # If item=5, G*=17.5.
    
    # This encourages:
    # - Large items: Best Fit (tightest possible).
    # - Small items: Worst Fit (leave space for future small items? No, 
    #   actually, for small items, we want to consolidate them to keep 
    #   bins available for large items. So Best Fit for small items too?)
    
    # Let's reconsider.
    # Best Fit Decreasing (BFD) is strong.
    # But online, we don't know future sizes.
    # If we place a small item in a nearly full bin, we use up that bin.
    # If we place a small item in a nearly empty bin, we fragment the empty space.
    
    # The "Thermodynamic" idea suggests that there is an optimal distribution 
    # of gaps. If we have many bins with gap ~G*, the system is in equilibrium.
    # So, we should prefer bins that result in a remainder close to G*.
    
    # Let's set G* to be a constant value that represents the "average" 
    # useful gap.
    # For capacity 100, a gap of 15-20 seems reasonable.
    
    G_star = 18.0
    
    # Score is negative squared error from G_star
    # score = -(remaining - G_star)^2
    
    # However, if remaining is much larger than G_star, we are wasting space.
    # If remaining is much smaller than G_star, we are fragmenting.
    # The squared error penalizes both.
    
    # But we must also ensure that we don't waste a bin if the remainder is 
    # very small (e.g., 1). In that case, the bin is effectively full.
    # Placing an item that leaves 1 is better than placing it that leaves 18?
    # No, because the bin with 1 remaining is "dead" for future large items.
    # The bin with 18 remaining can still take a small item.
    
    # So, targeting G* is a good heuristic to keep bins "alive".
    
    error = (valid_remaining - G_star) ** 2
    
    # To avoid floating point issues and ensure distinct scores, 
    # we can add a small penalty for very large remainders (waste) 
    # and very small remainders (fragmentation).
    # The squared error already does this.
    
    scores[valid_indices] = -error
    
    return scores
# EVOLVE-BLOCK-END
