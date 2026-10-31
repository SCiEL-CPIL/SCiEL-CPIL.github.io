"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Polarization-aware bin scoring: prefer bins that will become full or nearly full,
    reducing the number of 'half-full' bins that are hard to pack efficiently.
    
    Score components:
    1. Best-fit base: minimize leftover space
    2. Polarization bonus: reward bins that will cross a 'fullness threshold'
    3. Fragmentation penalty: penalize creating new half-full states
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
    valid_bins_before = bins[valid_mask]
    
    # Base best-fit score: prefer smaller remaining
    base_score = -valid_remaining
    
    # Polarization: bins that are already fairly full and will become even fuller
    # are good. Bins that are empty and will become half-full are bad.
    # Define "fullness" as (capacity - remaining) / capacity, but we don't know capacity.
    # Instead, use relative fill: item / (bins_before) shows how much of this bin we're using.
    # A bin where item fills a large fraction is being "completed".
    
    # Fraction of bin filled by this item
    fill_fraction = item / valid_bins_before  # in (0, 1]
    
    # Polarization score: reward bins where the item completes a bin (high fill_fraction)
    # and penalize bins where the item creates a new partial bin (low fill_fraction on empty bin)
    # Use a smooth step-like function: sigmoid around 0.5 fill fraction
    # Bins with fill_fraction > 0.5 get bonus, < 0.5 get penalty
    
    # Smooth polarization: tanh-based
    polar_weight = 2.0
    polarization = polar_weight * np.tanh(4.0 * (fill_fraction - 0.5))
    
    # Fragmentation penalty: penalize bins that will end up in the "danger zone" 
    # (remaining between 0.2*capacity and 0.8*capacity roughly)
    # Since we don't know capacity, use remaining/item ratio as proxy
    # If remaining is comparable to item (0.5*item < remaining < 2*item), it's fragmented
    # Use a smooth bump function centered at remaining = item
    
    frag_center = item
    frag_width = 0.5 * item + 1e-10
    frag_bump = np.exp(-((valid_remaining - frag_center) / frag_width) ** 2)
    frag_penalty = 1.5 * frag_bump
    
    # Combine: base + polarization - fragmentation
    scores[valid_indices] = base_score + polarization - frag_penalty
    
    return scores
# EVOLVE-BLOCK-END
