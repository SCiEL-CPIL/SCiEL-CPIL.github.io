"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(demands, capacity):
    start_time = time.time()
    deadline = 320.0  # Safety margin before 330s kill
    
    n = len(demands)
    demands = np.array(demands, dtype=np.float64)
    
    # Calculate pair sums
    pair_sums = demands[:, None] + demands[None, :]
    
    # Identify valid pairs
    valid = pair_sums <= capacity
    
    # Calculate gaps
    gaps = capacity - pair_sums
    
    # Find the minimum item size
    min_d = np.min(demands)
    
    # Initialize heuristic matrix with a very small positive value for invalid pairs
    eta = np.full((n, n), 1e-9)
    
    # For valid pairs, compute a smooth attractiveness score
    # Strategy: reward tight fits, penalize wasteful gaps
    # A gap is "wasteful" if it's less than min_d (can't fit any item)
    # A gap is "useful" if it can fit at least one item
    
    # Smooth scoring function:
    # - Perfect fit (gap=0): highest score
    # - Small useful gaps: high score (tight packing)
    # - Large gaps: lower score (wastes space)
    # - Dead gaps (0 < gap < min_d): low but positive score
    
    # Base score: inversely related to gap, with a floor
    # Use a smooth function that peaks at gap=0 and decays
    
    # For valid pairs, compute score based on gap
    valid_gaps = gaps[valid]
    
    # Smooth decay: score = base / (1 + gap * decay_rate)
    # But we want to distinguish dead gaps from useful gaps
    # Let's use a piecewise smooth function
    
    # Component 1: Tightness reward (higher for smaller gaps)
    tightness = 1.0 / (1.0 + valid_gaps * 0.1)
    
    # Component 2: Usability penalty (penalize gaps that can't fit any item)
    # This is a smooth approximation of the step function at min_d
    # Use a sigmoid-like transition around min_d
    usability = 1.0 / (1.0 + np.exp(-5.0 * (valid_gaps - min_d) / min_d))
    
    # Combine: score = tightness * usability
    # Perfect fits: tightness=1, usability~0.5 (gap=0 < min_d) -> 0.5
    # Small useful gaps (gap~min_d): tightness~0.9, usability~0.5 -> 0.45
    # Large gaps: tightness~0.1, usability~1.0 -> 0.1
    # Dead gaps: tightness~0.8, usability~0.1 -> 0.08
    
    # Hmm, perfect fits should be highest. Let me adjust.
    # Use: score = tightness * (0.5 + 0.5 * usability)
    # This ensures perfect fits get ~0.5 * tightness(0) = 0.5
    # And useful gaps get higher usability
    
    # Actually, let's think differently:
    # The key insight is that we want to pack bins tightly.
    # A good next item is one that leaves a gap that either:
    # 1. Is zero (perfect fit)
    # 2. Can fit another item (useful gap)
    # Bad next items leave dead gaps.
    
    # Let's use: score = A / (1 + gap) + B * usability_indicator
    # where usability_indicator is smooth around min_d
    
    # Simpler approach: use the original structure but with smoother transitions
    # Perfect fits: very high
    # Gaps that can fit items: medium-high, decreasing with gap
    # Dead gaps: low
    
    # Smooth version:
    # score = C1 * exp(-gap / tau1) + C2 / (1 + gap) * smooth_step(gap, min_d)
    
    # Let's use a cleaner formulation:
    # The attractiveness should reflect how well item j completes the bin after item i.
    # A completion score based on how full the bin becomes.
    
    # Fill ratio after placing both: (demands[i] + demands[j]) / capacity
    # Higher fill ratio = better (more efficient use of space)
    # But we also need to consider if the remaining space is useful.
    
    fill_ratio = pair_sums[valid] / capacity
    
    # Score based on fill ratio: higher is better
    # But also penalize if the gap is "dead" (can't fit anything)
    
    # Smooth penalty for dead gaps
    # dead_penalty = 1 - smooth_step(gap, min_d) where smooth_step goes 0->1 as gap crosses min_d
    smooth_step = 1.0 / (1.0 + np.exp(-10.0 * (min_d - valid_gaps) / min_d))
    # When gap << min_d: smooth_step ~ 1 (dead)
    # When gap >> min_d: smooth_step ~ 0 (useful)
    
    # Score = fill_ratio * (1 - 0.5 * smooth_step)
    # Perfect fit: fill_ratio=1, smooth_step~1 -> 0.5
    # Useful tight: fill_ratio~0.9, smooth_step~0.5 -> 0.45
    # Useful loose: fill_ratio~0.5, smooth_step~0 -> 0.5
    # Dead gap: fill_ratio~0.9, smooth_step~1 -> 0.45
    
    # This doesn't distinguish well enough. Let me try:
    # score = (fill_ratio)^alpha * usability_factor
    # where usability_factor is high for useful gaps, low for dead gaps
    
    alpha = 2.0
    fill_score = fill_ratio ** alpha
    
    # Usability: 1 if gap >= min_d, 0 if gap < min_d, smoothed
    # Use a smooth transition
    usability_smooth = 1.0 / (1.0 + np.exp(-15.0 * (valid_gaps - min_d) / (min_d * 0.5)))
    
    # Combine: perfect fits should be highest
    # score = fill_score * (0.3 + 0.7 * usability_smooth)
    # Perfect: fill=1, usab~0.5 -> 0.65
    # Tight useful: fill~0.95, usab~0.9 -> 0.95^2 * 0.93 = 0.84
    # Loose useful: fill~0.5, usab~1.0 -> 0.25 * 1.0 = 0.25
    # Dead: fill~0.95, usab~0.1 -> 0.95^2 * 0.37 = 0.33
    
    # This makes tight useful better than perfect, which is wrong.
    
    # Let me go back to a simpler, more direct approach:
    # The original idea was good, just make it smoother.
    
    # Score = base_score * gap_factor * usability_factor
    # base_score: constant
    # gap_factor: decreases with gap (tighter is better)
    # usability_factor: 1 if gap can fit items, <1 if dead
    
    base = 100.0
    gap_factor = 1.0 / (1.0 + valid_gaps * 0.05)  # Smooth decay
    usability = 1.0 / (1.0 + np.exp(-20.0 * (min_d - valid_gaps) / min_d))
    # usability ~ 1 when gap << min_d (dead), ~0 when gap >> min_d (useful)
    # Wait, I want the opposite: high usability for useful gaps
    usability = 1.0 / (1.0 + np.exp(-20.0 * (valid_gaps - min_d) / min_d))
    # usability ~ 0 when gap << min_d (dead), ~1 when gap >> min_d (useful)
    
    scores = base * gap_factor * (0.5 + 0.5 * usability)
    
    # Perfect fits (gap=0): gap_factor=1, usability~0.5 -> 100 * 1 * 0.75 = 75
    # Small useful (gap~min_d=20): gap_factor~0.5, usability~0.5 -> 100 * 0.5 * 0.75 = 37.5
    # Large useful (gap~100): gap_factor~0.17, usability~1.0 -> 100 * 0.17 * 1.0 = 17
    # Dead gap (gap~10): gap_factor~0.83, usability~0.01 -> 100 * 0.83 * 0.505 = 42
    
    # Dead gaps score higher than large useful gaps, which is wrong.
    # I need to strongly penalize dead gaps.
    
    # Let's use: score = gap_factor * usability_weight
    # where usability_weight is very low for dead gaps
    
    # Alternative: just use the original discrete approach but with better constants
    # and add a small smooth component
    
    # Time check
    if time.time() > deadline:
        # Return a simple fallback
        eta[valid] = 1.0
        return eta
    
    # Let me try a different smooth approach:
    # The key is that dead gaps are bad because they waste space that can't be used.
    # Useful gaps are okay because they can be filled later.
    # Perfect fits are best.
    
    # Score = tightness_reward - dead_gap_penalty
    # tightness_reward = 1 / (1 + gap)
    # dead_gap_penalty = large if gap < min_d, small otherwise
    
    # Smooth penalty: penalty = K / (1 + exp(-(gap - min_d)/sigma))
    # When gap < min_d: penalty ~ K
    # When gap > min_d: penalty ~ 0
    
    K = 50.0
    sigma = min_d * 0.3
    penalty = K / (1.0 + np.exp((valid_gaps - min_d) / sigma))
    
    tightness = 1.0 / (1.0 + valid_gaps * 0.02)
    
    scores = tightness - penalty
    
    # Perfect: tightness=1, penalty~50 -> -49 (bad!)
    # This is wrong. Penalty should be 0 for perfect fits.
    
    # The issue is that perfect fits have gap=0 < min_d, so they get penalized.
    # I need to treat perfect fits separately or adjust the penalty.
    
    # Let's use: penalty only for 0 < gap < min_d
    # Smooth: penalty = K * sigmoid((gap - min_d)/sigma) * sigmoid((gap - 0)/sigma)
    # This is non-zero only when 0 < gap < min_d
    
    # Simpler: just use the original discrete approach with tuned constants.
    # The original scored 1.0325, so let's refine it slightly.
    
    # Use the original structure but with a smoother transition for useful gaps
    # and a slightly different dead gap penalty
    
    # Perfect fits
    perfect = valid & (gaps == 0)
    
    # Useful gaps (can fit at least one item)
    useful = valid & (gaps >= min_d)
    
    # Dead gaps
    dead = valid & (gaps > 0) & (gaps < min_d)
    
    # Perfect: high score
    eta[perfect] = 1000.0
    
    # Useful: score = A / (gap + B) with smooth decay
    # A and B tuned so that the score range is reasonable
    if np.any(useful):
        useful_gaps = gaps[useful]
        # Score decreases smoothly with gap
        # Use: score = 100 / (1 + gap/10)
        # gap=20: 100/3 = 33
        # gap=50: 100/6 = 16.7
        # gap=100: 100/11 = 9.1
        eta[useful] = 100.0 / (1.0 + useful_gaps / 10.0)
    
    # Dead gaps: low score, but not too low (1e-9 is effectively zero)
    # Use a score that's clearly below useful gaps but above 1e-9
    # Smallest useful score: 100 / (1 + 20/10) = 33
    # So dead gaps should be < 33. Let's use 5.0
    if np.any(dead):
        eta[dead] = 5.0
    
    return eta
# EVOLVE-BLOCK-END
