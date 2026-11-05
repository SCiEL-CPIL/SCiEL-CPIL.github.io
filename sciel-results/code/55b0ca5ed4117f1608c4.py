import numpy as np
import time
import random

def construct():
    n = 129 # 0 to 128
    target_min = 128
    target_max = 256
    targets = np.arange(target_min, target_max + 1)
    
    def get_stats(x):
        # x is numpy array of 0s and 1s
        # Returns (peak, count, valid)
        conv = np.convolve(x, x, mode='full')
        # conv[k] is ordered count
        # Unordered R[k]:
        # If k odd: conv[k] // 2
        # If k even: (conv[k] + x[k//2]) // 2
        
        R = np.zeros(len(targets), dtype=np.int32)
        
        # Vectorized parity check
        even_mask = (targets % 2 == 0)
        odd_mask = ~even_mask
        
        # Odd targets
        R[odd_mask] = conv[targets[odd_mask]] // 2
        
        # Even targets
        even_targets = targets[even_mask]
        mid_indices = even_targets // 2
        mid_vals = x[mid_indices]
        R[even_mask] = (conv[even_targets] + mid_vals) // 2
        
        # Validity: all R[k] >= 1
        valid = np.all(R >= 1)
        
        # Peak multiplicity
        peak = np.max(R)
        
        # Size
        size = np.sum(x)
        
        return peak, size, valid, R

    # 1. Greedy Construction
    # Start with empty set
    x = np.zeros(n, dtype=np.int8)
    
    # We want to cover all targets.
    # Strategy: Iteratively add the element that covers the most uncovered targets 
    # with the least penalty on peak.
    # However, a simpler greedy: just ensure we have enough elements.
    # A known good structure for additive bases is often dense or structured.
    # Let's try to build a set that covers [128, 256].
    # If A contains [0, 128], it covers everything but size is 129, peak is high.
    # We want small A.
    
    # Greedy Step:
    # Track uncovered targets
    uncovered = np.ones(len(targets), dtype=bool)
    
    # To speed up, we can pre-calculate contribution of each candidate element
    # But n is small (129), so we can just iterate.
    
    # Heuristic: Add elements in a way that spreads coverage.
    # Let's start with a random-ish greedy that prefers elements that complete many pairs.
    
    # Actually, a better greedy for this specific problem (covering an interval)
    # is to ensure that the sumset A+A covers the interval.
    # Let's use a simple heuristic:
    # 1. Add 0 and 128 (extremes help cover ends? No, 0+128=128, 128+128=256).
    x[0] = 1
    x[128] = 1
    # Update uncovered
    _, _, valid, R = get_stats(x)
    uncovered = R < 1
    
    # Now, greedily add elements that cover the most uncovered targets.
    # We will iterate until all are covered.
    max_iter = 100
    for _ in range(max_iter):
        if not np.any(uncovered):
            break
        
        best_idx = -1
        best_cover_count = -1
        best_peak_increase = 999
        
        # Try all candidates not in A
        candidates = np.nonzero(x == 0)[0]
        
        for idx in candidates:
            # Simulate adding idx
            x[idx] = 1
            peak, size, valid, R_new = get_stats(x)
            
            # Count how many new targets are covered
            # R_new > 0 where R_old was 0?
            # We don't have R_old stored easily, but we can check R_new >= 1
            # And we know which were uncovered.
            # new_covered = np.sum((R_new >= 1) & uncovered)
            
            # Actually, simpler: just check if it reduces the number of uncovered.
            # We need to be careful with the peak.
            # If peak increases too much, it's bad.
            
            # Let's just pick the one that minimizes the resulting peak, 
            # breaking ties by maximizing coverage of currently uncovered.
            
            if valid:
                # If it makes the whole set valid, great.
                cover_gain = np.sum((R_new >= 1) & uncovered)
            else:
                cover_gain = np.sum((R_new >= 1) & uncovered)
            
            # Metric: Minimize (peak * 1000 - cover_gain)
            # Or just: If peak is lower than current best, take it.
            # If peak is same, take higher cover_gain.
            
            # Revert for next iteration
            x[idx] = 0
            
            # Current peak of x (without idx)
            # We should compare against current peak.
            # Let's just track the best candidate based on (peak, -cover_gain)
            
            # Optimization: Only evaluate if it has potential.
            # For simplicity in this rewrite, we do a full scan.
            
            # To make it faster, we can just pick the candidate that adds the most
            # coverage to the "hardest" targets.
            
            # Let's use a simpler greedy:
            # Pick the element that maximizes the minimum representation count
            # among the currently uncovered targets? No, that's complex.
            
            # Let's just pick the one that results in the lowest peak, 
            # preferring higher coverage if peaks are equal.
            
            if best_idx == -1:
                best_idx = idx
                best_peak = peak
                best_cover = cover_gain
            else:
                if peak < best_peak or (peak == best_peak and cover_gain > best_cover):
                    best_idx = idx
                    best_peak = peak
                    best_cover = cover_gain
        
        if best_idx == -1:
            break
            
        # Apply best
        x[best_idx] = 1
        _, _, valid, R = get_stats(x)
        uncovered = R < 1
        
    # Ensure validity. If not valid, add random elements until valid.
    while True:
        _, _, valid, R = get_stats(x)
        if valid:
            break
        # Find an uncovered target
        uncov_targets = targets[R < 1]
        if len(uncov_targets) == 0:
             break
        t = uncov_targets[0]
        # Find a pair (a, b) such that a+b=t and (a,b) not both in A?
        # Just pick a random a in [0, t] and add t-a if in range.
        # To be deterministic, pick a=0, add t.
        # If t <= 128, add t.
        # If t > 128, add t-0? No, t-0=t is out of range if t>128.
        # If t > 128, we can add 128 and t-128?
        # Let's just add a random element that helps.
        # Simple fix: add 0 and 128 are already there.
        # If t is uncovered, it means no a,b in A sum to t.
        # Add a random candidate.
        rand_idx = random.randint(0, 128)
        x[rand_idx] = 1
        # Limit iterations to avoid infinite loop
        # (This loop should rarely run if greedy worked)
        if np.sum(x) > 128:
             # Fallback: return full set? No, that's bad.
             # Just break and hope local search fixes it?
             # Actually, if greedy failed, we have a problem.
             # Let's force validity by adding specific complements.
             for t in uncov_targets:
                 if t <= 128:
                     x[t] = 1
                 else:
                     # t in 129..256
                     # t = 128 + (t-128). If 128 in A, we need t-128 in A.
                     x[t-128] = 1
             break

    # 2. Local Search (Hill Climbing)
    # Objective: Minimize Peak. Secondary: Minimize Size.
    
    current_peak, current_size, current_valid, _ = get_stats(x)
    
    if not current_valid:
        # Should not happen, but if it does, try to fix
        pass

    # Removal Phase
    # Try to remove elements if it doesn't increase peak and keeps validity.
    changed = True
    while changed:
        changed = False
        for i in range(n):
            if x[i] == 1:
                x[i] = 0
                peak, size, valid, _ = get_stats(x)
                if valid and peak <= current_peak:
                    # Accept removal
                    current_peak = peak
                    current_size = size
                    changed = True
                else:
                    x[i] = 1
                    # Restore state
                    pass # x[i] is already 1

    # Swap Phase
    # Try to swap an element i in A with j not in A.
    # Accept if peak decreases, or (peak same and size same? No, size is same in swap).
    # Actually, swap keeps size same. So we just look for peak reduction.
    
    max_swap_iters = 50
    for _ in range(max_swap_iters):
        improved = False
        for i in range(n):
            if x[i] == 1:
                for j in range(n):
                    if x[j] == 0:
                        # Swap i and j
                        x[i] = 0
                        x[j] = 1
                        
                        peak, size, valid, _ = get_stats(x)
                        
                        if valid and peak < current_peak:
                            current_peak = peak
                            improved = True
                            # Break inner loop? Or continue to find more?
                            # Let's break to restart the scan with new state
                            break
                        else:
                            # Revert
                            x[j] = 0
                            x[i] = 1
                if improved:
                    break
        
        if not improved:
            break
            
    # Final Result
    result = np.nonzero(x)[0].tolist()
    
    if not result:
        return [64]
        
    return result
