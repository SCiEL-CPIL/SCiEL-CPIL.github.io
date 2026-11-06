"""Improved program: subgradient descent on smooth surrogate + fast local search."""
# EVOLVE-BLOCK-START
import time
import math
import random
from collections import defaultdict

N = 256
TIME_LIMIT = 80.0  # safety margin before 90s kill

def _precompute_aps():
    """Precompute all arithmetic progressions and their index lists."""
    aps = []
    for start in range(1, N + 1):
        for step in range(1, N + 1):
            length = 1
            # Find max length for this start, step
            while start + length * step <= N:
                length += 1
            length -= 1
            if length < 1:
                continue
            # For each possible sub-length? No, the problem says "any length including one"
            # Actually, re-reading: "any start, positive step and length including one"
            # This means we consider ALL sub-sequences? No, an AP is defined by start, step, length.
            # The sum is over that specific length.
            # So for a given (start, step), we have lengths 1, 2, ..., max_len.
            # We need to track the max absolute sum over ALL such (start, step, length).
            # This is a huge number of APs.
            # Let's count: sum_{s=1}^N sum_{d=1}^N floor((N-s)/d) + 1
            # This is approx N * sum_{d=1}^N N/d ~ N^2 log N ~ 256^2 * 6 ~ 400k.
            # But we need to store the indices for each? That's memory heavy.
            # Better: store (start, step, length) and compute sum on the fly?
            # Or store the list of indices.
            # Let's store the list of indices for each AP.
            indices = list(range(start - 1, start - 1 + length * step, step))
            aps.append(indices)
    return aps

def _max_abs_sum(signs, aps):
    """Compute the maximum absolute sum over all APs."""
    max_val = 0
    for ap in aps:
        s = 0
        for i in ap:
            s += signs[i]
        a = abs(s)
        if a > max_val:
            max_val = a
    return max_val

def _fast_max_abs_sum(signs, aps):
    """Faster computation using numpy if available, or optimized loops."""
    import numpy as np
    signs_np = np.array(signs)
    max_val = 0
    for ap in aps:
        s = signs_np[ap].sum()
        a = abs(s)
        if a > max_val:
            max_val = a
    return max_val

def construct():
    start_time = time.time()
    
    # Precompute APs
    aps = _precompute_aps()
    
    # Initialize with random signs
    rng = random.Random(42)
    signs = [rng.choice((-1, 1)) for _ in range(N)]
    
    best_signs = signs[:]
    best_score = _fast_max_abs_sum(signs, aps)
    
    # Subgradient descent on continuous surrogate
    # Objective: sum of squares of all AP sums
    # Gradient: for each position i, grad[i] = sum over all APs containing i of 2 * sum(AP) * sign(i in AP)
    # Since signs are ±1, sign(i in AP) is 1.
    # So grad[i] = 2 * sum_{APs containing i} sum(AP)
    
    # Precompute for each position, which APs contain it
    pos_to_aps = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for i in ap:
            pos_to_aps[i].append(idx)
    
    # Compute initial AP sums
    import numpy as np
    signs_np = np.array(signs, dtype=np.float64)
    ap_sums = np.array([signs_np[ap].sum() for ap in aps], dtype=np.float64)
    
    # Subgradient descent
    lr = 0.1
    decay = 0.999
    step_count = 0
    
    while time.time() - start_time < TIME_LIMIT * 0.4:
        # Compute gradient
        grad = np.zeros(N, dtype=np.float64)
        for i in range(N):
            if pos_to_aps[i]:
                # Sum of ap_sums for APs containing i
                grad[i] = 2.0 * np.sum(ap_sums[pos_to_aps[i]])
        
        # Update
        signs_np -= lr * grad
        step_count += 1
        
        # Recompute ap_sums
        for idx, ap in enumerate(aps):
            ap_sums[idx] = signs_np[ap].sum()
        
        # Project to ±1
        signs_proj = np.where(signs_np >= 0, 1, -1).astype(int)
        
        # Evaluate
        score = int(abs(np.array(signs_proj)[aps[0]].sum()))  # dummy, will use fast
        score = _fast_max_abs_sum(signs_proj.tolist(), aps)
        
        if score < best_score:
            best_score = score
            best_signs = signs_proj.tolist()
        
        lr *= decay
        
        if lr < 1e-4:
            break
    
    # Local search: flip single bits
    current_signs = best_signs[:]
    current_score = best_score
    
    # Precompute for local search: for each position, which APs contain it
    # Already have pos_to_aps
    
    # Use histogram-based max tracking for speed
    # Maintain counts of absolute AP sums
    # For speed, let's just do simple local search with fast evaluation
    
    while time.time() - start_time < TIME_LIMIT * 0.8:
        improved = False
        for i in range(N):
            # Try flipping position i
            old_val = current_signs[i]
            current_signs[i] = -old_val
            
            new_score = _fast_max_abs_sum(current_signs, aps)
            
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                current_signs[i] = old_val
        
        if not improved:
            break
    
    # Final check
    if current_score < best_score:
        best_score = current_score
        best_signs = current_signs[:]
    
    return best_signs
# EVOLVE-BLOCK-END
