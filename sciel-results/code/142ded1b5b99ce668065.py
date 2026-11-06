# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 50.0  # safety margin within 90s limit
    N = 256
    
    # Precompute all APs: (start, step, length) where 1 <= start, start + (length-1)*step <= N
    # We need to minimize max |sum over AP|
    
    # Strategy: Use simulated annealing with a carefully designed cost function
    # and precomputed AP data for fast evaluation.
    
    # Generate all APs
    aps = []  # list of index arrays (0-indexed)
    for step in range(1, N):
        for start in range(0, N):
            # Build the AP
            ap = []
            pos = start
            while pos < N:
                ap.append(pos)
                pos += step
                if time.time() > deadline - 40:
                    # Still generating, but let's be careful
                    pass
            if len(ap) >= 1:
                aps.append(ap)
    
    # Actually, for step=1, we have all contiguous subsequences.
    # Number of APs: for step s, number of starts is N, but APs of length>=1
    # Total count: sum over s=1 to N-1 of N = N*(N-1) ~ 65536, plus step=N gives singletons? 
    # Wait, step ranges. Let me reconsider.
    
    # An AP is defined by start, step, length. The problem says "any start, positive step and length including one".
    # So step >= 1. For a given start and step, the AP is determined, and its length is as long as it fits.
    # But actually, an AP can be any contiguous subsequence of the arithmetic progression?
    # "any arithmetic progression contained in [1,256]" - this means any subset of form {a, a+d, a+2d, ...} 
    # that is fully contained in [1, 256]. So for each (a, d), the full AP is {a, a+d, ..., a+kd} where a+kd <= 256.
    # But wait, it says "length including one", so even single elements count.
    # And "any start, positive step and length" - I think this means we consider all APs of any length >= 1.
    # So for each (start, step), we consider the maximal AP, but also all its prefixes? 
    # Actually, "any arithmetic progression" means any set {a, a+d, ..., a+ld} with l+1 >= 1 terms.
    # So for fixed (a, d), we have APs of length 1, 2, 3, ... up to the max that fits.
    
    # Let me regenerate properly.
    aps = []
    for start in range(N):  # 0-indexed start
        for step in range(1, N):
            ap = []
            pos = start
            while pos < N:
                ap.append(pos)
                pos += step
                if not ap:
                    break
            # Add all prefixes of this maximal AP as separate APs? 
            # Actually, each prefix is itself an AP. So for the maximal AP of length L,
            # we have L APs: prefix of length 1, 2, ..., L.
            # But that's a lot. Let me think about the count.
            # For step=1, start=0: APs are [0], [0,1], [0,1,2], ..., [0,1,...,255] -> 256 APs
            # For step=1, start=1: [1], [1,2], ..., [1,2,...,255] -> 255 APs
            # ...
            # This is O(N^3) total APs which is ~16 million. That might be too many to store.
            
            # Alternative: just store the maximal APs and compute prefix sums.
            # For each maximal AP, we can compute all prefix sums incrementally.
            pass
    
    # Let's use a different approach: iterate over all (start, step) and compute the running sum.
    # We'll do simulated annealing with incremental cost updates.
    
    # Precompute for each position, the list of (start, step) APs that contain it? That's complex.
    
    # Instead, let's just evaluate the max AP sum from scratch each time, but optimize with NumPy.
    
    def max_ap_sum(signs):
        """Compute max |sum| over all APs."""
        signs_arr = np.array(signs, dtype=np.int32)
        max_abs = 1  # at least 1 for singletons
        
        for step in range(1, N):
            # For each start, compute the AP sum incrementally
            # But doing this in Python is slow. Let's vectorize per step.
            # For a fixed step, the APs starting at positions 0, 1, ..., step-1 
            # (modulo step) are independent chains.
            # Actually, for fixed step, positions with same residue mod step form chains.
            for r in range(step):
                # Positions: r, r+step, r+2*step, ...
                if r >= N:
                    break
                # Get the chain
                chain = signs_arr[r::step]
                # Compute all prefix sums
                if len(chain) == 0:
                    continue
                cumsum = np.cumsum(chain)
                max_abs = max(max_abs, int(np.max(np.abs(cumsum))))
        
        return max_abs
    
    # This is O(N^2) per evaluation which is ~65536 operations, times number of steps... 
    # Actually for step=1, we do N chains of avg length N/2, so ~N^2/2.
    # For all steps, it's ~N^2 * H(N) ~ N^2 * log(N) ~ 256^2 * 8 ~ 500K operations per eval.
    # In NumPy, this should be fast enough for ~10K iterations in 50s.
    
    # Initialize with random
    rng = random.Random(42)
    signs = [rng.choice((-1, 1)) for _ in range(N)]
    
    current_cost = max_ap_sum(signs)
    best_signs = signs[:]
    best_cost = current_cost
    
    # Simulated annealing
    T = 10.0
    T_min = 0.1
    alpha = 0.999
    
    while time.time() < deadline - 5.0:
        # Pick a random position to flip
        i = rng.randint(0, N - 1)
        signs[i] = -signs[i]
        
        new_cost = max_ap_sum(signs)
        
        # Accept if better or with probability exp(-(new_cost - current_cost)/T)
        if new_cost < current_cost:
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_signs = signs[:]
        else:
            # Probability of accepting worse solution
            if new_cost == current_cost:
                accept = True
            else:
                diff = new_cost - current_cost
                p = np.exp(-diff / T) if T > 0 else 0.0
                accept = (rng.random() < p)
            
            if accept:
                current_cost = new_cost
            else:
                signs[i] = -signs[i]  # revert
        
        T *= alpha
        if T < T_min:
            T = T_min
    
    return best_signs
# EVOLVE-BLOCK-END
