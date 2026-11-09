"""Improved program for low discrepancy on arithmetic progressions."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 50  # 10s margin before 60s limit
    
    # Precompute all arithmetic progressions
    # An AP is defined by (start, step, length) where start + (length-1)*step < n
    # We need to minimize the max absolute sum over all such APs
    
    # For efficiency, precompute the APs as lists of indices
    # But there are too many APs to store all, so we'll compute on the fly
    # or use a smarter approach
    
    # Key insight: The worst-case APs are likely to be short (length 1, 2, 3)
    # or have small steps. Let's focus on the most constraining APs.
    
    # For length 1: sum is just the sign, so |sum| = 1 always.
    # For length 2: sum is s[i] + s[i+k], so |sum| can be 0 or 2.
    # For length 3: sum is s[i] + s[i+k] + s[i+2k], |sum| can be 1 or 3.
    
    # The maximum absolute sum is at least 1 (from singletons).
    # We want to minimize the maximum, so ideally we want it to be 1 or 2.
    
    # Let's use a local search approach:
    # 1. Start with a random assignment
    # 2. Compute the current max absolute sum
    # 3. Try flipping individual signs to reduce the max
    # 4. Repeat until no improvement or time runs out
    
    def compute_max_abs_sum(signs):
        """Compute the maximum absolute signed sum over all APs."""
        max_abs = 0
        s = signs
        nn = n
        # Iterate over all steps k from 1 to nn-1
        for k in range(1, nn):
            # For each start i, compute the AP sum
            for i in range(nn):
                curr = 0
                idx = i
                while idx < nn:
                    curr += s[idx]
                    if curr > max_abs:
                        max_abs = curr
                    elif curr < -max_abs:
                        max_abs = -curr
                    idx += k
                    if max_abs >= 4:  # Early exit if we already have a high value
                        return max_abs
        return max_abs
    
    def compute_max_abs_sum_fast(signs):
        """Faster version using numpy for vectorized operations where possible."""
        max_abs = 0
        s = signs
        nn = n
        for k in range(1, nn):
            # For step k, the APs starting at i=0,1,...,k-1 cover all positions
            # But we need all starts, not just i < k
            # Actually, for a fixed step k, the APs are:
            # start=0: 0, k, 2k, ...
            # start=1: 1, 1+k, 1+2k, ...
            # ...
            # start=k-1: k-1, 2k-1, 3k-1, ...
            # And starts >= k are subsets of these (they start later in the same residue class)
            # So we only need to check starts 0 to k-1, and for each, all prefixes
            
            for i in range(k):
                curr = 0
                idx = i
                while idx < nn:
                    curr += s[idx]
                    if curr > max_abs:
                        max_abs = curr
                    elif curr < -max_abs:
                        max_abs = -curr
                    idx += k
                    if max_abs >= 4:
                        return max_abs
        return max_abs
    
    # Initialize with random assignment
    rng = random.Random(12345)
    best_signs = [rng.choice((-1, 1)) for _ in range(n)]
    best_max = compute_max_abs_sum_fast(best_signs)
    
    # Local search: try flipping each position
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        for pos in range(n):
            if (time.time() - start_time) > deadline:
                break
            # Try flipping position pos
            old_val = best_signs[pos]
            best_signs[pos] = -old_val
            new_max = compute_max_abs_sum_fast(best_signs)
            if new_max < best_max:
                best_max = new_max
                improved = True
            else:
                best_signs[pos] = old_val
    
    # If we still have time, try some random restarts
    while (time.time() - start_time) < deadline - 5:
        # Random restart
        candidate = [rng.choice((-1, 1)) for _ in range(n)]
        cand_max = compute_max_abs_sum_fast(candidate)
        
        # Local search on candidate
        for _ in range(100):  # Limit iterations per restart
            if (time.time() - start_time) > deadline:
                break
            local_improved = False
            for pos in range(n):
                if (time.time() - start_time) > deadline:
                    break
                old_val = candidate[pos]
                candidate[pos] = -old_val
                new_max = compute_max_abs_sum_fast(candidate)
                if new_max < cand_max:
                    cand_max = new_max
                    local_improved = True
                else:
                    candidate[pos] = old_val
                if not local_improved:
                    break
            if not local_improved:
                break
        
        if cand_max < best_max:
            best_max = cand_max
            best_signs = candidate[:]
    
    return best_signs
# EVOLVE-BLOCK-END
