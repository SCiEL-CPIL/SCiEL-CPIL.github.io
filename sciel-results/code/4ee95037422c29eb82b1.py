"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    n = 256
    
    # Precompute all arithmetic progressions (APs) in [1, 256]
    # An AP is defined by (start, step, length) where:
    #   start >= 1, step >= 1, length >= 1
    #   start + (length-1)*step <= 256
    # We need to minimize max |sum of signs on any AP|
    
    # For efficiency, we'll focus on the most impactful APs.
    # The worst-case discrepancy is often achieved by short APs (length 1, 2, 3)
    # or by APs with small steps.
    
    # Let's generate a representative set of APs.
    # All singletons (length 1): sum is just the sign, so |sum| = 1 always.
    # So the minimum possible max abs sum is at least 1.
    # Length 2: (a, a+d) -> sum = s[a] + s[a+d]
    # Length 3: (a, a+d, a+2d) -> sum = s[a] + s[a+d] + s[a+2d]
    # etc.
    
    # Since singletons always give |sum|=1, the answer is at least 1.
    # If we can achieve max abs sum = 1 for all APs, that's optimal.
    # But that's impossible for length-2 APs: s[a]+s[a+d] can be -2, 0, or 2.
    # So we need to balance.
    
    # Strategy: Use a greedy/local search approach.
    # 1. Start with a random assignment.
    # 2. Compute the max abs AP sum.
    # 3. Find the AP(s) achieving the max.
    # 4. Flip a sign in that AP to reduce the discrepancy.
    # 5. Repeat until time runs out.
    
    # For efficiency, we'll precompute APs as index arrays.
    # But there are too many APs to store all. Let's be selective.
    
    # Key insight: The maximum discrepancy is likely achieved by:
    # - Length 1: always 1
    # - Length 2: small steps (d=1,2,3,...)
    # - Length 3: small steps
    # - Longer APs with small steps
    
    # Let's generate APs up to a certain length and step.
    
    def generate_aps(n, max_step=20, max_len=10):
        """Generate a subset of APs for evaluation."""
        aps = []
        # Singletons
        for i in range(n):
            aps.append((i,))
        
        # Length 2
        for d in range(1, min(max_step+1, n)):
            for start in range(n - d):
                aps.append((start, start + d))
        
        # Length 3+
        for d in range(1, min(max_step+1, n // 2 + 1)):
            for length in range(3, max_len+1):
                max_start = n - (length - 1) * d
                if max_start <= 0:
                    break
                for start in range(max_start):
                    aps.append(tuple(start + k * d for k in range(length)))
        
        return aps
    
    # This might still be too many. Let's count:
    # d up to 20, length up to 10
    # For d=1: length 2: 255, length 3: 254, ..., length 10: 247 -> ~2500
    # For d=2: length 2: 254, length 3: 252, ..., length 10: 239 -> ~2500
    # ... roughly 20 * 2500 = 50,000 APs
    # That's manageable.
    
    aps = generate_aps(n, max_step=15, max_len=8)
    aps = np.array(aps, dtype=np.int32)
    
    def compute_max_abs_sum(signs, aps):
        """Compute max absolute signed sum over all APs."""
        # signs is array of -1/+1
        # For each AP, sum signs[indices]
        # Use matrix multiplication or direct sum
        # aps shape: (num_aps, max_len) with -1 padding? No, variable length.
        # Better: iterate or use a different approach.
        
        # Since APs have variable length, let's use a different method.
        # We'll compute sums by iterating over APs, but vectorize where possible.
        
        max_abs = 0
        # Group by length for efficiency
        from collections import defaultdict
        by_len = defaultdict(list)
        for ap in aps:
            by_len[len(ap)].append(ap)
        
        for length, ap_list in by_len.items():
            ap_arr = np.array(ap_list, dtype=np.int32)
            sums = signs[ap_arr].sum(axis=1)
            local_max = np.max(np.abs(sums))
            if local_max > max_abs:
                max_abs = local_max
        
        return max_abs
    
    def find_worst_ap(signs, aps):
        """Find the AP with maximum absolute sum."""
        from collections import defaultdict
        by_len = defaultdict(list)
        for ap in aps:
            by_len[len(ap)].append(ap)
        
        global_max = 0
        worst_ap = None
        for length, ap_list in by_len.items():
            ap_arr = np.array(ap_list, dtype=np.int32)
            sums = signs[ap_arr].sum(axis=1)
            abs_sums = np.abs(sums)
            idx = np.argmax(abs_sums)
            if abs_sums[idx] > global_max:
                global_max = abs_sums[idx]
                worst_ap = tuple(ap_list[idx])
        
        return worst_ap, global_max
    
    # Initial random assignment
    rng = np.random.RandomState(42)
    signs = rng.choice([-1, 1], size=n).astype(np.int32)
    
    # Evaluate initial
    current_max = compute_max_abs_sum(signs, aps)
    best_signs = signs.copy()
    best_max = current_max
    
    # Local search: flip one sign at a time to reduce max AP sum
    # Focus on signs that appear in the worst AP
    
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        # Find worst AP
        worst_ap, worst_val = find_worst_ap(signs, aps)
        
        if worst_val <= 1:
            break  # Optimal (can't do better than 1 due to singletons)
        
        # Try flipping each sign in the worst AP
        best_flip = None
        best_new_max = worst_val
        
        for idx in worst_ap:
            # Flip sign at idx
            old_sign = signs[idx]
            signs[idx] = -old_sign
            
            new_max = compute_max_abs_sum(signs, aps)
            
            if new_max < best_new_max:
                best_new_max = new_max
                best_flip = idx
            
            # Revert
            signs[idx] = old_sign
        
        if best_flip is not None:
            signs[best_flip] = -signs[best_flip]
            current_max = best_new_max
            if current_max < best_max:
                best_max = current_max
                best_signs = signs.copy()
        else:
            # No improvement from flipping in worst AP
            # Try random flips
            for _ in range(5):
                if time.time() >= deadline:
                    break
                idx = rng.randint(0, n)
                old_sign = signs[idx]
                signs[idx] = -old_sign
                new_max = compute_max_abs_sum(signs, aps)
                if new_max < best_max:
                    best_max = new_max
                    best_signs = signs.copy()
                else:
                    signs[idx] = old_sign
    
    return best_signs.tolist()
# EVOLVE-BLOCK-END
