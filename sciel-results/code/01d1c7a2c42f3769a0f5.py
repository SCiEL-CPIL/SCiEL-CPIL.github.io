"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import random
import numpy as np
sys.setrecursionlimit(10000)

def construct():
    n = 256
    start_time = time.time()
    time_limit = 55.0  # Leave 5s buffer

    # Precompute APs
    ap_list = []
    index_aps = [[] for _ in range(n)]
    
    for d in range(1, n):
        max_len = (n - 1) // d + 1
        for L in range(1, max_len + 1):
            max_start = n - (L - 1) * d
            for a in range(max_start):
                indices = [a + k * d for k in range(L)]
                ap_id = len(ap_list)
                ap_list.append(indices)
                for idx in indices:
                    index_aps[idx].append(ap_id)

    # Convert to NumPy arrays for fast vectorized ops
    index_aps_np = [np.array(aps, dtype=np.int32) for aps in index_aps]
    ap_lengths = np.array([len(ap) for ap in ap_list], dtype=np.int32)
    
    best_s = None
    best_B = float('inf')
    rng = random.Random(42)
    
    B = 1
    while time.time() - start_time < time_limit:
        s = np.zeros(n, dtype=np.int8)
        ap_sums = np.zeros(len(ap_list), dtype=np.int32)
        unassigned_counts = ap_lengths.copy()
        
        t_start_B = time.time()
        
        def is_consistent(k, sigma):
            idx = index_aps_np[k]
            if idx.size == 0:
                return True
            new_sums = ap_sums[idx] + sigma
            remaining = unassigned_counts[idx] - 1
            # Check if any AP sum cannot be brought back to [-B, B]
            # Max possible future sum: new_sum + remaining
            # Min possible future sum: new_sum - remaining
            if np.any(new_sums - remaining > B) or np.any(new_sums + remaining < -B):
                return False
            return True

        def bt(k):
            nonlocal s, ap_sums, unassigned_counts
            if time.time() - t_start_B > 5.0:
                return False
            if k == n:
                return True
            
            idx = index_aps_np[k]
            
            # Determine value order
            # If sum of current AP sums is positive, try -1 first
            # If negative, try 1 first
            # If zero, randomize
            if idx.size > 0:
                pressure = np.sum(ap_sums[idx])
                if pressure > 0:
                    vals = [-1, 1]
                elif pressure < 0:
                    vals = [1, -1]
                else:
                    # Randomize to avoid symmetry breaking issues
                    if rng.random() < 0.5:
                        vals = [1, -1]
                    else:
                        vals = [-1, 1]
            else:
                vals = [1, -1]
                
            for sigma in vals:
                if is_consistent(k, sigma):
                    s[k] = sigma
                    ap_sums[idx] += sigma
                    unassigned_counts[idx] -= 1
                    if bt(k + 1):
                        return True
                    ap_sums[idx] -= sigma
                    unassigned_counts[idx] += 1
                    s[k] = 0
            return False

        if bt(0):
            best_s = s.tolist()
            best_B = B
            break
        else:
            B += 1
            if B > 10:
                break

    if best_s is None:
        rng_fallback = random.Random(42)
        best_s = [rng_fallback.choice([-1, 1]) for _ in range(n)]
        
    return best_s

# EVOLVE-BLOCK-END
