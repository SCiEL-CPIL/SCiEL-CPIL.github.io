import time
import random
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 110.0
    
    # Precompute all 4-term arithmetic progressions
    aps = []
    max_d = (n - 1) // 3
    for d in range(1, max_d + 1):
        max_a = n - 1 - 3 * d
        for a in range(max_a + 1):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
            
    # Convert to numpy array for fast indexing
    # Shape: (num_aps, 4)
    aps_arr = np.array(aps, dtype=np.int32)
    num_aps = len(aps_arr)
    
    def count_mono(c):
        # c is a numpy array of 0s and 1s
        # Check if all 4 elements are equal
        # c[aps_arr[:, 0]] == c[aps_arr[:, 1]] etc
        # Using broadcasting or direct indexing
        c0 = c[aps_arr[:, 0]]
        c1 = c[aps_arr[:, 1]]
        c2 = c[aps_arr[:, 2]]
        c3 = c[aps_arr[:, 3]]
        
        # Mono if c0==c1==c2==c3
        # (c0==c1) & (c1==c2) & (c2==c3)
        mono_mask = (c0 == c1) & (c1 == c2) & (c2 == c3)
        return int(np.sum(mono_mask))

    best_cost = float('inf')
    best_c = None
    
    # Use random restarts with local search (hill climbing)
    # Each restart:
    # 1. Init random c
    # 2. Loop:
    #    a. Try flipping each i? No, that's O(N) evaluations.
    #    b. Instead, pick a random i, flip it, check cost.
    #       If better, keep. If not, revert.
    #       This is stochastic local search.
    #       Or: Scan all i, find best improvement.
    #       Scanning all i with NumPy:
    #       We can't easily vectorize the "flip and check" for all i at once.
    #       So we loop i from 0 to N-1.
    #       For each i, flip, count, revert.
    #       This is N evaluations per step.
    #       256 evaluations * 1ms/eval = 256ms per step.
    #       110s / 0.256s = ~400 steps.
    #       This might be too slow for thorough search.
        
    # Better: Use the delta calculation logic but with NumPy?
    # Or just use the Python list version which was estimated to be fast?
    # Let's re-evaluate Python list speed.
    # 5000 ops per step. Python does ~10-50M ops/sec.
    # So ~1000-5000 steps/sec.
    # 110s * 2000 steps/s = 220,000 steps.
    # This is much better than the NumPy brute-force flip-check.
    
    # So, stick with the Python list version for the local search step, 
    # but optimize the inner loop.
    
    # Let's refine the Python list version.
    # Precompute ap_indices as lists of tuples of other indices?
    # ap_indices[i] = [(ap_idx, other1, other2, other3), ...]
    # Then in delta_cost, we don't need to look up 'aps' and extract others.
    
    ap_indices_opt = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for pos, val in enumerate(ap):
            # Get the other 3 indices
            others = []
            for p in range(4):
                if p != pos:
                    others.append(ap[p])
            ap_indices_opt[val].append((idx, others))
            
    # Note: We don't actually need idx if we just check the values.
    # We just need the other 3 indices.
    # So ap_indices_opt[i] can be a list of tuples (o1, o2, o3).
    
    ap_indices_opt2 = [[] for _ in range(n)]
    for ap in aps:
        for pos, val in enumerate(ap):
            others = []
            for p in range(4):
                if p != pos:
                    others.append(ap[p])
            ap_indices_opt2[val].append((others[0], others[1], others[2]))

    def delta_cost_fast(c, i):
        old_val = c[i]
        new_val = 1 - old_val
        delta = 0
        for o1, o2, o3 in ap_indices_opt2[i]:
            # Check was_mono
            # c[o1] == old_val and c[o2] == old_val and c[o3] == old_val
            if c[o1] == old_val and c[o2] == old_val and c[o3] == old_val:
                # Was mono
                # Check is_mono
                # c[o1] == new_val and c[o2] == new_val and c[o3] == new_val
                if c[o1] == new_val and c[o2] == new_val and c[o3] == new_val:
                    # Both mono? Impossible if old_val != new_val
                    # Wait, if c[o1]==old_val, it cannot be == new_val.
                    # So if it was mono, it cannot be mono after flip.
                    # So delta -= 1
                    delta -= 1
                else:
                    # Was mono, not mono after. delta -= 1
                    delta -= 1
            else:
                # Not was mono
                # Check is_mono
                if c[o1] == new_val and c[o2] == new_val and c[o3] == new_val:
                    # Not was mono, is mono after. delta += 1
                    delta += 1
                else:
                    # Neither. delta += 0
                    pass
        return delta

    def count_mono_fast(c):
        count = 0
        for ap in aps:
            if c[ap[0]] == c[ap[1]] == c[ap[2]] == c[ap[3]]:
                count += 1
        return count

    best_cost = float('inf')
    best_c = None
    
    while time.time() < deadline:
        c = [random.randint(0, 1) for _ in range(n)]
        current_cost = count_mono_fast(c)
        
        # Local search
        while time.time() < deadline:
            best_delta = 0
            best_i = -1
            
            # Scan all bits
            for i in range(n):
                d = delta_cost_fast(c, i)
                if d < best_delta:
                    best_delta = d
                    best_i = i
                    
            if best_i != -1 and best_delta < 0:
                c[best_i] = 1 - c[best_i]
                current_cost += best_delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_c = c[:]
            else:
                # Stuck. Kick.
                kick_size = random.randint(1, 3)
                for _ in range(kick_size):
                    k = random.randint(0, n - 1)
                    c[k] = 1 - c[k]
                current_cost = count_mono_fast(c)
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_c = c[:]
                # Break to restart
                break

    if best_c is None:
        return [(i // 2) % 2 for i in range(256)]
        
    return best_c
