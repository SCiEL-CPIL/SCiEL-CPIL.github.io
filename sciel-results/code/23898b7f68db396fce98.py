import time
import random
import numpy as np

def construct():
    n = 256
    t0 = time.time()
    deadline = t0 + 55
    
    # 1. Precompute all arithmetic progressions
    progressions = []
    index_to_progs = [[] for _ in range(n)]
    
    pid = 0
    for d in range(1, n):
        for a in range(d):
            max_k = (n - 1 - a) // d + 1
            for k in range(1, max_k + 1):
                indices = tuple(a + j * d for j in range(k))
                progressions.append(indices)
                for idx in indices:
                    index_to_progs[idx].append(pid)
                pid += 1
                
    num_progs = len(progressions)
    
    # 2. Greedy initialization: build signs one by one
    # For each position, choose +1 or -1 to minimize the worst AP sum
    signs = np.zeros(n, dtype=np.int8)
    sums = np.zeros(num_progs, dtype=np.int16)
    
    # Precompute for each position which progressions it's in
    # and what the "contribution" would be
    for pos in range(n):
        affected = index_to_progs[pos]
        # Try +1
        best_sign = 1
        best_worst = float('inf')
        for s in (1, -1):
            # Compute what the max |sum| would be if we set signs[pos] = s
            # We need to compute sums for affected progressions
            # sums[affected] + s (since pos was 0, adding s gives the new contribution)
            new_sums = sums[affected] + s
            # For progressions not yet affected by pos, sums are unchanged
            # The max is max(max over affected of |new_sums|, max over unaffected of |sums|)
            # But we don't track the global max efficiently here
            # Simple approach: just compute max over all
            # This is O(num_progs) per position per sign, so O(n * num_progs) total
            # num_progs ~ n^2/2 = 32768, n=256, so ~8M ops, acceptable
            # But let's be smarter: track current max and only check if it could improve
            local_max = np.max(np.abs(new_sums))
            # Global max is max(local_max, current_global_max_for_unaffected)
            # Since unaffected sums are unchanged, their max is some value
            # Let's just compute the full max for simplicity in init
            all_sums = sums.copy()
            all_sums[affected] = new_sums
            w = np.max(np.abs(all_sums))
            if w < best_worst:
                best_worst = w
                best_sign = s
        signs[pos] = best_sign
        sums[affected] += best_sign
    
    current_max = np.max(np.abs(sums))
    best_max = current_max
    best_signs = signs.copy()
    
    # 3. Simulated annealing with multi-flip moves
    rng = random.Random(123)
    t_start = time.time()
    iterations = 0
    
    # Temperature schedule: start high, cool down
    T0 = 5.0
    T_end = 0.1
    # We'll use geometric cooling
    
    # Track the current max more efficiently using a count of |sum| values
    # |sum| ranges from 0 to n (max 256), but in practice much smaller
    # Use a histogram: hist[v] = number of progressions with |sum| = v
    max_possible = n  # max |sum| is n
    hist = np.zeros(max_possible + 1, dtype=np.int32)
    for s in sums:
        hist[abs(s)] += 1
    
    def get_max_from_hist():
        for v in range(max_possible, -1, -1):
            if hist[v] > 0:
                return v
        return 0
    
    current_max = get_max_from_hist()
    
    while time.time() - t_start < deadline - t0 - 2:
        iterations += 1
        
        # Temperature
        progress = min(1.0, iterations / 100000.0)
        T = T0 * (T_end / T0) ** progress
        
        # Choose move: single flip (80%) or multi-flip (20%)
        if rng.random() < 0.8:
            # Single flip: pick from worst progressions or random
            if rng.random() < 0.7:
                # Pick a random progression with high |sum|
                # Sample from top few
                worst_pid = np.argmax(np.abs(sums))
                worst_indices = progressions[worst_pid]
                flip_indices = [rng.choice(worst_indices)]
            else:
                flip_indices = [rng.randint(0, n - 1)]
        else:
            # Multi-flip: 2-3 random positions
            num_flips = rng.randint(2, 3)
            flip_indices = rng.sample(range(n), num_flips)
        
        # Apply flip(s)
        old_signs = {}
        for idx in flip_indices:
            old_signs[idx] = signs[idx]
        
        # Update sums
        for idx in flip_indices:
            old_sign = signs[idx]
            delta = -2 * old_sign
            signs[idx] = -old_sign
            affected = index_to_progs[idx]
            for p in affected:
                old_abs = abs(sums[p])
                sums[p] += delta
                new_abs = abs(sums[p])
                hist[old_abs] -= 1
                hist[new_abs] += 1
        
        new_max = get_max_from_hist()
        
        # Acceptance criterion
        accept = False
        if new_max <= current_max:
            accept = True
        else:
            # Simulated annealing: accept with probability exp(-delta/T)
            delta = new_max - current_max
            if delta > 0:
                prob = np.exp(-delta / max(T, 1e-6))
                if rng.random() < prob:
                    accept = True
        
        if accept:
            current_max = new_max
            if new_max < best_max:
                best_max = new_max
                best_signs = signs.copy()
        else:
            # Revert
            for idx in flip_indices:
                old_sign = old_signs[idx]
                delta = -2 * old_sign
                signs[idx] = old_sign
                affected = index_to_progs[idx]
                for p in affected:
                    old_abs = abs(sums[p])
                    sums[p] -= delta
                    new_abs = abs(sums[p])
                    hist[old_abs] -= 1
                    hist[new_abs] += 1
        
        # Periodic restart from best if stuck
        if iterations % 50000 == 0 and current_max > best_max:
            # Reset to best and restart with higher temperature
            signs[:] = best_signs
            # Recompute sums and hist
            sums[:] = 0
            for p_idx, indices in enumerate(progressions):
                sums[p_idx] = np.sum(signs[list(indices)])
            hist[:] = 0
            for s in sums:
                hist[abs(s)] += 1
            current_max = get_max_from_hist()
    
    return [int(x) for x in best_signs]
