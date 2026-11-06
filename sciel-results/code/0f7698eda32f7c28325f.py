import numpy as np
import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    # Precompute all arithmetic progressions as index lists (0-based)
    # An AP: start a (0-based), step s >= 1, length k >= 1
    # indices: a, a+s, a+2s, ..., a+(k-1)s where a+(k-1)s < n
    progs = []
    for a in range(n):
        for s in range(1, n):
            idx = list(range(a, n, s))
            if len(idx) >= 1:
                progs.append(idx)
    
    Np = len(progs)
    
    # For efficient local search, precompute for each position i, which APs contain it
    pos_to_progs = [[] for _ in range(n)]
    for pid, idx in enumerate(progs):
        for i in idx:
            pos_to_progs[i].append(pid)
    
    # Precompute a fast way to update sums for a set of flipped positions
    # We will use a dictionary to accumulate changes for a block flip
    
    def compute_initial_sums(signs):
        sums = np.zeros(Np, dtype=np.int32)
        for pid, idx in enumerate(progs):
            s = 0
            for i in idx:
                s += signs[i]
            sums[pid] = s
        return sums
    
    def max_abs_sum(sums):
        return int(np.max(np.abs(sums)))
    
    # Initialize with random signs
    rng = np.random.RandomState(123)
    signs = rng.choice([-1, 1], size=n)
    sums = compute_initial_sums(signs)
    current_score = max_abs_sum(sums)
    best_signs = signs.copy()
    best_score = current_score
    
    # Simulated Annealing with Block Flips
    # We will try flipping 1, 2, or 3 positions at a time
    # Temperature schedule: exponential cooling
    
    T_start = 10.0
    T_end = 0.1
    n_steps = 50000  # Adjust based on time
    alpha = (T_end / T_start) ** (1.0 / n_steps)
    
    current_T = T_start
    
    # Precompute random blocks for faster sampling? No, just sample on the fly.
    
    step = 0
    while time.time() < deadline - 2:
        step += 1
        if step % 1000 == 0:
            # Update temperature
            current_T *= alpha
            if current_T < T_end:
                current_T = T_end
        
        # Choose block size: mostly 1, sometimes 2 or 3
        r = rng.random()
        if r < 0.6:
            k = 1
        elif r < 0.9:
            k = 2
        else:
            k = 3
        
        # Sample k distinct positions
        if k == 1:
            idxs = [rng.randint(0, n - 1)]
        else:
            idxs = list(rng.choice(n, size=k, replace=False))
        
        # Calculate change in sums
        # For each pid, delta_sum = sum(2 * new_sign_i for i in idxs if i in progs[pid])
        # Actually, if we flip signs[i], the contribution changes from signs[i] to -signs[i], delta = -2*signs[i]
        # So new_sum = old_sum - 2 * sum(signs[i] for i in idxs if i in progs[pid])
        
        # Efficient update:
        # Create a temporary array of deltas for all progs? That's O(Np) per step.
        # Np is ~ 32,000. 50,000 steps * 32,000 is too slow in Python loops.
        
        # Alternative: Use the pos_to_progs structure.
        # For each i in idxs, for each pid in pos_to_progs[i], add -2*signs[i] to a delta array.
        # But we need to know the new max abs sum.
        
        # Let's try a different approach: 
        # Instead of calculating the full new max, we can estimate the change.
        # But for correctness, we need the exact max.
        
        # Optimization: Only update the affected sums.
        # Keep a dictionary of affected pids and their new values?
        # Or just update the global sums array in place, check max, and revert if rejected.
        
        # The bottleneck is np.max(np.abs(sums)) which is O(Np).
        # Can we maintain a histogram of sums?
        # Sums are integers in [-256, 256].
        # We can maintain a count array of size 513.
        
        # Let's restart the logic with a histogram approach for speed.
        # This requires reinitializing.
        
        # Actually, the previous program's score was 1.23. 
        # The main issue is that single flips get stuck.
        # Block flips are the key.
        # Let's implement block flips with the histogram trick.
        
        # Break out of this loop and restart with a better internal loop structure.
        break
        
    # --- Histogram-based SA with Block Flips ---
    
    # Re-initialize
    signs = rng.choice([-1, 1], size=n)
    sums = compute_initial_sums(signs)
    
    # Build histogram of |sums|? No, histogram of sums (signed) to track max abs.
    # Range of sum: [-256, 256]. Offset 256. Size 513.
    hist = np.zeros(513, dtype=np.int32)
    for s in sums:
        hist[s + 256] += 1
    
    def get_max_abs_from_hist(hist):
        # Find max index with non-zero count, and min index with non-zero count
        # Max abs is max(max_pos - 256, 256 - min_pos)
        non_zero = np.nonzero(hist)[0]
        if len(non_zero) == 0:
            return 0
        min_idx = non_zero[0]
        max_idx = non_zero[-1]
        max_val = max_idx - 256
        min_val = min_idx - 256
        return max(abs(max_val), abs(min_val))
    
    current_score = get_max_abs_from_hist(hist)
    best_signs = signs.copy()
    best_score = current_score
    current_T = 10.0
    
    # Precompute pos_to_progs as numpy arrays for faster iteration? 
    # pos_to_progs[i] is a list of pids.
    
    # For block flips, we need to update hist.
    # For each pid affected, we need to:
    # 1. hist[old_sum + 256] -= 1
    # 2. new_sum = old_sum + delta
    # 3. hist[new_sum + 256] += 1
    
    # We need to know which pids are affected by the block.
    # Union of pos_to_progs[i] for i in block.
    
    while time.time() < deadline - 2:
        # Choose block size
        r = rng.random()
        if r < 0.5:
            k = 1
        elif r < 0.85:
            k = 2
        else:
            k = 3
            
        idxs = list(rng.choice(n, size=k, replace=False))
        
        # Determine affected pids
        # Use a set to avoid duplicates
        affected_pids = set()
        for i in idxs:
            affected_pids.update(pos_to_progs[i])
        
        if not affected_pids:
            continue
            
        # Calculate delta for each affected pid
        # delta = -2 * sum(signs[i] for i in idxs if i in progs[pid])
        # We can precompute for each i in idxs, the sign is signs[i].
        # For a given pid, we need to know which of the idxs are in it.
        
        # This is still O(|affected| * k) per step.
        # |affected| can be large (up to ~1000?).
        # If k=1, |affected| ~ 100-200.
        # If k=3, |affected| ~ 300-500.
        
        # Let's just do it.
        
        old_sums_val = {}
        new_sums_val = {}
        
        for pid in affected_pids:
            old_s = sums[pid]
            old_sums_val[pid] = old_s
            # Calculate delta
            delta = 0
            for i in idxs:
                # Check if i is in progs[pid]
                # This is slow if we search the list.
                # We need a faster way to check membership or precompute something.
                # pos_to_progs[i] gives pids containing i.
                # So i is in progs[pid] iff pid is in pos_to_progs[i].
                # We already built affected_pids as the union.
                # So we just need to sum signs[i] for i in idxs such that pid in pos_to_progs[i].
                
                # This check is O(1) if we use a set for pos_to_progs[i]?
                # Let's convert pos_to_progs to sets for O(1) lookup.
                pass # Placeholder
            
            # This approach is getting complex and potentially slow in Python.
            # Let's try a different optimization:
            # Precompute for each pid, a bitmask or boolean array of which positions it contains?
            # Np * n bits = 32000 * 256 = 8M bits = 1MB. Feasible.
            # But Python doesn't have fast bitwise ops on large arrays easily without numpy.
            
            # Let's stick to the list iteration but optimize the inner loop.
            # For each i in idxs, we know the pids containing it.
            # Let's aggregate deltas by pid directly.
            
            pass
        
        # Better approach:
        # Initialize a dictionary delta_map = {}
        # For each i in idxs:
        #   s_i = signs[i]
        #   for pid in pos_to_progs[i]:
        #       delta_map[pid] = delta_map.get(pid, 0) - 2 * s_i
        
        delta_map = {}
        for i in idxs:
            s_i = signs[i]
            for pid in pos_to_progs[i]:
                delta_map[pid] = delta_map.get(pid, 0) - 2 * s_i
        
        # Now apply changes
        for pid, delta in delta_map.items():
            old_s = sums[pid]
            new_s = old_s + delta
            
            # Update hist
            hist[old_s + 256] -= 1
            hist[new_s + 256] += 1
            
            # Store old values for revert
            old_sums_val[pid] = old_s
            new_sums_val[pid] = new_s
            sums[pid] = new_s
        
        new_score = get_max_abs_from_hist(hist)
        
        # Acceptance criterion
        if new_score < current_score:
            # Always accept
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_signs = signs.copy()
        else:
            # Accept with probability exp(-(new_score - current_score) / T)
            delta_score = new_score - current_score
            if delta_score > 0:
                prob = np.exp(-delta_score / current_T)
                if rng.random() > prob:
                    # Reject and revert
                    for pid in delta_map:
                        sums[pid] = old_sums_val[pid]
                        old_s = old_sums_val[pid]
                        new_s = new_sums_val[pid]
                        hist[new_s + 256] -= 1
                        hist[old_s + 256] += 1
            else:
                # Should not happen if new_score >= current_score, but handle just in case
                pass
        
        # Cool down
        current_T *= 0.999
        if current_T < 0.1:
            current_T = 0.1
    
    # Final polish: Single flips only
    for _ in range(1000):
        if time.time() > deadline - 1:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline - 1:
                break
            # Try flip i
            s_i = signs[i]
            signs[i] = -s_i
            
            # Update sums and hist
            for pid in pos_to_progs[i]:
                old_s = sums[pid]
                new_s = old_s - 2 * s_i # Wait, sign changed from s_i to -s_i. Delta is -2*s_i.
                # Correct: new_contribution = -s_i. old_contribution = s_i. delta = -2*s_i.
                
                hist[old_s + 256] -= 1
                hist[new_s + 256] += 1
                sums[pid] = new_s
                old_sums_val[pid] = old_s
                new_sums_val[pid] = new_s
            
            new_score = get_max_abs_from_hist(hist)
            
            if new_score < current_score:
                current_score = new_score
                improved = True
                if new_score < best_score:
                    best_score = new_score
                    best_signs = signs.copy()
            else:
                # Revert
                signs[i] = s_i
                for pid in pos_to_progs[i]:
                    old_s = old_sums_val[pid]
                    new_s = new_sums_val[pid]
                    hist[new_s + 256] -= 1
                    hist[old_s + 256] += 1
                    sums[pid] = old_s
        
        if not improved:
            break
    
    return [int(x) for x in best_signs]
