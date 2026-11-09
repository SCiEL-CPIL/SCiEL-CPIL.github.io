import time
import random
import numpy as np

def construct():
    n = 256
    t0 = time.time()
    deadline = t0 + 85  # 5s safety margin before 90s kill

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
    
    # 2. Initialize signs using Thue-Morse sequence
    signs = np.array([1 if (bin(i).count('1') % 2 == 0) else -1 for i in range(n)], dtype=np.int8)
    
    # Precompute sums for all progressions
    sums = np.zeros(num_progs, dtype=np.int16)
    for p_idx, indices in enumerate(progressions):
        s = np.sum(signs[list(indices)])
        sums[p_idx] = s
        
    current_max = int(np.max(np.abs(sums)))
    best_max = current_max
    best_signs = signs.copy()
    
    # Precompute affected indices as numpy arrays
    affected_pids_arrays = [np.array(lst, dtype=np.int32) for lst in index_to_progs]
    
    # 3. CSP-style iterative bound tightening with local search
    # Strategy: Try to achieve a lower max discrepancy by:
    # a) Identifying progressions that violate the target bound
    # b) Flipping signs to reduce violations
    # c) If stuck, use simulated annealing to escape local minima
    
    rng = random.Random(42)
    
    # Target: try to get below current best
    target_max = best_max - 1
    
    # Track which progressions are "active" (exceed target)
    active_progs = np.where(np.abs(sums) > target_max)[0]
    
    # Local search with CSP-inspired pruning
    stuck_count = 0
    max_stuck = 300
    temperature = 5.0  # Initial "temperature" for acceptance
    
    while time.time() - t0 < deadline:
        # Check if we've achieved the target
        if current_max <= target_max:
            # Try to tighten further
            target_max -= 1
            active_progs = np.where(np.abs(sums) > target_max)[0]
            stuck_count = 0
            temperature = max(temperature * 0.95, 0.1)
            continue
        
        if len(active_progs) == 0:
            # No violations - we've achieved target_max
            best_max = min(best_max, current_max)
            best_signs = signs.copy()
            target_max -= 1
            active_progs = np.where(np.abs(sums) > target_max)[0]
            stuck_count = 0
            continue
        
        # Pick a random active progression to fix
        prog_idx = rng.choice(active_progs)
        prog_indices = progressions[prog_idx]
        
        # Try flipping a random index in this progression
        flip_idx = rng.choice(prog_indices)
        
        old_sign = signs[flip_idx]
        delta = -2 * old_sign
        
        # Identify affected progressions
        affected_pids = affected_pids_arrays[flip_idx]
        
        # Apply flip
        signs[flip_idx] = -old_sign
        sums[affected_pids] += delta
        
        # Calculate new max and check for improvements
        new_max = int(np.max(np.abs(sums)))
        
        # Check if any progression now exceeds the old best
        violates_best = new_max > best_max
        
        if new_max < current_max:
            # Improvement
            current_max = new_max
            stuck_count = 0
            if new_max < best_max:
                best_max = new_max
                best_signs = signs.copy()
                target_max = min(target_max, new_max - 1)
                active_progs = np.where(np.abs(sums) > target_max)[0]
        elif new_max == current_max and not violates_best:
            # Neutral move
            stuck_count += 1
        else:
            # Worse - revert unless we're in annealing mode
            signs[flip_idx] = old_sign
            sums[affected_pids] -= delta
            stuck_count += 1
            
            # Escape local minima with random kick
            if stuck_count > max_stuck:
                stuck_count = 0
                max_stuck = int(max_stuck * 1.2)
                # Do multiple random flips
                for _ in range(5):
                    kick_idx = rng.randint(0, n - 1)
                    kick_old_sign = signs[kick_idx]
                    kick_delta = -2 * kick_old_sign
                    kick_affected = affected_pids_arrays[kick_idx]
                    
                    signs[kick_idx] = -kick_old_sign
                    sums[kick_affected] += kick_delta
                    
                    current_max = int(np.max(np.abs(sums)))
                    
                    if current_max < best_max:
                        best_max = current_max
                        best_signs = signs.copy()
                        target_max = min(target_max, best_max - 1)
                        active_progs = np.where(np.abs(sums) > target_max)[0]
        
        # Update active progressions periodically
        if stuck_count % 50 == 0:
            active_progs = np.where(np.abs(sums) > target_max)[0]
    
    return [int(x) for x in best_signs]
