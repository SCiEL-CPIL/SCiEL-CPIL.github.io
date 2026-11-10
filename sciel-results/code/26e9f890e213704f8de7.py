import time
import numpy as np
import random

def get_divisors(n):
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    return sorted([d for d in divisors if d > 1])

def construct():
    N = 51975
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    divisors = get_divisors(N)
    # Precompute indices for each divisor? 
    # No, arange is fast enough.
    
    best_uncovered = N
    best_classes = []
    
    # Run multiple iterations of Greedy + SA
    num_restarts = 5
    
    for restart in range(num_restarts):
        if time.time() > deadline - 20:
            break
            
        # Greedy Construction
        covered = np.zeros(N, dtype=np.uint8)
        cover_count = np.zeros(N, dtype=np.int16) # Use int16 to be safe, max 47
        classes = []
        available_moduli = set(divisors)
        
        # Greedy Step
        while available_moduli:
            if time.time() > deadline - 20:
                break
            
            best_gain = 0
            best_m = None
            best_r = None
            
            # To speed up, we can shuffle the order or just iterate
            # Iterating all is fine, 47 moduli
            for m in list(available_moduli):
                # Find best r for this m
                # We want to maximize sum(covered == 0) at indices r, r+m, ...
                # Vectorized:
                # For each r, count uncovered in arange(r, N, m)
                # This is a loop over r (0 to m-1).
                # Total work per m is N.
                # Total work per greedy step is |M| * N.
                # 47 * 52000 = 2.4M ops. Very fast.
                
                # Optimization: If m is large, N/m is small.
                # If m is small, N/m is large.
                
                max_cov = 0
                best_r_for_m = 0
                
                # We can check if adding this m can even improve.
                # Max possible new coverage is N/m.
                # If N/m <= best_gain, skip? No, best_gain is global.
                
                for r in range(m):
                    idx = np.arange(r, N, m)
                    # Count how many are currently uncovered
                    # covered[idx] == 0
                    cov = int(np.sum(covered[idx] == 0))
                    if cov > max_cov:
                        max_cov = cov
                        best_r_for_m = r
                
                if max_cov > best_gain:
                    best_gain = max_cov
                    best_m = m
                    best_r = best_r_for_m
            
            if best_gain <= 0:
                break
            
            # Add the best class
            m = best_m
            r = best_r
            idx = np.arange(r, N, m)
            covered[idx] = 1
            cover_count[idx] += 1
            classes.append([m, r])
            available_moduli.remove(m)
            
        # Current uncovered
        cur_uncovered = int(np.sum(covered == 0))
        
        if cur_uncovered < best_uncovered:
            best_uncovered = cur_uncovered
            best_classes = [c[:] for c in classes]
        
        # Local Search (Simulated Annealing)
        if time.time() > deadline - 10:
            continue
            
        # Prepare for SA
        # We need to be able to undo moves
        # State: classes list, cover_count, covered, uncovered_count
        
        current_uncovered = cur_uncovered
        current_classes = [c[:] for c in classes]
        current_cover_count = cover_count.copy()
        current_covered = covered.copy()
        
        # SA Parameters
        T = 1.0
        cooling = 0.99
        min_T = 0.01
        max_steps = 5000
        
        used_moduli = {c[0] for c in current_classes}
        unused_moduli = [m for m in divisors if m not in used_moduli]
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            if T < min_T:
                break
                
            # Pick a random move type
            # 1. Change residue of an existing class
            # 2. Swap modulus of an existing class with an unused one
            
            if len(current_classes) == 0:
                # Add a random class?
                if unused_moduli:
                    m = random.choice(unused_moduli)
                    r = random.randint(0, m-1)
                    idx = np.arange(r, N, m)
                    # Calculate delta
                    # This is complex, just add and check
                    # For simplicity, if empty, just add best greedy one
                    # But SA assumes non-empty mostly.
                    pass
                else:
                    break
            
            move_type = random.random()
            
            if move_type < 0.7:
                # Change residue
                if not current_classes:
                    continue
                i = random.randint(0, len(current_classes) - 1)
                m, old_r = current_classes[i]
                new_r = random.randint(0, m - 1)
                if new_r == old_r:
                    continue
                
                old_idx = np.arange(old_r, N, m)
                new_idx = np.arange(new_r, N, m)
                
                # Apply change
                current_cover_count[old_idx] -= 1
                current_cover_count[new_idx] += 1
                
                # Update covered and uncovered count
                # We need to check which indices changed status
                # Indices in old_idx that went from 1 to 0: became uncovered
                # Indices in new_idx that went from 0 to 1: became covered
                # But overlaps exist.
                
                # Efficient delta calculation:
                # Let's just recompute uncovered for the union of indices?
                # Or simpler: 
                # old_uncovered = current_uncovered
                # new_uncovered = int(np.sum(current_cover_count == 0))
                # This is O(N). 5000 * 52000 = 260M ops. Might be too slow.
                
                # Let's try to be smarter.
                # We know current_covered (bool array).
                # We update current_cover_count.
                # We can update current_covered in place?
                # current_covered[old_idx] = (current_cover_count[old_idx] > 0)
                # current_covered[new_idx] = (current_cover_count[new_idx] > 0)
                
                # But we need the delta.
                # delta = (new_uncovered - old_uncovered)
                
                # Let's check the specific indices involved.
                # I_old, I_new.
                # I_diff = I_old union I_new.
                # We can extract counts for I_diff before and after?
                # No, we already modified current_cover_count.
                
                # Let's revert if we can't compute delta fast.
                # Actually, np.sum is quite fast. Let's benchmark mentally.
                # 52k elements. ~0.01 ms?
                # 5000 steps * 0.01 ms = 50 ms.
                # This is fine!
                
                new_uncovered = int(np.sum(current_cover_count == 0))
                delta = new_uncovered - current_uncovered
                
                if delta < 0 or (delta == 0 and random.random() < 0.1) or (random.random() < np.exp(-delta / T)):
                    # Accept
                    current_uncovered = new_uncovered
                    current_classes[i][1] = new_r
                    # Update covered array? We only use cover_count for delta.
                    # We don't strictly need 'covered' bool array if we use cover_count for sum.
                else:
                    # Reject
                    current_cover_count[new_idx] -= 1
                    current_cover_count[old_idx] += 1
                    # current_uncovered remains same
            
            else:
                # Swap modulus
                if not current_classes or not unused_moduli:
                    continue
                
                i = random.randint(0, len(current_classes) - 1)
                old_m, old_r = current_classes[i]
                new_m = random.choice(unused_moduli)
                new_r = random.randint(0, new_m - 1)
                
                old_idx = np.arange(old_r, N, old_m)
                new_idx = np.arange(new_r, N, new_m)
                
                current_cover_count[old_idx] -= 1
                current_cover_count[new_idx] += 1
                
                new_uncovered = int(np.sum(current_cover_count == 0))
                delta = new_uncovered - current_uncovered
                
                if delta < 0 or (delta == 0 and random.random() < 0.1) or (random.random() < np.exp(-delta / T)):
                    # Accept
                    current_uncovered = new_uncovered
                    # Update classes
                    current_classes[i][0] = new_m
                    current_classes[i][1] = new_r
                    # Update used/unused moduli
                    used_moduli.remove(old_m)
                    used_moduli.add(new_m)
                    unused_moduli.remove(new_m)
                    unused_moduli.append(old_m)
                else:
                    # Reject
                    current_cover_count[new_idx] -= 1
                    current_cover_count[old_idx] += 1
            
            T *= cooling
            
            if current_uncovered < best_uncovered:
                best_uncovered = current_uncovered
                best_classes = [c[:] for c in current_classes]
        
        # End SA for this restart
        
    return best_classes
