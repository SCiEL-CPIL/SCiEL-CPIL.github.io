import numpy as np
import time
import random

def construct():
    N = 51975
    divisors = []
    for i in range(3, N + 1, 2):
        if N % i == 0:
            divisors.append(i)
    
    # Sort divisors
    divisors.sort()
    
    # Greedy Initialization
    classes = []
    covered = np.zeros(N, dtype=np.int8)
    score = N
    used_moduli = set()
    available = set(divisors)
    
    while available:
        best_m = None
        best_r = None
        best_gain = -1
        
        for m in available:
            min_nonzero = N
            best_r_for_m = 0
            
            # Loop over residues
            # Optimization: If m is large, we can sample? No, full scan is better for greedy.
            for r in range(m):
                nz = np.count_nonzero(covered[r::m])
                if nz < min_nonzero:
                    min_nonzero = nz
                    best_r_for_m = r
                    if min_nonzero == 0:
                        break
            
            total_len = (N - 1 - best_r_for_m) // m + 1
            gain = total_len - min_nonzero
            
            if gain > best_gain:
                best_gain = gain
                best_m = m
                best_r = best_r_for_m
        
        if best_gain <= 0:
            break
            
        classes.append([best_m, best_r])
        used_moduli.add(best_m)
        available.remove(best_m)
        
        covered[best_r::best_m] += 1
        score -= best_gain

    # Local Search
    start_time = time.time()
    deadline = start_time + 140
    
    best_score = score
    best_classes = [c[:] for c in classes]
    
    curr_classes = [c[:] for c in classes]
    curr_covered = covered.copy()
    curr_score = score
    curr_used_moduli = set(used_moduli)
    
    # Get initial uncovered
    uncovered_indices = np.flatnonzero(curr_covered == 0)
    
    # If no uncovered, we are done
    if len(uncovered_indices) == 0:
        return best_classes
        
    while time.time() < deadline:
        # Pick a random uncovered target
        # If the list is stale, we might pick a covered one. That's okay.
        if len(uncovered_indices) > 0:
            idx = random.randint(0, len(uncovered_indices) - 1)
            target = uncovered_indices[idx]
        else:
            target = random.randint(0, N - 1)
            
        move_applied = False
        
        # Try to improve by modifying an existing class
        if curr_classes:
            i = random.randint(0, len(curr_classes) - 1)
            m, r = curr_classes[i]
            r_new = target % m
            
            if r_new != r:
                old_slice = curr_covered[r::m]
                new_slice = curr_covered[r_new::m]
                
                lost = np.count_nonzero(old_slice == 1)
                gained = np.count_nonzero(new_slice == 0)
                
                delta = lost - gained
                
                if delta < 0:
                    curr_covered[r::m] -= 1
                    curr_covered[r_new::m] += 1
                    curr_classes[i][1] = r_new
                    curr_score += delta
                    move_applied = True
                    if curr_score < best_score:
                        best_score = curr_score
                        best_classes = [c[:] for c in curr_classes]
        
        # If no move applied, try adding a new class (if moduli available)
        if not move_applied:
            unused = [m for m in divisors if m not in curr_used_moduli]
            if unused:
                m_new = random.choice(unused)
                r_new = target % m_new
                
                new_slice = curr_covered[r_new::m_new]
                gained = np.count_nonzero(new_slice == 0)
                delta = -gained
                
                if delta < 0:
                    curr_covered[r_new::m_new] += 1
                    curr_classes.append([m_new, r_new])
                    curr_used_moduli.add(m_new)
                    curr_score += delta
                    move_applied = True
                    if curr_score < best_score:
                        best_score = curr_score
                        best_classes = [c[:] for c in curr_classes]
        
        # If no move applied, try removing a redundant class
        if not move_applied and curr_classes:
            i = random.randint(0, len(curr_classes) - 1)
            m, r = curr_classes[i]
            old_slice = curr_covered[r::m]
            lost = np.count_nonzero(old_slice == 1)
            
            if lost == 0:
                curr_covered[r::m] -= 1
                curr_used_moduli.remove(m)
                curr_classes.pop(i)
                move_applied = True
                
        # Update uncovered list periodically to prevent excessive staleness
        # Checking every 1000 iterations
        # But we don't have a counter here. 
        # We can just update if move_applied is True? 
        # No, updating is O(N). 
        # Let's update every 5000 steps roughly?
        # Since we don't track steps, let's just update if we want to be safe.
        # Actually, if we don't update, we might keep targeting the same few points.
        # But `random.randint` picks from the list. If the list is static, we pick from the same set.
        # If points get covered, we pick them, but move fails.
        # If points become uncovered, we don't pick them.
        # This is a bias. 
        # Let's update the list every 10000 iterations? 
        # We can use a time-based update or a counter.
        
        # Let's add a simple counter.
        if not hasattr(construct, 'steps'):
            construct.steps = 0
        construct.steps += 1
        
        if construct.steps % 10000 == 0:
            uncovered_indices = np.flatnonzero(curr_covered == 0)
            if len(uncovered_indices) == 0:
                break

    return best_classes
