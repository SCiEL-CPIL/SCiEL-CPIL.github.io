import time
import numpy as np
from itertools import combinations

def score_set(U):
    """Compute the score for a given set U."""
    U = list(set(U))
    if 0 not in U:
        return 0.0
    n = len(U)
    if n < 2:
        return 0.0
    
    # Compute sumset U+U
    sumset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
    
    # Compute difference set U-U
    diffset = set()
    for a in U:
        for b in U:
            diffset.add(a - b)
    
    max_u = max(U)
    if max_u <= 0:
        return 0.0
    
    import math
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return 0.0
    
    numerator = math.log(ratio)
    denominator = math.log(2 * max_u + 1)
    
    if denominator <= 0:
        return 0.0
    
    return 1.0 + numerator / denominator

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    best_set = [0, 1, 3]
    best_score = score_set(best_set)
    
    def check_time():
        return time.time() > deadline
    
    # Try various geometric and structured sets
    candidates = []
    
    # Geometric sequences with different bases
    for base in range(2, 12):
        for n_elems in range(3, 15):
            s = [0]
            val = 1
            for i in range(n_elems - 1):
                s.append(val)
                val *= base
            if max(s) < 10**6:  # Keep max reasonable
                candidates.append(s)
    
    # Mixed: start geometric, then add some smaller elements
    for base in range(2, 8):
        for n_geom in range(3, 10):
            s = [0]
            val = 1
            for i in range(n_geom - 1):
                s.append(val)
                val *= base
            # Try adding some elements in gaps
            for extra in range(1, min(val // 2, 20)):
                s2 = s + [extra]
                if len(set(s2)) == len(s2):
                    candidates.append(s2)
    
    # Arithmetic-like with growing gaps
    for start_gap in range(1, 6):
        for growth in [1.5, 2.0, 2.5, 3.0]:
            s = [0]
            gap = start_gap
            for i in range(8):
                s.append(s[-1] + gap)
                gap = int(gap * growth) + 1
            if max(s) < 10**6:
                candidates.append(s)
    
    # Evaluate all candidates
    for cand in candidates:
        if check_time():
            break
        sc = score_set(cand)
        if sc > best_score:
            best_score = sc
            best_set = list(cand)
    
    # Local search on best set
    if not check_time():
        current = list(best_set)
        current_score = best_score
        max_val = max(current)
        
        for iteration in range(500):
            if check_time():
                break
            
            improved = False
            # Try adding a new element
            for new_val in range(1, max_val + 50):
                if new_val in current:
                    continue
                candidate = sorted(current + [new_val])
                sc = score_set(candidate)
                if sc > current_score + 1e-10:
                    current = candidate
                    current_score = sc
                    max_val = max(current)
                    improved = True
                    break
            
            # Try removing an element (except 0)
            if not improved:
                for i in range(1, len(current)):
                    candidate = current[:i] + current[i+1:]
                    sc = score_set(candidate)
                    if sc > current_score + 1e-10:
                        current = candidate
                        current_score = sc
                        max_val = max(current)
                        improved = True
                        break
            
            # Try replacing an element
            if not improved:
                for i in range(1, len(current)):
                    for new_val in range(1, max_val + 30):
                        if new_val in current:
                            continue
                        candidate = current[:i] + [new_val] + current[i+1:]
                        candidate = sorted(set(candidate))
                        if 0 not in candidate:
                            continue
                        sc = score_set(candidate)
                        if sc > current_score + 1e-10:
                            current = candidate
                            current_score = sc
                            max_val = max(current)
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                # Try perturbation: add random element
                import random
                random.seed(iteration)
                for _ in range(5):
                    new_val = random.randint(1, max_val + 100)
                    if new_val not in current:
                        candidate = sorted(current + [new_val])
                        sc = score_set(candidate)
                        if sc > current_score + 1e-10:
                            current = candidate
                            current_score = sc
                            max_val = max(current)
                            improved = True
                            break
            
            if not improved and iteration > 50:
                break
        
        if current_score > best_score:
            best_score = current_score
            best_set = current
    
    # Final: ensure 0 is in the set and return sorted
    best_set = sorted(set(best_set))
    if 0 not in best_set:
        best_set = [0] + best_set
    
    return best_set

# EVOLVE-BLOCK-END
