"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def _is_sidon(subset):
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_size(A):
    n = len(A)
    best = 0
    # Check subsets in decreasing size order for early termination
    for size in range(n, best, -1):
        found = False
        # Generate combinations of this size
        from itertools import combinations
        for combo in combinations(range(n), size):
            sub = [A[i] for i in combo]
            if _is_sidon(sub):
                best = size
                found = True
                break
        if found:
            break
    return best

def _evaluate_set(A):
    return _max_sidon_size(A)

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    # Try several structured initializations that create sum collisions
    # Strategy: use arithmetic progressions and clustered sets
    candidates = []
    
    # 1. Dense arithmetic progression (many collisions)
    candidates.append(list(range(12)))
    
    # 2. Two interleaved progressions
    candidates.append([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    
    # 3. Clustered: two groups close together
    candidates.append(sorted([0, 1, 2, 3, 4, 5, 20, 21, 22, 23, 24, 25]))
    
    # 4. Three clusters
    candidates.append(sorted([0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23]))
    
    # 5. Spread out but with internal structure
    candidates.append(sorted([0, 1, 2, 10, 11, 12, 20, 21, 22, 30, 31, 32]))
    
    # 6. Geometric-like with small steps
    candidates.append(sorted([0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]))
    
    # 7. Random samples from small range
    random.seed(42)
    for seed in range(20):
        random.seed(seed)
        c = sorted(random.sample(range(30), 12))
        candidates.append(c)
    
    # 8. Random samples from full range
    for seed in range(20):
        random.seed(seed + 100)
        c = sorted(random.sample(range(129), 12))
        candidates.append(c)
    
    # Evaluate all candidates
    best_set = None
    best_val = 12  # worst case
    
    for c in candidates:
        if time.time() > deadline - 5:
            break
        val = _evaluate_set(c)
        if val < best_val:
            best_val = val
            best_set = c[:]
        if best_val <= 3:
            break
    
    if best_set is None:
        best_set = list(range(12))
        best_val = _evaluate_set(best_set)
    
    # Local search refinement
    current_set = best_set[:]
    current_val = best_val
    random.seed(12345)
    
    while time.time() < deadline - 10:
        # Pick a random element to replace
        idx = random.randint(0, 11)
        old_elem = current_set[idx]
        
        # Try replacing with a random element not in the set
        new_elem = random.randint(0, 128)
        if new_elem in current_set:
            continue
        
        current_set[idx] = new_elem
        current_set.sort()
        
        new_val = _evaluate_set(current_set)
        
        if new_val < current_val:
            current_val = new_val
            if current_val < best_val:
                best_val = current_val
                best_set = current_set[:]
            if best_val <= 3:
                break
        elif new_val == current_val:
            # Accept with probability to explore
            if random.random() < 0.3:
                current_val = new_val
            else:
                current_set[idx] = old_elem
                current_set.sort()
        else:
            # Revert
            current_set[idx] = old_elem
            current_set.sort()
        
        # Occasionally try a swap instead
        if random.random() < 0.2:
            idx1, idx2 = random.sample(range(12), 2)
            current_set[idx1], current_set[idx2] = current_set[idx2], current_set[idx1]
            current_set.sort()
            new_val = _evaluate_set(current_set)
            if new_val <= current_val:
                current_val = new_val
                if current_val < best_val:
                    best_val = current_val
                    best_set = current_set[:]
                if best_val <= 3:
                    break
            else:
                current_set[idx1], current_set[idx2] = current_set[idx2], current_set[idx1]
                current_set.sort()
        
        # Occasionally restart from a new random point
        if random.random() < 0.05:
            current_set = sorted(random.sample(range(129), 12))
            current_val = _evaluate_set(current_set)
            if current_val < best_val:
                best_val = current_val
                best_set = current_set[:]
            if best_val <= 3:
                break
    
    return best_set
# EVOLVE-BLOCK-END
