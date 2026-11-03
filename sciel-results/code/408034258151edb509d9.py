import math
import time
import random
import numpy as np
import itertools

# Known optimal Golomb rulers (minimal diameter for given number of marks)
# These maximize |U-U|/|U+U| for a given max(U)
_GOLOMB_RULERS = {
    2: [0, 1],
    3: [0, 1, 3],
    4: [0, 1, 4, 6],
    5: [0, 1, 4, 9, 11],
    6: [0, 1, 4, 10, 12, 17],
    7: [0, 1, 4, 10, 18, 23, 31],
    8: [0, 1, 4, 10, 18, 23, 29, 36],
    9: [0, 1, 4, 10, 18, 23, 29, 38, 44],
    10: [0, 1, 4, 10, 18, 23, 29, 38, 44, 52],
    11: [0, 1, 4, 10, 18, 23, 29, 38, 44, 52, 60],
    12: [0, 1, 4, 10, 18, 23, 29, 38, 44, 52, 60, 68],
    13: [0, 1, 4, 10, 18, 23, 29, 38, 44, 52, 60, 68, 76],
    14: [0, 1, 4, 10, 18, 23, 29, 38, 44, 52, 60, 68, 76, 84],
    15: [0, 1, 4, 10, 18, 23, 29, 38, 44, 52, 60, 68, 76, 84, 92],
}

def _score(U):
    """Score a set U. Returns 0 for invalid sets."""
    if not U or 0 not in U:
        return 0
    U = sorted(set(U))
    if not U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    # Use NumPy for faster computation
    U_arr = np.array(U, dtype=np.int64)
    
    # Compute all pairwise sums
    sums_arr = U_arr[:, None] + U_arr[None, :]
    unique_sums = np.unique(sums_arr)
    
    # Compute all pairwise differences
    diffs_arr = U_arr[:, None] - U_arr[None, :]
    unique_diffs = np.unique(diffs_arr)
    
    ratio = len(unique_diffs) / len(unique_sums)
    if ratio <= 0:
        return 0
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 1.0
    return 1 + math.log(ratio) / denom

def _score_fast(U):
    """Faster scoring for small sets using Python sets (avoids NumPy overhead)."""
    if not U or 0 not in U:
        return 0
    U = sorted(set(U))
    if not U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    sumset = set()
    diffset = set()
    n = len(U)
    for i in range(n):
        for j in range(n):
            sumset.add(U[i] + U[j])
            diffset.add(U[i] - U[j])
    
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return 0
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 1.0
    return 1 + math.log(ratio) / denom

def _local_search(U, max_iterations=500, deadline=None):
    """Local search to refine a set with targeted moves."""
    U = sorted(set(U))
    best_score = _score(U)
    best_U = U[:]
    current_U = U[:]
    current_score = best_score
    no_improve_count = 0
    
    for i in range(max_iterations):
        if deadline and time.time() > deadline:
            break
        
        max_u = max(current_U) if current_U else 0
        improved = False
        
        # Move 1: Try adding a random element
        if random.random() < 0.3:
            if random.random() < 0.5:
                new_val = random.randint(1, max(max_u * 2, 10))
            else:
                new_val = random.randint(1, max_u + 15)
            
            candidate = sorted(set(current_U + [new_val]))
            s = _score(candidate)
            if s > current_score:
                current_U = candidate
                current_score = s
                improved = True
                if s > best_score:
                    best_score = s
                    best_U = candidate[:]
        
        # Move 2: Try removing an element (not 0)
        if len(current_U) > 1 and random.random() < 0.2:
            idx = random.randint(1, len(current_U) - 1)
            candidate = sorted(set(current_U[:idx] + current_U[idx+1:]))
            s = _score(candidate)
            if s > current_score:
                current_U = candidate
                current_score = s
                improved = True
                if s > best_score:
                    best_score = s
                    best_U = candidate[:]
        
        # Move 3: Try shifting an element by small amount
        if len(current_U) > 1 and random.random() < 0.25:
            idx = random.randint(1, len(current_U) - 1)
            shift = random.choice([-3, -2, -1, 1, 2, 3])
            new_val = current_U[idx] + shift
            if new_val >= 0:
                candidate = sorted(set([current_U[j] if j != idx else new_val for j in range(len(current_U))]))
                if len(candidate) == len(current_U):  # No collisions
                    s = _score(candidate)
                    if s > current_score:
                        current_U = candidate
                        current_score = s
                        improved = True
                        if s > best_score:
                            best_score = s
                            best_U = candidate[:]
        
        # Move 4: Try replacing with value near existing gaps
        if len(current_U) > 1 and random.random() < 0.15:
            idx = random.randint(1, len(current_U) - 1)
            # Try values in a window around current position
            base = current_U[idx]
            offsets = random.sample([-10, -8, -6, -4, -2, 2, 4, 6, 8, 10, 15, 20], 3)
            for off in offsets:
                new_val = base + off
                if new_val >= 0 and new_val not in current_U:
                    candidate = sorted(set([current_U[j] if j != idx else new_val for j in range(len(current_U))]))
                    s = _score(candidate)
                    if s > current_score:
                        current_U = candidate
                        current_score = s
                        improved = True
                        if s > best_score:
                            best_score = s
                            best_U = candidate[:]
                        break
        
        # Move 5: Try a larger perturbation
        if len(current_U) > 1 and random.random() < 0.1:
            idx = random.randint(1, len(current_U) - 1)
            new_val = random.randint(1, max(max_u * 3, 20))
            candidate = sorted(set([current_U[j] if j != idx else new_val for j in range(len(current_U))]))
            if len(candidate) == len(current_U):
                s = _score(candidate)
                if s > current_score:
                    current_U = candidate
                    current_score = s
                    improved = True
                    if s > best_score:
                        best_score = s
                        best_U = candidate[:]
        
        if not improved:
            no_improve_count += 1
        else:
            no_improve_count = 0
        
        # If stuck for too long, try a bigger jump
        if no_improve_count > 20:
            idx = random.randint(1, len(current_U) - 1)
            new_val = random.randint(1, max(max_u * 4, 30))
            candidate = sorted(set([current_U[j] if j != idx else new_val for j in range(len(current_U))]))
            if len(candidate) == len(current_U):
                s = _score(candidate)
                if s > current_score:
                    current_U = candidate
                    current_score = s
                    if s > best_score:
                        best_score = s
                        best_U = candidate[:]
                    no_improve_count = 0
                        
    return best_U

def _exhaustive_search_small(max_val, n_min, n_max, deadline):
    """Exhaustive search for small sets."""
    best_score = 0
    best_set = [0, 1]
    
    for n in range(n_min, n_max + 1):
        if time.time() > deadline:
            break
        # We need to choose n-1 elements from 1..max_val
        for combo in itertools.combinations(range(1, max_val + 1), n - 1):
            if time.time() > deadline:
                break
            U = (0,) + combo
            s = _score_fast(list(U))
            if s > best_score:
                best_score = s
                best_set = list(U)
    
    return best_set, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 20s safety margin
    
    # Known good starting points: Golomb rulers + other candidates
    candidates = list(_GOLOMB_RULERS.values())
    # Additional candidates that may have better ratio
    candidates.extend([
        [0, 1, 2, 4, 8],
        [0, 1, 3, 7, 12],
        [0, 1, 2, 5, 11],
        [0, 1, 3, 7, 14],
        [0, 1, 3, 8, 13],
        [0, 2, 3, 7, 12],
        [0, 1, 4, 11, 13],
        [0, 1, 4, 9, 13],
        [0, 1, 2, 4, 8],
    ])
    
    best_U = [0, 1]
    best_score = 0
    
    # Evaluate all candidates
    for c in candidates:
        s = _score(c)
        if s > best_score:
            best_score = s
            best_U = c[:]
    
    # Phase 1: Exhaustive search for small sets
    # Try n=3,4,5 with max_val up to 30
    for max_val in [15, 20, 25, 30]:
        if time.time() > deadline:
            break
        n_max = 5 if max_val <= 25 else 4
        found_set, found_score = _exhaustive_search_small(max_val, 3, n_max, deadline)
        if found_score > best_score:
            best_score = found_score
            best_U = found_set[:]
    
    # Phase 2: Local search refinement on best candidate
    if time.time() < deadline - 15:
        refined = _local_search(best_U, max_iterations=2000, deadline=deadline - 10)
        s = _score(refined)
        if s > best_score:
            best_score = s
            best_U = refined
    
    # Phase 3: Try local search on Golomb rulers of various sizes
    random.seed(42)
    for n in range(3, min(16, len(_GOLOMB_RULERS) + 1)):
        if time.time() > deadline - 10:
            break
        if n in _GOLOMB_RULERS:
            gr = _GOLOMB_RULERS[n]
            s = _score(gr)
            if s > best_score:
                best_score = s
                best_U = gr[:]
            
            # Refine Golomb ruler with local search
            if time.time() < deadline - 10:
                refined = _local_search(gr, max_iterations=500, deadline=deadline - 5)
                s_r = _score(refined)
                if s_r > best_score:
                    best_score = s_r
                    best_U = refined
    
    # Phase 4: Try random restarts with local search
    for _ in range(15):
        if time.time() > deadline - 5:
            break
        
        n = random.randint(3, 10)
        max_val = random.randint(15, 80)
        
        # Generate a random set with 0 and n-1 other elements
        other_elements = random.sample(range(1, max_val), min(n - 1, max_val - 1))
        U = sorted([0] + other_elements)
        
        s = _score(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
        
        # Refine with local search
        if time.time() < deadline - 5:
            refined = _local_search(U, max_iterations=400, deadline=deadline - 5)
            s_r = _score(refined)
            if s_r > best_score:
                best_score = s_r
                best_U = refined
    
    # Phase 5: Final polish - try adding/removing elements to best set
    if time.time() < deadline - 2:
        current_set = set(best_U)
        max_current = max(best_U)
        
        # Try adding elements in a wider range
        for new_elem in range(1, min(max_current + 30, 150)):
            if time.time() > deadline:
                break
            if new_elem in current_set:
                continue
            new_set = sorted(current_set | {new_elem})
            s = _score(new_set)
            if s > best_score:
                best_score = s
                best_U = new_set
                current_set = set(new_set)
                max_current = max(best_U)
        
        # Try removing elements
        if len(best_U) > 2:
            for elem in list(best_U):
                if time.time() > deadline:
                    break
                if elem == 0:
                    continue
                new_set = sorted(set(best_U) - {elem})
                if len(new_set) < 2:
                    continue
                s = _score(new_set)
                if s > best_score:
                    best_score = s
                    best_U = new_set
        
        # Try small shifts on each element
        if len(best_U) > 2 and time.time() < deadline - 1:
            for idx in range(1, len(best_U)):
                if time.time() > deadline:
                    break
                for shift in [-2, -1, 1, 2]:
                    new_val = best_U[idx] + shift
                    if new_val < 0 or new_val in best_U:
                        continue
                    new_set = sorted(set([best_U[j] if j != idx else new_val for j in range(len(best_U))]))
                    if len(new_set) != len(best_U):
                        continue
                    s = _score(new_set)
                    if s > best_score:
                        best_score = s
                        best_U = new_set
    
    return best_U
