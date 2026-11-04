import math
import time
import random
import numpy as np
import itertools
from functools import lru_cache

_score_cache = {}

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
    """Faster scoring for small sets using Python sets with memoization."""
    if not U or 0 not in U:
        return 0
    U = sorted(set(U))
    if not U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    # Use tuple as cache key
    key = tuple(U)
    if key in _score_cache:
        return _score_cache[key]
    
    sumset = set()
    diffset = set()
    n = len(U)
    for i in range(n):
        for j in range(n):
            sumset.add(U[i] + U[j])
            diffset.add(U[i] - U[j])
    
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        result = 0
    else:
        denom = math.log(2 * max_u + 1)
        if denom == 0:
            result = 1.0
        else:
            result = 1 + math.log(ratio) / denom
    
    _score_cache[key] = result
    return result

def _marginal_ratio_gain(U, new_elem):
    """Compute the marginal gain in |U-U|/|U+U| ratio when adding new_elem to U."""
    U_set = set(U)
    if new_elem in U_set:
        return -1
    U_new = sorted(U_set | {new_elem})
    
    # Compute sum and diff sets for current U
    sumset_old = set()
    diffset_old = set()
    n = len(U)
    for i in range(n):
        for j in range(n):
            sumset_old.add(U[i] + U[j])
            diffset_old.add(U[i] - U[j])
    
    # Compute sum and diff sets for U + {new_elem}
    sumset_new = set(sumset_old)
    diffset_new = set(diffset_old)
    for u in U:
        sumset_new.add(u + new_elem)
        sumset_new.add(new_elem + u)
        diffset_new.add(u - new_elem)
        diffset_new.add(new_elem - u)
    sumset_new.add(2 * new_elem)
    diffset_new.add(0)
    
    ratio_old = len(diffset_old) / len(sumset_old)
    ratio_new = len(diffset_new) / len(sumset_new)
    
    return ratio_new - ratio_old

def _greedy_construct(max_val, target_size, deadline):
    """Greedy construction: add elements one by one to maximize marginal ratio gain."""
    U = [0]
    current_score = _score_fast(U)
    
    for _ in range(target_size - 1):
        if time.time() > deadline:
            break
        
        best_elem = None
        best_gain = -float('inf')
        best_new_score = 0
        
        # Candidate elements to try
        max_current = max(U)
        candidates = set()
        for d in range(1, max_val + 1):
            if d not in U:
                candidates.add(d)
        
        # Prioritize candidates near existing elements and at strategic positions
        priority_candidates = []
        for u in U:
            for delta in range(-5, 6):
                c = u + delta
                if 1 <= c <= max_val and c not in U:
                    priority_candidates.append(c)
        # Add some spread-out candidates
        for c in range(1, max_val + 1, max(1, max_val // 20)):
            if c not in U:
                priority_candidates.append(c)
        
        # Evaluate a subset of candidates (not all, for speed)
        eval_candidates = list(set(priority_candidates))
        if len(eval_candidates) > 100:
            random.shuffle(eval_candidates)
            eval_candidates = eval_candidates[:100]
        
        for elem in eval_candidates:
            if time.time() > deadline:
                break
            U_new = sorted(U + [elem])
            s = _score_fast(U_new)
            if s > best_new_score:
                best_new_score = s
                best_elem = elem
        
        if best_elem is not None and best_new_score > current_score:
            U.append(best_elem)
            U.sort()
            current_score = best_new_score
        else:
            break
    
    return U, current_score

def _local_search(U, max_iterations=500, deadline=None):
    """Local search to refine a set with structured perturbations."""
    U = sorted(set(U))
    best_score = _score(U)
    best_U = U[:]
    current_U = U[:]
    current_score = best_score
    
    for i in range(max_iterations):
        if deadline and time.time() > deadline:
            break
        
        # Try adding a random element
        max_u = max(current_U) if current_U else 0
        if random.random() < 0.5:
            new_val = random.randint(1, max(max_u * 2, 10))
        else:
            new_val = random.randint(1, max_u + 10)
        
        candidate = sorted(set(current_U + [new_val]))
        s = _score(candidate)
        if s > current_score:
            current_U = candidate
            current_score = s
            if s > best_score:
                best_score = s
                best_U = candidate[:]
        
        # Try removing an element (not 0)
        if len(current_U) > 1:
            idx = random.randint(1, len(current_U) - 1)
            candidate = sorted(set(current_U[:idx] + current_U[idx+1:]))
            s = _score(candidate)
            if s > current_score:
                current_U = candidate
                current_score = s
                if s > best_score:
                    best_score = s
                    best_U = candidate[:]
        
        # Try shifting an element
        if len(current_U) > 1:
            idx = random.randint(1, len(current_U) - 1)
            shift = random.randint(-5, 5)
            if shift == 0:
                continue
            new_val = current_U[idx] + shift
            if new_val >= 0:
                candidate = sorted(set([current_U[j] if j != idx else new_val for j in range(len(current_U))]))
                if len(candidate) >= len(current_U) - 1:  # Allow at most one collision
                    s = _score(candidate)
                    if s > current_score:
                        current_U = candidate
                        current_score = s
                        if s > best_score:
                            best_score = s
                            best_U = candidate[:]
        
        # Try a larger perturbation
        if len(current_U) > 1 and random.random() < 0.3:
            idx = random.randint(1, len(current_U) - 1)
            new_val = random.randint(1, max(max_u * 3, 20))
            candidate = sorted(set([current_U[j] if j != idx else new_val for j in range(len(current_U))]))
            if len(candidate) >= len(current_U) - 1:
                s = _score(candidate)
                if s > current_score:
                    current_U = candidate
                    current_score = s
                    if s > best_score:
                        best_score = s
                        best_U = candidate[:]
        
        # Structured perturbation: try to maintain Golomb-like properties
        if len(current_U) > 2 and random.random() < 0.2:
            # Try moving an element to a position that preserves difference uniqueness
            idx = random.randint(1, len(current_U) - 1)
            # Try positions near existing elements
            nearby_positions = []
            for j in range(len(current_U)):
                if j != idx:
                    for delta in range(-3, 4):
                        pos = current_U[j] + delta
                        if pos >= 0 and pos not in current_U:
                            nearby_positions.append(pos)
            
            if nearby_positions:
                new_val = random.choice(nearby_positions)
                candidate = sorted(set([current_U[j] if j != idx else new_val for j in range(len(current_U))]))
                if len(candidate) >= len(current_U) - 1:
                    s = _score(candidate)
                    if s > current_score:
                        current_U = candidate
                        current_score = s
                        if s > best_score:
                            best_score = s
                            best_U = candidate[:]
                        
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

def _parametric_candidates():
    """Generate parametric/geometric candidate sets."""
    candidates = []
    
    # Golomb rulers (maximize |U-U| for given diameter)
    golomb_rulers = [
        [0, 1, 4],
        [0, 1, 4, 9],
        [0, 1, 4, 10, 18],
        [0, 1, 4, 10, 18, 23],
        [0, 1, 4, 10, 18, 23, 31],
        [0, 1, 4, 10, 18, 23, 31, 39],
        [0, 1, 4, 10, 18, 23, 31, 39, 47],
        [0, 1, 4, 10, 18, 23, 31, 39, 47, 55],
        [0, 1, 4, 10, 18, 23, 31, 39, 47, 55, 62],
    ]
    candidates.extend(golomb_rulers)
    
    # Geometric progressions
    for base in [2, 3, 4, 5]:
        for start in [1, 2]:
            seq = [0]
            val = start
            while val < 200:
                seq.append(val)
                val *= base
            if len(seq) >= 2:
                candidates.append(seq[:8])  # Limit to 8 elements
    
    # Arithmetic progressions with gaps
    for start in [1, 2, 3]:
        for step in [2, 3, 4, 5]:
            seq = [0]
            val = start
            while val < 100 and len(seq) < 8:
                seq.append(val)
                val += step
            candidates.append(seq)
    
    # Mixed: some geometric, some arithmetic
    for base in [2, 3]:
        for offset in [0, 1, 2]:
            seq = [0]
            for i in range(1, 7):
                val = int(base ** i) + offset
                if val not in seq:
                    seq.append(val)
            candidates.append(seq)
    
    # Known good candidates from literature/exploration
    extra_candidates = [
        [0, 1, 2, 4, 8],
        [0, 1, 3],
        [0, 1, 4, 9, 11],
        [0, 1, 3, 7, 12],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 2, 5, 11],
        [0, 1, 3, 7, 14],
        [0, 1, 3, 8, 13],
        [0, 2, 3, 7, 12],
        [0, 1, 4, 11, 13],
        [0, 1, 4, 9, 13],
        [0, 1, 2, 4, 8],
        [0, 1, 4, 10, 18, 23, 31, 39],
        [0, 1, 4, 10, 18, 23, 31],
        [0, 1, 4, 10, 18, 23],
        [0, 1, 4, 10, 18],
        [0, 1, 4, 10],
        [0, 1, 4],
        [0, 2, 3, 7, 12, 17],
        [0, 1, 3, 7, 12, 17],
        [0, 1, 3, 8, 13, 18],
        [0, 1, 4, 9, 14, 19],
        # Additional candidates targeting high |U-U|/|U+U| ratio
        [0, 1, 3, 4, 9, 11],
        [0, 1, 4, 6, 9, 11],
        [0, 1, 3, 7, 8, 12],
        [0, 1, 5, 6, 10, 12],
        [0, 2, 3, 5, 8, 11],
        [0, 1, 4, 5, 9, 12],
        [0, 1, 3, 5, 8, 11],
        [0, 2, 5, 6, 9, 12],
    ]
    candidates.extend(extra_candidates)
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 40s safety margin for 150s limit
    
    # Generate parametric candidates
    candidates = _parametric_candidates()
    
    best_U = [0, 1]
    best_score = 0
    
    # Evaluate candidates
    for c in candidates:
        s = _score_fast(c)
        if s > best_score:
            best_score = s
            best_U = c[:]
    
    # Phase 1: Exhaustive search for small sets
    for max_val in [20, 30, 40]:
        if time.time() > deadline:
            break
        n_max = 5 if max_val <= 30 else 4
        found_set, found_score = _exhaustive_search_small(max_val, 3, n_max, deadline)
        if found_score > best_score:
            best_score = found_score
            best_U = found_set[:]
    
    # Phase 2: Greedy construction with different parameters
    for max_val in [30, 50, 80]:
        if time.time() > deadline - 25:
            break
        for target_size in [4, 5, 6, 7, 8]:
            if time.time() > deadline - 20:
                break
            greedy_set, greedy_score = _greedy_construct(max_val, target_size, deadline - 15)
            if greedy_score > best_score:
                best_score = greedy_score
                best_U = greedy_set[:]
    
    # Phase 3: Local search refinement with more iterations
    if time.time() < deadline - 20:
        refined = _local_search(best_U, max_iterations=2000, deadline=deadline)
        s = _score(refined)
        if s > best_score:
            best_score = s
            best_U = refined
    
    # Phase 4: Try random restarts with local search
    random.seed(42)
    for _ in range(40):
        if time.time() > deadline - 15:
            break
        
        n = random.randint(3, 9)
        max_val = random.randint(15, 80)
        
        # Generate a random set with 0 and n-1 other elements
        other_elements = random.sample(range(1, max_val), min(n - 1, max_val - 1))
        U = sorted([0] + other_elements)
        
        s = _score_fast(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
        
        # Refine with local search
        if time.time() < deadline - 8:
            refined = _local_search(U, max_iterations=500, deadline=deadline)
            s_r = _score(refined)
            if s_r > best_score:
                best_score = s_r
                best_U = refined
    
    # Phase 5: Final high-precision polish - try adding/removing elements to best set
    if time.time() < deadline - 5:
        current_set = set(best_U)
        max_current = max(best_U)
        
        # Try adding elements in a wider range
        for new_elem in range(1, min(max_current + 50, 200)):
            if time.time() > deadline:
                break
            if new_elem in current_set:
                continue
            new_set = sorted(current_set | {new_elem})
            s = _score_fast(new_set)
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
                s = _score_fast(new_set)
                if s > best_score:
                    best_score = s
                    best_U = new_set
    
    # Phase 6: Final local search polish with tighter tolerance
    if time.time() < deadline - 2:
        refined = _local_search(best_U, max_iterations=800, deadline=deadline)
        s = _score(refined)
        if s > best_score:
            best_score = s
            best_U = refined
    
    return best_U
