import math
import time
import random

def _compute_score(U):
    """Compute the lower bound 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)"""
    if not U or 0 not in U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    
    # Compute sumset U+U
    sum_set = set()
    for a in U:
        for b in U:
            sum_set.add(a + b)
    P = len(sum_set)
    
    # Compute difference set U-U
    diff_set = set()
    for a in U:
        for b in U:
            diff_set.add(a - b)
    D = len(diff_set)
    
    if P == 0:
        return 0.0
    
    denom = math.log(2 * max_u + 1, 2)
    if denom == 0:
        return 0.0
    
    score = 1.0 + math.log(D / P, 2) / denom
    return score

def _brute_force_small(max_k):
    """Brute force search over subsets of {1,...,max_k} to find high-ratio sets."""
    best_score = 0
    best_set = [0, 1, 3]
    
    for K in range(1, max_k + 1):
        # Iterate all subsets of {1, ..., K} using bit masks
        for mask in range(1, 1 << K):
            S_small = []
            for i in range(K):
                if mask & (1 << i):
                    S_small.append(i + 1)
            A_small = [0] + S_small
            S = len(A_small)
            
            # Calculate P (sumset size) and D (difference set size)
            sum_set = set()
            diff_set = set()
            for a in A_small:
                for b in A_small:
                    sum_set.add(a + b)
                    diff_set.add(a - b)
            P = len(sum_set)
            D = len(diff_set)
            
            max_u = max(A_small)
            if max_u == 0:
                continue
            denom = math.log(2 * max_u + 1, 2)
            if denom == 0:
                continue
            score = 1.0 + math.log(D / P, 2) / denom
            
            if score > best_score:
                best_score = score
                best_set = A_small[:]
    
    return best_set, best_score

def _greedy_construct(target_size, max_val):
    """Greedy construction: add elements that maximize the ratio improvement."""
    U = [0]
    current_score = 0.0
    
    for _ in range(target_size - 1):
        best_candidate = None
        best_candidate_score = -1
        
        # Try candidates from a range
        candidates = range(1, max_val + 1)
        
        # For efficiency, sample if max_val is large
        if max_val > 200:
            candidates = random.sample(range(1, max_val + 1), min(200, max_val))
        
        for c in candidates:
            if c in U:
                continue
            new_U = U + [c]
            new_score = _compute_score(new_U)
            if new_score > best_candidate_score:
                best_candidate_score = new_score
                best_candidate = c
        
        if best_candidate is not None:
            U.append(best_candidate)
            U.sort()
            current_score = best_candidate_score
        else:
            break
    
    return U, current_score

def _structured_family():
    """Try structured families like geometric progressions with perturbations."""
    best_score = 0
    best_set = [0, 1, 3]
    
    # Geometric progression: 0, 1, r, r^2, ...
    for r in range(2, 20):
        for n in range(2, 15):
            U = [0]
            val = 1
            for _ in range(n):
                U.append(val)
                val *= r
            if max(U) > 100000:
                continue
            score = _compute_score(U)
            if score > best_score:
                best_score = score
                best_set = U[:]
    
    # Arithmetic progression with gaps
    for base in range(2, 10):
        for n in range(2, 20):
            U = [i * base for i in range(n)]
            if max(U) > 50000:
                continue
            score = _compute_score(U)
            if score > best_score:
                best_score = score
                best_set = U[:]
    
    # Mixed: some small numbers then geometric
    for small_n in range(1, 6):
        small_part = list(range(1, small_n + 1))
        for r in range(2, 15):
            for geo_n in range(1, 12):
                U = [0] + small_part
                val = small_n + 1
                for _ in range(geo_n):
                    U.append(val)
                    val = val * r + 1
                if max(U) > 100000:
                    continue
                score = _compute_score(U)
                if score > best_score:
                    best_score = score
                    best_set = U[:]
    
    return best_set, best_score

def construct():
    deadline = time.time() + 140  # 10s safety margin
    
    best_set = [0, 1, 3]
    best_score = _compute_score(best_set)
    
    # Phase 1: Brute force small sets
    if time.time() < deadline:
        for K in range(1, 16):
            if time.time() > deadline:
                break
            # For larger K, use random sampling instead of full enumeration
            if K <= 12:
                full_mask_range = range(1, 1 << K)
            else:
                full_mask_range = [random.randint(1, (1 << K) - 1) for _ in range(5000)]
            
            for mask in full_mask_range:
                if time.time() > deadline:
                    break
                S_small = []
                for i in range(K):
                    if mask & (1 << i):
                        S_small.append(i + 1)
                A_small = [0] + S_small
                score = _compute_score(A_small)
                if score > best_score:
                    best_score = score
                    best_set = A_small[:]
    
    # Phase 2: Structured families
    if time.time() < deadline:
        struct_set, struct_score = _structured_family()
        if struct_score > best_score:
            best_score = struct_score
            best_set = struct_set
    
    # Phase 3: Greedy refinement from best found
    if time.time() < deadline:
        # Try to extend the best set found
        start_set = best_set[:]
        U = start_set
        for _ in range(50):
            if time.time() > deadline:
                break
            current_score = _compute_score(U)
            best_candidate = None
            best_candidate_score = current_score
            
            # Try adding new elements
            max_u = max(U)
            for c in range(1, max_u + 20):
                if c in U:
                    continue
                new_U = U + [c]
                new_score = _compute_score(new_U)
                if new_score > best_candidate_score:
                    best_candidate_score = new_score
                    best_candidate = c
            
            if best_candidate is not None:
                U.append(best_candidate)
                U.sort()
                best_score = best_candidate_score
                best_set = U[:]
            else:
                break
    
    # Phase 4: Randomized local search
    if time.time() < deadline:
        random.seed(42)
        for _ in range(100):
            if time.time() > deadline:
                break
            # Perturb the best set
            U = best_set[:]
            # Remove a random element (except 0)
            if len(U) > 1:
                idx = random.randint(1, len(U) - 1)
                U.pop(idx)
            # Add a random new element
            max_u = max(U)
            new_elem = random.randint(1, max_u + 5)
            if new_elem not in U:
                U.append(new_elem)
                U.sort()
            score = _compute_score(U)
            if score > best_score:
                best_score = score
                best_set = U[:]
    
    return sorted(set(best_set))
