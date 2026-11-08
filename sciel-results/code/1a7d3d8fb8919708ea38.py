import math
import time
import random
import numpy as np

def _compute_score(U):
    """Compute the lower bound 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)"""
    if not U or 0 not in U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    
    n = len(U)
    U_arr = np.array(U, dtype=np.int64)
    
    # Compute sumset U+U using broadcasting
    sums = U_arr[:, None] + U_arr[None, :]
    P = len(np.unique(sums))
    
    # Compute difference set U-U
    diffs = U_arr[:, None] - U_arr[None, :]
    D = len(np.unique(diffs))
    
    if P == 0:
        return 0.0
    
    denom = math.log(2 * max_u + 1, 2)
    if denom == 0:
        return 0.0
    
    score = 1.0 + math.log(D / P, 2) / denom
    return score

def _compute_score_fast(U):
    """Faster scoring for small sets using set operations."""
    if not U or 0 not in U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    
    U_set = set(U)
    n = len(U)
    
    if n <= 200:
        sum_set = set()
        diff_set = set()
        for a in U:
            for b in U:
                sum_set.add(a + b)
                diff_set.add(a - b)
        P = len(sum_set)
        D = len(diff_set)
    else:
        U_arr = np.array(U, dtype=np.int64)
        sums = U_arr[:, None] + U_arr[None, :]
        P = len(np.unique(sums))
        diffs = U_arr[:, None] - U_arr[None, :]
        D = len(np.unique(diffs))
    
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
        for mask in range(1, 1 << K):
            S_small = []
            for i in range(K):
                if mask & (1 << i):
                    S_small.append(i + 1)
            A_small = [0] + S_small
            score = _compute_score_fast(A_small)
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
        
        if max_val > 200:
            candidates = random.sample(range(1, max_val + 1), min(200, max_val))
        else:
            candidates = range(1, max_val + 1)
        
        for c in candidates:
            if c in U:
                continue
            new_U = U + [c]
            new_score = _compute_score_fast(new_U)
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
    
    for r in range(2, 20):
        for n in range(2, 15):
            U = [0]
            val = 1
            for _ in range(n):
                U.append(val)
                val *= r
            if max(U) > 100000:
                continue
            score = _compute_score_fast(U)
            if score > best_score:
                best_score = score
                best_set = U[:]
    
    for base in range(2, 10):
        for n in range(2, 20):
            U = [i * base for i in range(n)]
            if max(U) > 50000:
                continue
            score = _compute_score_fast(U)
            if score > best_score:
                best_score = score
                best_set = U[:]
    
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
                score = _compute_score_fast(U)
                if score > best_score:
                    best_score = score
                    best_set = U[:]
    
    return best_set, best_score

def _simulated_annealing(U, max_u, iterations, start_temp, end_temp):
    """Simulated annealing to optimize the set."""
    current_U = sorted(U)
    current_score = _compute_score_fast(current_U)
    best_U = current_U[:]
    best_score = current_score
    
    n_iter = iterations
    for i in range(n_iter):
        temp = start_temp * (end_temp / start_temp) ** (i / max(1, n_iter - 1))
        
        # Random move: add, remove, or swap
        move_type = random.random()
        new_U = current_U[:]
        
        if move_type < 0.4 and len(current_U) < 200:
            # Add a random element
            c = random.randint(1, max_u)
            if c not in new_U:
                new_U.append(c)
                new_U.sort()
        elif move_type < 0.7 and len(current_U) > 2:
            # Remove a random element (not 0)
            idx = random.randint(1, len(new_U) - 1)
            new_U.pop(idx)
        elif len(current_U) > 1:
            # Swap: remove one, add another
            if len(new_U) > 2:
                idx = random.randint(1, len(new_U) - 1)
                new_U.pop(idx)
            c = random.randint(1, max_u)
            if c not in new_U:
                new_U.append(c)
                new_U.sort()
        
        new_score = _compute_score_fast(new_U)
        
        if new_score >= current_score or random.random() < math.exp((new_score - current_score) / max(temp, 1e-10)):
            current_U = new_U
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_U = current_U[:]
    
    return best_U, best_score

def construct():
    deadline = time.time() + 140  # 10s safety margin
    
    best_set = [0, 1, 3]
    best_score = _compute_score_fast(best_set)
    
    # Phase 1: Brute force small sets
    if time.time() < deadline:
        for K in range(1, 16):
            if time.time() > deadline:
                break
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
                score = _compute_score_fast(A_small)
                if score > best_score:
                    best_score = score
                    best_set = A_small[:]
    
    # Phase 2: Structured families
    if time.time() < deadline:
        struct_set, struct_score = _structured_family()
        if struct_score > best_score:
            best_score = struct_score
            best_set = struct_set
    
    # Phase 3: Simulated annealing
    if time.time() < deadline:
        random.seed(42)
        max_u = max(best_set) + 100
        iters = max(100, int((deadline - time.time()) * 2))
        anneal_set, anneal_score = _simulated_annealing(best_set, max_u, iters, 0.1, 0.001)
        if anneal_score > best_score:
            best_score = anneal_score
            best_set = anneal_set
    
    # Phase 4: Greedy refinement
    if time.time() < deadline:
        U = best_set[:]
        for _ in range(50):
            if time.time() > deadline:
                break
            current_score = _compute_score_fast(U)
            best_candidate = None
            best_candidate_score = current_score
            
            max_u = max(U)
            for c in range(1, max_u + 20):
                if c in U:
                    continue
                new_U = U + [c]
                new_score = _compute_score_fast(new_U)
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
    
    # Phase 5: Final simulated annealing polish
    if time.time() < deadline and len(best_set) > 3:
        remaining = deadline - time.time()
        if remaining > 5:
            max_u = max(best_set) + 50
            iters = max(50, int(remaining * 3))
            polish_set, polish_score = _simulated_annealing(best_set, max_u, iters, 0.05, 0.0001)
            if polish_score > best_score:
                best_score = polish_score
                best_set = polish_set
    
    return sorted(set(best_set))
