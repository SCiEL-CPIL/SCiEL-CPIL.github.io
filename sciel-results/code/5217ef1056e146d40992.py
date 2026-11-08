import time
from math import log, floor, log2
import random
import numpy as np
from itertools import combinations

def score_set(U):
    """Compute the score for a set U."""
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    sums = set()
    diffs = set()
    
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sums.add(ui + U[j])
        for j in range(i + 1, n):
            d = U[j] - ui
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    
    sum_count = len(sums)
    diff_count = len(diffs)
    
    if sum_count == 0:
        return -1
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
        
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def generate_candidates(U, limit_val):
    """Generate candidate elements for addition based on sums and differences."""
    U_set = set(U)
    candidates = set()
    
    for a in U:
        for b in U:
            s = a + b
            d = abs(a - b)
            if 0 < s <= limit_val and s not in U_set:
                candidates.add(s)
            if 0 < d <= limit_val and d not in U_set:
                candidates.add(d)
            
    return list(candidates)

def local_search_dynamic(U, deadline, max_val_limit=2000, max_n_limit=50):
    """Local search with dynamic candidate generation."""
    U = sorted(set(U))
    current_score = score_set(U)
    best_local_score = current_score
    best_local_set = list(U)
    
    candidates = generate_candidates(U, max_val_limit)
    
    no_improvement_count = 0
    max_no_improvement = 25
    
    while time.time() < deadline - 2:
        if no_improvement_count > max_no_improvement:
            break
            
        improved = False
        move_type = random.random()
        
        if not candidates or random.random() < 0.1:
            current_max = max(U)
            new_limit = max(max_val_limit, current_max * 2)
            candidates = generate_candidates(U, new_limit)
        
        if move_type < 0.5:
            if len(U) < max_n_limit and candidates:
                random.shuffle(candidates)
                for c in candidates[:15]:
                    if c in U:
                        continue
                    new_U = U + [c]
                    new_U.sort()
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = generate_candidates(U, max(max_val_limit, max(U)*2))
                        break
        elif move_type < 0.7:
            if len(U) > 2:
                removable = [x for x in U if x != 0]
                if removable:
                    c = random.choice(removable)
                    new_U = [x for x in U if x != c]
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = generate_candidates(U, max_val_limit)
        else:
            if len(U) > 1 and candidates:
                idx = random.randint(1, len(U) - 1)
                old_val = U[idx]
                random.shuffle(candidates)
                for new_val in candidates[:15]:
                    if new_val == old_val or new_val in U:
                        continue
                    new_U = U.copy()
                    new_U[idx] = new_val
                    new_U.sort()
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = generate_candidates(U, max(max_val_limit, max(U)*2))
                        break
            
        if improved:
            no_improvement_count = 0
            if current_score > best_local_score:
                best_local_score = current_score
                best_local_set = list(U)
        else:
            no_improvement_count += 1
                
    return best_local_score, best_local_set

def digit_based_local_search(U, deadline):
    """Local search treating the problem as discrete spectral optimization."""
    U = sorted(set(U))
    current_score = score_set(U)
    best_score = current_score
    best_set = list(U)
    
    if len(U) == 1:
        return best_score, best_set
    
    max_val = max(U)
    b = max(2, int(log(max_val, 10)) + 2) if max_val > 10 else 2
    
    powers = []
    p = 1
    while p <= max_val * 2:
        powers.append(p)
        p *= b
        
    no_improvement = 0
    max_no_improvement = 35
    
    while time.time() < deadline - 5:
        if no_improvement > max_no_improvement:
            break
            
        improved = False
        idx = random.randint(1, len(U) - 1)
        old_val = U[idx]
        
        digits = []
        temp = old_val
        for p in powers:
            digits.append(temp % b)
            temp //= b
            if temp == 0:
                break
        
        for _ in range(8):
            new_digits = digits[:]
            if not new_digits:
                new_digits = [0]
            
            d_idx = random.randint(0, len(new_digits) - 1)
            if random.random() < 0.5:
                new_digits[d_idx] = max(0, min(b-1, new_digits[d_idx] + random.randint(-2, 2)))
            else:
                new_digits[d_idx] = random.randint(0, b-1)
                
            new_val = 0
            for i, d in enumerate(new_digits):
                if i < len(powers):
                    new_val += d * powers[i]
                else:
                    new_val += d * (powers[-1] * b)
                    
            if new_val <= 0 or new_val in U:
                continue
                
            if new_val > max_val * 3:
                continue
                
            new_U = U.copy()
            new_U[idx] = new_val
            new_U = sorted(set(new_U))
            
            if len(new_U) != len(U):
                continue
                
            new_score = score_set(new_U)
            if new_score > current_score:
                U = new_U
                current_score = new_score
                improved = True
                break
        else:
            if len(U) > 1:
                idx2 = random.randint(1, len(U) - 1)
                if idx2 != idx:
                    new_U = U.copy()
                    new_U[idx], new_U[idx2] = new_U[idx2], new_U[idx]
                    new_U = sorted(set(new_U))
                    if len(new_U) == len(U):
                        new_score = score_set(new_U)
                        if new_score > current_score:
                            U = new_U
                            current_score = new_score
                            improved = True
        
        if improved:
            no_improvement = 0
            if current_score > best_score:
                best_score = current_score
                best_set = list(U)
        else:
            no_improvement += 1
            
    return best_score, best_set

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    best_set = [0, 1, 4, 9, 11]
    best_score = score_set(best_set)

    # 1. Structured Candidates
    structured_candidates = [
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81],
        [0, 1, 2, 4, 8, 16, 32, 64, 128],
        [0, 1, 3, 7, 15, 31, 63, 127],
        [0, 1, 4, 10, 20, 35, 56, 84],
        [0, 1, 3, 4, 9, 10, 12, 13],
        [0, 1, 4, 10, 18, 28, 40, 54],
        [0, 1, 2, 4, 8, 16, 32, 64, 128, 256],
        [0, 1, 3, 9, 27, 81, 243],
        [0, 1, 4, 9, 11],
        [0, 1, 2, 5, 10, 20, 40],
        [0, 1, 3, 6, 12, 24, 48],
        [0, 2, 5, 11, 23, 47],
        [0, 1, 5, 10, 21, 42],
    ]
    
    for cand in structured_candidates:
        s = score_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # 2. Exhaustive Search for Small n (optimized)
    def exhaustive_search(n, max_val):
        """Find best set of size n (including 0) with elements <= max_val."""
        best_s = -1
        best_c = None
        range_vals = list(range(1, max_val + 1))
        for comb in combinations(range_vals, n - 1):
            U = [0] + list(comb)
            s = score_set(U)
            if s > best_s:
                best_s = s
                best_c = U
        return best_s, best_c

    if time.time() < deadline - 60:
        s, c = exhaustive_search(3, 60)
        if s > best_score:
            best_score = s
            best_set = c
        
        s, c = exhaustive_search(4, 50)
        if s > best_score:
            best_score = s
            best_set = c
            
        s, c = exhaustive_search(5, 40)
        if s > best_score:
            best_score = s
            best_set = c

    # 3. Geometric Search with finer grid
    b_candidates = [b_int / 10.0 for b_int in range(11, 120)]
    best_b = None
    best_m = None
    best_grid_score = -1

    for b in b_candidates:
        if time.time() > deadline - 50:
            break
        for m in range(2, 25):
            U = [0]
            seen = {0}
            for k in range(1, m + 1):
                val = floor(b ** k)
                if val <= 0:
                    val = 1
                if val not in seen:
                    U.append(val)
                    seen.add(val)
            U.sort()
            s = score_set(U)
            if s > best_grid_score:
                best_grid_score = s
                best_b = b
                best_m = m
                best_set = U
                best_score = s

    # 4. Local Search from Best Seeds
    search_seeds = [best_set]
    
    for cand in structured_candidates:
        if cand != best_set:
            search_seeds.append(cand)
            
    for _ in range(8):
        pert = best_set[:]
        if len(pert) > 1:
            idx = random.randint(1, len(pert)-1)
            pert[idx] += random.randint(-8, 8)
            if pert[idx] > 0:
                pert = sorted(set(pert))
                if len(pert) >= 2 and 0 in pert:
                    search_seeds.append(pert)

    # 5. Hybrid Local Search Phase
    for seed in search_seeds:
        if time.time() > deadline - 15:
            break
        
        local_score, local_set = local_search_dynamic(seed, deadline)
        if local_score > best_score:
            best_score = local_score
            best_set = local_set
            
        if time.time() < deadline - 15:
            digit_score, digit_set = digit_based_local_search(local_set, deadline)
            if digit_score > best_score:
                best_score = digit_score
                best_set = digit_set

    # 6. Polish pass with tighter tolerance
    if time.time() < deadline - 5:
        polish_score, polish_set = local_search_dynamic(best_set, deadline, max_val_limit=3000, max_n_limit=60)
        if polish_score > best_score:
            best_score = polish_score
            best_set = polish_set

    return best_set

if __name__ == "__main__":
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
