import time
import random
import numpy as np
from math import log

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    U_arr = np.array(U, dtype=np.int64)
    # Sumset: all pairwise sums (with repetition)
    sums = U_arr[:, None] + U_arr[None, :]
    sums_unique = np.unique(sums)
    # Difference set: all pairwise differences (with sign)
    diffs = U_arr[:, None] - U_arr[None, :]
    diffs_unique = np.unique(diffs)
    
    sum_count = len(sums_unique)
    diff_count = len(diffs_unique)
    
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

def score_set_fast(U):
    """Vectorized scoring for speed."""
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    U_arr = np.array(U, dtype=np.int64)
    sums = U_arr[:, None] + U_arr[None, :]
    sums_unique = np.unique(sums)
    diffs = U_arr[:, None] - U_arr[None, :]
    diffs_unique = np.unique(diffs)
    sum_count = len(sums_unique)
    diff_count = len(diffs_unique)
    if sum_count == 0:
        return -1
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
    max_u = int(U_arr.max())
    D = 2 * max_u + 1
    if D <= 1:
        return -1
    return 1 + log(ratio) / log(D)

def generate_structural_seeds():
    """Generate diverse structural seeds focused on geometric and mixed structures."""
    seeds = []
    
    # Pure Powers of 2 (Baseline)
    for k in range(3, 15):
        seeds.append([0] + [2**i for i in range(1, k)])
        
    # Geometric with various bases
    for base in [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.5, 3.0, 4.0]:
        U = [0]
        val = 1
        seen = {0}
        for _ in range(30):
            val = int(round(val * base))
            if val <= 0:
                val = 1
            if val not in seen:
                U.append(val)
                seen.add(val)
            else:
                val += 1
                if val not in seen:
                    U.append(val)
                    seen.add(val)
        if len(U) > 1:
            seeds.append(sorted(U))
            
    # "Block" structures: clusters of dense points separated by large gaps
    for block_size in [2, 3, 4, 5]:
        for gap in [10, 20, 50, 100, 200]:
            U = [0]
            seen = {0}
            for i in range(1, 12):
                start = i * gap
                for j in range(block_size):
                    v = start + j
                    if v not in seen:
                        U.append(v)
                        seen.add(v)
            if len(U) > 1:
                seeds.append(sorted(U))
                
    # Random sparse with exponential spacing
    for _ in range(30):
        n = random.randint(8, 20)
        U = [0]
        last = 0
        for _ in range(n-1):
            step = random.randint(max(1, int(last * 0.3)), max(2, int(last * 0.8)))
            last += step
            U.append(last)
        seeds.append(sorted(U))
        
    # Fibonacci-like sequences
    for start_ratio in [1.5, 1.6, 1.618, 1.7, 1.8]:
        U = [0, 1]
        seen = {0, 1}
        a, b = 1, 1
        for _ in range(25):
            a, b = b, int(round(a * start_ratio + b * 0.1))
            if b not in seen and b > 0:
                U.append(b)
                seen.add(b)
        if len(U) > 1:
            seeds.append(sorted(U))
            
    # Mixed: alternating dense and sparse
    for _ in range(15):
        U = [0]
        seen = {0}
        cur = 0
        for i in range(20):
            if i % 3 == 0:
                cur += random.randint(1, 3)
            else:
                cur += random.randint(5, 20)
            if cur not in seen:
                U.append(cur)
                seen.add(cur)
        seeds.append(sorted(U))
        
    return seeds

def local_search_structural(U, deadline, max_n=40):
    """
    Structural local search with smarter moves.
    """
    U = sorted(set(U))
    if 0 not in U:
        U = [0] + [x for x in U if x > 0]
    current_score = score_set_fast(U)
    best_score = current_score
    best_U = list(U)
    
    improved = True
    iteration = 0
    max_val_bound = max(U) * 5 + 200 if U else 200
    
    while improved and time.time() < deadline - 2 and iteration < 800:
        improved = False
        iteration += 1
        n = len(U)
        
        # 1. Move elements with wider search
        for i in range(1, n):
            if time.time() > deadline - 2:
                break
            old_val = U[i]
            prev_val = U[i-1]
            next_val = U[i+1] if i < n-1 else old_val + max(20, old_val//3)
            
            search_low = max(1, prev_val + 1)
            search_high = min(max_val_bound, next_val - 1)
            
            if search_high > search_low:
                candidates = set()
                # Wider neighborhood
                for delta in range(-15, 16):
                    v = old_val + delta
                    if search_low <= v <= search_high:
                        candidates.add(v)
                # Random in gap
                for _ in range(10):
                    candidates.add(random.randint(search_low, search_high))
                # Try values that are multiples or related to other elements
                for other in U:
                    if other != old_val:
                        for mult in [2, 3]:
                            v = other * mult
                            if search_low <= v <= search_high:
                                candidates.add(v)
                
                for new_val in sorted(candidates):
                    if new_val in U:
                        continue
                    new_U = [U[j] if j != i else new_val for j in range(n)]
                    new_U = sorted(new_U)
                    new_score = score_set_fast(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        best_score = new_score
                        best_U = list(U)
                        improved = True
                        break
            if improved:
                break
                
        # 2. Add element with smarter candidates
        if not improved and n < max_n and time.time() < deadline - 2:
            max_u = max(U)
            candidates = set()
            # Near max with wider range
            for v in range(max(1, max_u - 20), max_u + 30):
                if v not in U:
                    candidates.add(v)
            # In gaps
            for i in range(1, n):
                gap = U[i] - U[i-1]
                if gap > 5:
                    mid = (U[i] + U[i-1]) // 2
                    for v in [mid, mid-1, mid+1, mid-2, mid+2]:
                        if U[i-1] < v < U[i] and v not in U:
                            candidates.add(v)
            # Try values that are sums or differences of existing elements
            for a in U:
                for b in U:
                    if a != b:
                        v = a + b
                        if v not in U and v <= max_val_bound:
                            candidates.add(v)
                        v2 = abs(a - b)
                        if v2 > 0 and v2 not in U and v2 <= max_val_bound:
                            candidates.add(v2)
                            
            for new_val in sorted(candidates):
                new_U = sorted(U + [new_val])
                new_score = score_set_fast(new_U)
                if new_score > current_score:
                    U = new_U
                    current_score = new_score
                    best_score = new_score
                    best_U = list(U)
                    improved = True
                    break
                    
        # 3. Remove element
        if not improved and n > 3 and time.time() < deadline - 2:
            for i in range(1, n):
                new_U = [U[j] for j in range(n) if j != i]
                new_score = score_set_fast(new_U)
                if new_score > current_score:
                    U = new_U
                    current_score = new_score
                    best_score = new_score
                    best_U = list(U)
                    improved = True
                    break
                    
    return best_score, best_U

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Generate diverse seeds
    seeds = generate_structural_seeds()
    
    # Score all seeds
    best_score = -1
    best_U = [0, 1]
    for seed in seeds:
        if time.time() > deadline - 40:
            break
        s = score_set_fast(seed)
        if s > best_score:
            best_score = s
            best_U = list(seed)
    
    # Run local search from the best seed and top others
    ls_deadline = start_time + 95
    
    # Run LS from the best seed and top 5 others
    ls_seeds = [best_U]
    scored_seeds = [(score_set_fast(s), s) for s in seeds]
    scored_seeds.sort(reverse=True)
    for s_score, s in scored_seeds:
        if s != best_U and len(ls_seeds) < 6:
            ls_seeds.append(list(s))
            
    for seed in ls_seeds:
        if time.time() > ls_deadline - 5:
            break
        ls_score, ls_U = local_search_structural(seed, ls_deadline, max_n=40)
        if ls_score > best_score:
            best_score = ls_score
            best_U = ls_U
    
    # Final polish: run local search again from the best result
    if time.time() < deadline - 5:
        final_deadline = deadline - 2
        ls_score, ls_U = local_search_structural(best_U, final_deadline, max_n=40)
        if ls_score > best_score:
            best_score = ls_score
            best_U = ls_U
    
    return best_U

if __name__ == "__main__":
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
