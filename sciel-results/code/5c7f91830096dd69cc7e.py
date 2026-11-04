"""
Refined approach: Focus on the family of geometric/power-of-2 sets which are known to
perform well, with a much faster scoring function, a smarter continuous search over
the base parameter, and a local search that is aware of the arithmetic structure.
"""
import time
import random
import numpy as np
from math import log

# EVOLVE-BLOCK-START

def score_set(U):
    """
    Efficiently score a set U.
    Returns 1 + log(|U-U|/|U+U|) / log(2*max(U)+1)
    """
    U = sorted(set(U))
    if 0 not in U:
        return -1.0
    n = len(U)
    if n < 2:
        return -1.0
    
    # For very small sets, Python sets might be faster due to overhead
    if n <= 20:
        U_set = set(U)
        sums = set()
        diffs = set()
        for i in range(n):
            for j in range(i, n):
                sums.add(U[i] + U[j])
            for j in range(n):
                diffs.add(U[i] - U[j])
        sum_count = len(sums)
        diff_count = len(diffs)
    else:
        U_arr = np.array(U, dtype=np.int64)
        # Sums: unique values of U[i] + U[j] for all i, j
        # We can use broadcasting, but for large n this is O(n^2) memory.
        # For n up to ~50, n^2 = 2500, which is fine.
        sums_arr = U_arr[:, None] + U_arr[None, :]
        sum_count = len(np.unique(sums_arr))
        
        diffs_arr = U_arr[:, None] - U_arr[None, :]
        diff_count = len(np.unique(diffs_arr))
        
    if sum_count == 0:
        return -1.0
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1.0
        
    max_u = U[-1]
    D = 2 * max_u + 1
    if D <= 1:
        return -1.0
        
    return 1 + log(ratio) / log(D)

def score_geometric_fast(b, m):
    """
    Score a geometric progression U = {floor(b^k) for k=0..m} with 0.
    Optimized for speed by avoiding repeated list construction where possible,
    though construction is still needed for the set logic.
    """
    if b <= 1.0 or m < 1:
        return -1.0
    
    # Generate the set
    U = [0]
    val = 1.0
    for k in range(1, m + 1):
        U.append(int(val))
        val *= b
    U = sorted(set(U))
    if len(U) < 2:
        return -1.0
        
    return score_set(U)

def continuous_geometric_search(deadline):
    """
    Perform a focused search over geometric progressions.
    The key insight is that for base 2 (binary), the ratio is high.
    We will sweep b around 2.0 and m in a reasonable range.
    """
    best_score = -1.0
    best_params = (2.0, 10) # Initial guess
    
    # Coarse grid search
    # b is likely near 2. Let's check 1.5 to 3.0
    b_grid = np.linspace(1.5, 3.0, 30)
    m_grid = range(4, 20)
    
    for m in m_grid:
        if time.time() > deadline - 5:
            break
        for b in b_grid:
            s = score_geometric_fast(b, m)
            if s > best_score:
                best_score = s
                best_params = (b, m)
                
    b0, m0 = best_params
    
    # Fine-grained ternary search on b for the best m
    b_lo = max(1.5, b0 - 0.2)
    b_hi = min(3.0, b0 + 0.2)
    
    lo = b_lo
    hi = b_hi
    for _ in range(50):
        if time.time() > deadline - 5:
            break
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        s1 = score_geometric_fast(m1, m0)
        s2 = score_geometric_fast(m2, m0)
        if s1 < s2:
            lo = m1
        else:
            hi = m2
            
    b_opt = (lo + hi) / 2.0
    s_opt = score_geometric_fast(b_opt, m0)
    
    # Check neighbors in m
    for dm in [-2, -1, 1, 2]:
        m_adj = m0 + dm
        if m_adj < 2:
            continue
        if time.time() > deadline - 5:
            break
        # Quick ternary for b at this m
        lo2 = max(1.5, b_opt - 0.1)
        hi2 = min(3.0, b_opt + 0.1)
        for _ in range(30):
            if time.time() > deadline - 5:
                break
            m1 = lo2 + (hi2 - lo2) / 3.0
            m2 = hi2 - (hi2 - lo2) / 3.0
            s1 = score_geometric_fast(m1, m_adj)
            s2 = score_geometric_fast(m2, m_adj)
            if s1 < s2:
                lo2 = m1
            else:
                hi2 = m2
        b_adj = (lo2 + hi2) / 2.0
        s_adj = score_geometric_fast(b_adj, m_adj)
        if s_adj > s_opt:
            s_opt = s_adj
            b_opt = b_adj
            m0 = m_adj
            
    # Construct the final set
    U = [0]
    val = 1.0
    for k in range(1, m0 + 1):
        U.append(int(val))
        val *= b_opt
    U = sorted(set(U))
    
    return U, s_opt

def generate_candidates(U, limit_val):
    """
    Generate candidate elements for addition based on the structure of U.
    """
    U_set = set(U)
    candidates = set()
    
    # Sums and differences
    for a in U:
        for b in U:
            s = a + b
            d = abs(a - b)
            if 0 < s <= limit_val and s not in U_set:
                candidates.add(s)
            if 0 < d <= limit_val and d not in U_set:
                candidates.add(d)
                
    # Add powers of 2 nearby
    if U:
        max_u = max(U)
        # Add next power of 2
        p = 1
        while p < max_u * 2:
            p *= 2
        if p <= limit_val:
            candidates.add(p)
            candidates.add(p + 1)
            candidates.add(p - 1)
            
    return list(candidates)

def local_search_dynamic(U, deadline, max_val_limit=2000, max_n_limit=30):
    """
    Local search with dynamic candidate generation.
    """
    U = sorted(set(U))
    current_score = score_set(U)
    best_local_score = current_score
    best_local_set = list(U)
    
    candidates = generate_candidates(U, max_val_limit)
    
    no_improvement_count = 0
    max_no_improvement = 30
    
    while time.time() < deadline - 2:
        if no_improvement_count > max_no_improvement:
            break
            
        improved = False
        move_type = random.random()
        
        if not candidates or random.random() < 0.2:
            current_max = max(U) if U else 0
            new_limit = max(max_val_limit, int(current_max * 2))
            candidates = generate_candidates(U, new_limit)
            
        if move_type < 0.5:
            # ADD
            if len(U) < max_n_limit and candidates:
                random.shuffle(candidates)
                # Try top few candidates
                for c in candidates[:10]:
                    if c in U:
                        continue
                    new_U = U + [c]
                    new_U.sort()
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = generate_candidates(U, max(max_val_limit, int(max(U)*2)))
                        break
        elif move_type < 0.7:
            # REMOVE
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
            # REPLACE
            if len(U) > 1 and candidates:
                idx = random.randint(1, len(U) - 1)
                old_val = U[idx]
                random.shuffle(candidates)
                for new_val in candidates[:10]:
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
                        candidates = generate_candidates(U, max(max_val_limit, int(max(U)*2)))
                        break
                        
        if improved:
            no_improvement_count = 0
            if current_score > best_local_score:
                best_local_score = current_score
                best_local_set = list(U)
        else:
            no_improvement_count += 1
            
    return best_local_score, best_local_set

def construct():
    deadline = time.time() + 145 # Safety margin for 150s limit
    best_score = -1.0
    best_set = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    
    # 1. Evaluate initial geometric sets
    # Powers of 2 are a strong baseline
    for k in range(5, 25):
        U = [1 << i for i in range(k)]
        U.insert(0, 0)
        s = score_set(U)
        if s > best_score:
            best_score = s
            best_set = U[:]
            
    # 2. Continuous geometric search to find optimal base and length
    if time.time() < deadline - 20:
        geo_set, geo_score = continuous_geometric_search(deadline)
        if geo_set is not None and geo_score > best_score:
            best_score = geo_score
            best_set = geo_set[:]
            
    # 3. Local search refinement
    # Start from the best geometric set found
    if time.time() < deadline - 5:
        local_score, local_set = local_search_dynamic(best_set, deadline, max_val_limit=5000, max_n_limit=35)
        if local_score > best_score:
            best_score = local_score
            best_set = local_set
            
    # 4. Try a few random restarts from other structured sets if time permits
    structured_starts = [
        [0, 1, 3, 9, 27, 81, 243, 729, 2187], # Powers of 3
        [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048], # Powers of 2
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121], # Squares
    ]
    
    for start in structured_starts:
        if time.time() < deadline - 10:
            local_score, local_set = local_search_dynamic(start, deadline, max_val_limit=5000, max_n_limit=35)
            if local_score > best_score:
                best_score = local_score
                best_set = local_set

    return best_set

# EVOLVE-BLOCK-END
