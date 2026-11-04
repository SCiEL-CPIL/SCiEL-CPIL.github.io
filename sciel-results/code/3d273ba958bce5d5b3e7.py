"""
Optimized search for maximizing sumset/difference-set exponent lower bound.
Uses fast numpy-based scoring, geometric progression optimization, and local search.
"""
import time
import random
import numpy as np
from math import log

# EVOLVE-BLOCK-START

def score_set(U):
    """Score a set U of nonnegative integers containing 0."""
    U = sorted(set(U))
    if 0 not in U or len(U) < 2:
        return -1.0
    
    n = len(U)
    
    # Fast path with numpy for moderate-sized sets
    if n <= 150:
        arr = np.array(U, dtype=np.int64)
        # Pairwise sums (upper triangle including diagonal)
        i_idx, j_idx = np.triu_indices(n)
        sums = arr[i_idx] + arr[j_idx]
        sum_count = len(np.unique(sums))
        
        # Pairwise differences (strictly upper triangle)
        i_idx, j_idx = np.triu_indices(n, k=1)
        if len(diffs := arr[j_idx] - arr[i_idx]) > 0:
            all_diffs = np.concatenate([diffs, -diffs, [0]])
        else:
            all_diffs = np.array([0])
        diff_count = len(np.unique(all_diffs))
    else:
        # Fallback for large sets
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
        return -1.0
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1.0
        
    max_u = U[-1]
    D = 2 * max_u + 1
    if D <= 1:
        return -1.0
        
    return 1 + log(ratio) / log(D)

def score_geometric(b, m):
    """Score a geometric progression U = {floor(b^k) for k=0..m} with 0."""
    if b <= 1.0 or m < 1:
        return -1.0
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
    """Perform continuous parameter sweep over geometric progressions."""
    best_score = -1.0
    best_b = None
    best_m = None
    
    # Coarse grid search
    for m in range(3, 20):
        if time.time() > deadline - 15:
            break
        for b in np.linspace(1.5, 6.0, 30):
            s = score_geometric(b, m)
            if s > best_score:
                best_score = s
                best_b = b
                best_m = m
    
    if best_b is None:
        return None, -1.0
    
    # Ternary search refinement
    b_lo = max(1.2, best_b - 0.5)
    b_hi = min(8.0, best_b + 0.5)
    
    for _ in range(50):
        if time.time() > deadline - 10:
            break
        m1 = b_lo + (b_hi - b_lo) / 3.0
        m2 = b_hi - (b_hi - b_lo) / 3.0
        s1 = score_geometric(m1, best_m)
        s2 = score_geometric(m2, best_m)
        if s1 < s2:
            b_lo = m1
        else:
            b_hi = m2
    
    b_opt = (b_lo + b_hi) / 2.0
    s_opt = score_geometric(b_opt, best_m)
    
    # Try adjacent m values
    for dm in [-2, -1, 1, 2]:
        m_adj = best_m + dm
        if m_adj < 2 or time.time() > deadline - 10:
            continue
        b_lo2 = max(1.2, b_opt - 0.3)
        b_hi2 = min(8.0, b_opt + 0.3)
        for _ in range(30):
            if time.time() > deadline - 10:
                break
            m1 = b_lo2 + (b_hi2 - b_lo2) / 3.0
            m2 = b_hi2 - (b_hi2 - b_lo2) / 3.0
            s1 = score_geometric(m1, m_adj)
            s2 = score_geometric(m2, m_adj)
            if s1 < s2:
                b_lo2 = m1
            else:
                b_hi2 = m2
        b_adj = (b_lo2 + b_hi2) / 2.0
        s_adj = score_geometric(b_adj, m_adj)
        if s_adj > s_opt:
            s_opt = s_adj
            b_opt = b_adj
            best_m = m_adj
    
    # Construct final set
    U = [0]
    val = 1.0
    for k in range(1, best_m + 1):
        U.append(int(val))
        val *= b_opt
    U = sorted(set(U))
    
    return U, s_opt

def generate_candidates(U, limit_val):
    """Generate candidate elements from arithmetic neighborhoods."""
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
    
    # Add some random small integers
    for _ in range(15):
        r = random.randint(1, min(30, limit_val))
        if r not in U_set:
            candidates.add(r)
            
    return list(candidates)

def local_search_dynamic(U, deadline, max_val_limit=800, max_n_limit=40):
    """Local search with dynamic candidate generation."""
    U = sorted(set(U))
    current_score = score_set(U)
    best_local_score = current_score
    best_local_set = list(U)
    
    candidates = generate_candidates(U, max_val_limit)
    
    no_improvement_count = 0
    max_no_improvement = 30
    
    while time.time() < deadline - 3:
        if no_improvement_count > max_no_improvement:
            break
            
        improved = False
        move_type = random.random()
        
        # Update candidate pool periodically
        if not candidates or random.random() < 0.12:
            current_max = max(U)
            new_limit = max(max_val_limit, current_max * 2)
            candidates = generate_candidates(U, new_limit)
        
        if move_type < 0.5:
            # Try to ADD an element
            if len(U) < max_n_limit and candidates:
                random.shuffle(candidates)
                for c in candidates[:12]:
                    if c in U:
                        continue
                    new_U = U + [c]
                    new_U.sort()
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = generate_candidates(U, max(max_val_limit, max(U) * 2))
                        break
        elif move_type < 0.7:
            # Try to REMOVE an element
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
            # Try to REPLACE an element
            if len(U) > 1 and candidates:
                idx = random.randint(1, len(U) - 1)
                old_val = U[idx]
                random.shuffle(candidates)
                for new_val in candidates[:12]:
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
                        candidates = generate_candidates(U, max(max_val_limit, max(U) * 2))
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
    deadline = time.time() + 145  # Safety margin for 150s limit
    best_score = -1.0
    best_set = [0, 1, 4, 9, 11]
    
    # Initial best
    s = score_set(best_set)
    if s > best_score:
        best_score = s
        best_set = list(best_set)

    # Structured candidates
    structured_candidates = [
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81],
        [0, 1, 2, 4, 8, 16, 32, 64, 128],
        [0, 1, 3, 7, 15, 31, 63, 127],
        [0, 1, 4, 10, 20, 35, 56, 84],
        [0, 1, 3, 4, 9, 10, 12, 13],
        [0, 1, 4, 10, 18, 28, 40, 54],
        [0, 1, 2, 4, 8, 16, 32, 64, 128, 256],
        [0, 1, 3, 9, 27, 81, 243],
        [0, 1, 5, 25, 125, 625],
        [0, 1, 6, 36, 216],
        [0, 2, 6, 18, 54, 162],
        [0, 1, 7, 49, 343],
        [0, 1, 8, 64, 512],
    ]
    
    for cand in structured_candidates:
        s = score_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)
            
    # Continuous geometric search
    if time.time() < deadline - 20:
        geo_set, geo_score = continuous_geometric_search(deadline)
        if geo_set is not None and geo_score > best_score:
            best_score = geo_score
            best_set = geo_set[:]
            structured_candidates.append(geo_set)
    
    # Local Search with Dynamic Candidate Generation
    max_val_limit = 700
    max_n_limit = 35
    
    restarts = 0
    while time.time() < deadline - 8:
        restarts += 1
        # Choose a starting set
        if restarts % 3 == 0:
            base = random.choice(structured_candidates)
            U = list(base)
        else:
            n = random.randint(4, 16)
            elements = random.sample(range(1, max_val_limit + 1), n - 1)
            U = [0] + elements
            U.sort()
        
        local_score, local_set = local_search_dynamic(U, deadline, max_val_limit, max_n_limit)
        
        if local_score > best_score:
            best_score = local_score
            best_set = local_set

    # Final polish
    if time.time() < deadline - 2:
        U = list(best_set)
        U.sort()
        final_score, final_set = local_search_dynamic(U, deadline, max_val_limit, max_n_limit)
        if final_score > best_score:
            best_score = final_score
            best_set = final_set
            
    return best_set

# EVOLVE-BLOCK-END
