"""
Combined approach: Structural insight (dynamic candidate generation from arithmetic neighborhoods)
+ Refinement machinery (efficient scoring, ternary/geometric search, and local search with SA).
+ Polynomial support insight: |U+U| ~ support(P^2), |U-U| ~ support(P(x)P(x^-1)).
  We use this to guide candidate generation and a final polish that exploits the algebraic structure.
"""
import time
import random
import numpy as np
from math import log

# EVOLVE-BLOCK-START

def score_set(U):
    """
    Compute the score: 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)
    Uses NumPy for speed.
    """
    U = sorted(set(U))
    if 0 not in U:
        return -1.0
    n = len(U)
    if n < 2:
        return -1.0
    
    U_arr = np.array(U, dtype=np.int64)
    
    # Compute all sums (i <= j)
    sums_arr = U_arr[:, None] + U_arr[None, :]
    sum_vals = np.unique(sums_arr)
    sum_count = len(sum_vals)
    
    # Compute all differences (i < j, and their negatives, plus 0)
    diffs_arr = U_arr[:, None] - U_arr[None, :]
    diff_vals = np.unique(diffs_arr)
    diff_count = len(diff_vals)
    
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
    best_params = None
    
    # Coarse grid over b and m
    b_grid = np.linspace(1.5, 5.0, 25)
    m_grid = range(3, 15)
    
    for m in m_grid:
        if time.time() > deadline - 10:
            break
        for b in b_grid:
            s = score_geometric(b, m)
            if s > best_score:
                best_score = s
                best_params = (b, m)
    
    if best_params is None:
        return None, -1.0
    
    # Phase 2: Ternary search around the best coarse point
    b0, m0 = best_params
    b_lo = max(1.5, b0 - 0.5)
    b_hi = min(5.0, b0 + 0.5)
    
    lo = b_lo
    hi = b_hi
    for _ in range(40):
        if time.time() > deadline - 10:
            break
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        s1 = score_geometric(m1, m0)
        s2 = score_geometric(m2, m0)
        if s1 < s2:
            lo = m1
        else:
            hi = m2
    
    b_opt = (lo + hi) / 2.0
    s_opt = score_geometric(b_opt, m0)
    
    # Also try adjacent m values
    for m_adj in [m0 - 1, m0 + 1]:
        if m_adj < 2 or time.time() > deadline - 10:
            continue
        lo2 = max(1.5, b_opt - 0.3)
        hi2 = min(5.0, b_opt + 0.3)
        for _ in range(20):
            if time.time() > deadline - 10:
                break
            m1 = lo2 + (hi2 - lo2) / 3.0
            m2 = hi2 - (hi2 - lo2) / 3.0
            s1 = score_geometric(m1, m_adj)
            s2 = score_geometric(m2, m_adj)
            if s1 < s2:
                lo2 = m1
            else:
                hi2 = m2
        b_adj = (lo2 + hi2) / 2.0
        s_adj = score_geometric(b_adj, m_adj)
        if s_adj > s_opt:
            s_opt = s_adj
            b_opt = b_adj
            m0 = m_adj
    
    U = [0]
    val = 1.0
    for k in range(1, m0 + 1):
        U.append(int(val))
        val *= b_opt
    U = sorted(set(U))
    
    return U, s_opt

def generate_candidates(U, limit_val):
    """
    Generate candidate elements for addition based on sums and differences of existing elements.
    This is the structural improvement: exploring the arithmetic neighborhood.
    """
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
    
    for _ in range(20):
        r = random.randint(1, min(50, limit_val))
        if r not in U_set:
            candidates.add(r)
            
    return list(candidates)

def local_search_dynamic(U, deadline, max_val_limit=1000, max_n_limit=40):
    """
    Local search with dynamic candidate generation.
    """
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
        
        if not candidates or random.random() < 0.12:
            current_max = max(U)
            new_limit = max(max_val_limit, int(current_max * 1.5))
            candidates = generate_candidates(U, new_limit)
        
        if move_type < 0.45:
            if len(U) < max_n_limit and candidates:
                random.shuffle(candidates)
                for c in candidates[:8]:
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
        elif move_type < 0.65:
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
                for new_val in candidates[:8]:
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

def polish_polynomial(U, deadline):
    """
    Final polish using the polynomial insight.
    We model U as the support of a polynomial P(x).
    The key idea: try small perturbations that exploit the algebraic structure.
    Specifically, we try to add elements that are "aligned" with the existing structure
    in a way that maximizes |U-U|/|U+U|.
    """
    U = sorted(set(U))
    current_score = score_set(U)
    best_score = current_score
    best_set = list(U)
    
    # Generate a richer candidate pool based on the polynomial structure
    # Candidates: sums, differences, and also "near-misses" that could fill gaps
    U_set = set(U)
    max_u = max(U)
    candidates = set()
    
    # Sums and differences
    for a in U:
        for b in U:
            s = a + b
            d = abs(a - b)
            if 0 < s <= 2 * max_u and s not in U_set:
                candidates.add(s)
            if 0 < d <= 2 * max_u and d not in U_set:
                candidates.add(d)
    
    # Also try small integers and multiples
    for k in range(1, 5):
        for u in U:
            val = u * k
            if 0 < val <= 2 * max_u and val not in U_set:
                candidates.add(val)
    
    candidates = list(candidates)
    random.shuffle(candidates)
    
    # Try adding one element at a time
    for c in candidates[:50]:
        if time.time() > deadline - 1:
            break
        new_U = U + [c]
        new_U.sort()
        new_score = score_set(new_U)
        if new_score > current_score:
            U = new_U
            current_score = new_score
            U_set = set(U)
    
    # Try removing one element
    for c in [x for x in U if x != 0][:20]:
        if time.time() > deadline - 1:
            break
        new_U = [x for x in U if x != c]
        new_score = score_set(new_U)
        if new_score > current_score:
            U = new_U
            current_score = new_score
            U_set = set(U)
    
    if current_score > best_score:
        best_score = current_score
        best_set = list(U)
    
    return best_score, best_set

def construct():
    deadline = time.time() + 110
    best_score = -1.0
    best_set = [0, 1, 4, 9, 11]
    
    s = score_set(best_set)
    if s > best_score:
        best_score = s
        best_set = list(best_set)

    structured_candidates = [
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81],
        [0, 1, 2, 4, 8, 16, 32, 64, 128],
        [0, 1, 3, 7, 15, 31, 63, 127],
        [0, 1, 4, 10, 20, 35, 56, 84],
        [0, 1, 3, 4, 9, 10, 12, 13],
        [0, 1, 4, 10, 18, 28, 40, 54],
        [0, 1, 2, 4, 8, 16, 32, 64, 128, 256],
        [0, 1, 3, 9, 27, 81, 243],
    ]
    
    for cand in structured_candidates:
        s = score_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)
            
    if time.time() < deadline - 20:
        geo_set, geo_score = continuous_geometric_search(deadline)
        if geo_set is not None and geo_score > best_score:
            best_score = geo_score
            best_set = geo_set[:]
            structured_candidates.append(geo_set)
    
    max_val_limit = 500
    max_n_limit = 30
    
    restarts = 0
    while time.time() < deadline - 8:
        restarts += 1
        if restarts % 3 == 0:
            base = random.choice(structured_candidates)
            U = list(base)
        else:
            n = random.randint(4, 15)
            if n - 1 > max_val_limit:
                n = max_val_limit
            elements = random.sample(range(1, max_val_limit + 1), n - 1)
            U = [0] + elements
            U.sort()
        
        local_score, local_set = local_search_dynamic(U, deadline, max_val_limit, max_n_limit)
        
        if local_score > best_score:
            best_score = local_score
            best_set = local_set

    # Final polish using polynomial insight
    if time.time() < deadline - 2:
        U = list(best_set)
        U.sort()
        final_score, final_set = polish_polynomial(U, deadline)
        if final_score > best_score:
            best_score = final_score
            best_set = final_set
            
    return best_set

# EVOLVE-BLOCK-END
