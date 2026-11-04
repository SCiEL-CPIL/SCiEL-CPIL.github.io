"""
Refined approach: Incremental scoring for fast local search + targeted replacement
+ geometric search + greedy construction with carry-free structure.
"""
import time
import random
import numpy as np
from math import log

# EVOLVE-BLOCK-START

def score_set(U):
    """Compute the score for a set U. Returns -1 if invalid."""
    U = sorted(set(U))
    if 0 not in U:
        return -1.0
    n = len(U)
    if n < 2:
        return -1.0
    
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

def compute_sums_diffs(U):
    """Compute sum set and diff set for sorted U."""
    n = len(U)
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
    return sums, diffs

def incremental_add_score(U, c, current_sums, current_diffs):
    """
    Compute the new score if we add element c to U.
    U is sorted, c not in U.
    Returns (new_score, new_sums, new_diffs) or (-1, None, None) if invalid.
    """
    n = len(U)
    new_sums = set(current_sums)
    new_diffs = set(current_diffs)
    
    # Add sums involving c
    for u in U:
        new_sums.add(u + c)
    
    # Add diffs involving c
    for u in U:
        d = c - u
        new_diffs.add(d)
        new_diffs.add(-d)
    
    sum_count = len(new_sums)
    diff_count = len(new_diffs)
    
    if sum_count == 0:
        return -1.0, None, None
    
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1.0, None, None
    
    max_u = max(U[-1], c)
    D = 2 * max_u + 1
    if D <= 1:
        return -1.0, None, None
    
    return 1 + log(ratio) / log(D), new_sums, new_diffs

def incremental_remove_score(U, idx, current_sums, current_diffs):
    """
    Compute the new score if we remove element at index idx from U.
    Returns (new_score, new_sums, new_diffs) or (-1, None, None) if invalid.
    """
    if len(U) <= 2:
        return -1.0, None, None
    if idx == 0:  # Can't remove 0
        return -1.0, None, None
    
    new_U = U[:idx] + U[idx+1:]
    n = len(new_U)
    if n < 2:
        return -1.0, None, None
    
    # Recompute from scratch for removal (simpler and still fast for small n)
    sums = set()
    diffs = set()
    for i in range(n):
        ui = new_U[i]
        for j in range(i, n):
            sums.add(ui + new_U[j])
        for j in range(i + 1, n):
            d = new_U[j] - ui
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    
    sum_count = len(sums)
    diff_count = len(diffs)
    
    if sum_count == 0:
        return -1.0, None, None
    
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1.0, None, None
    
    max_u = new_U[-1]
    D = 2 * max_u + 1
    if D <= 1:
        return -1.0, None, None
    
    return 1 + log(ratio) / log(D), sums, diffs

def greedy_construction(deadline, max_val_limit=50000, max_n_limit=40):
    """
    Greedily construct U by selecting elements that maximize the score improvement,
    using incremental scoring for efficiency.
    """
    U = [0]
    U_set = {0}
    current_sums, current_diffs = compute_sums_diffs(U)
    current_score = score_set(U)
    
    while len(U) < max_n_limit and time.time() < deadline - 10:
        # Generate candidate pool
        candidates = set()
        
        # Powers of 2 (carry-free)
        for k in range(1, 25):
            val = 1 << k
            if val <= max_val_limit and val not in U_set:
                candidates.add(val)
        
        # Arithmetic neighbors
        for a in U:
            for b in U:
                s = a + b
                d = abs(a - b)
                if 0 < s <= max_val_limit and s not in U_set:
                    candidates.add(s)
                if 0 < d <= max_val_limit and d not in U_set:
                    candidates.add(d)
        
        # Some random values
        for _ in range(30):
            r = random.randint(1, min(200, max_val_limit))
            if r not in U_set:
                candidates.add(r)
        
        if not candidates:
            break
        
        cand_list = list(candidates)
        random.shuffle(cand_list)
        
        best_cand = None
        best_score = current_score
        best_new_sums = None
        best_new_diffs = None
        
        # Try up to 80 candidates
        max_to_try = min(len(cand_list), 80)
        
        for c in cand_list[:max_to_try]:
            new_score, new_sums, new_diffs = incremental_add_score(U, c, current_sums, current_diffs)
            if new_score > best_score:
                best_score = new_score
                best_cand = c
                best_new_sums = new_sums
                best_new_diffs = new_diffs
        
        if best_cand is not None and best_score > current_score:
            U.append(best_cand)
            U.sort()
            U_set.add(best_cand)
            current_sums = best_new_sums
            current_diffs = best_new_diffs
            current_score = best_score
        else:
            # No improvement found, try to break plateau with smallest sum growth
            best_cand2 = None
            best_growth = float('inf')
            for c in cand_list[:40]:
                # Count new sums
                new_sum_count = 0
                for u in U:
                    if (u + c) not in current_sums:
                        new_sum_count += 1
                if new_sum_count < best_growth:
                    best_growth = new_sum_count
                    best_cand2 = c
            
            if best_cand2 is not None and best_cand2 not in U_set:
                U.append(best_cand2)
                U.sort()
                U_set.add(best_cand2)
                current_sums, current_diffs = compute_sums_diffs(U)
                current_score = score_set(U)
            else:
                break
    
    return U

def continuous_geometric_search(deadline):
    """Perform continuous parameter sweep over geometric progressions."""
    best_score = -1.0
    best_params = None
    
    # Coarse grid
    b_grid = np.linspace(1.5, 6.0, 25)
    m_grid = range(3, 18)
    
    for m in m_grid:
        if time.time() > deadline - 10:
            break
        for b in b_grid:
            U = [0]
            val = 1.0
            for k in range(1, m + 1):
                U.append(int(val))
                val *= b
            U = sorted(set(U))
            if len(U) < 2:
                continue
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_params = (b, m)
    
    if best_params is None:
        return None, -1.0
    
    # Ternary search around best
    b0, m0 = best_params
    lo = max(1.5, b0 - 0.5)
    hi = min(6.0, b0 + 0.5)
    
    for _ in range(50):
        if time.time() > deadline - 10:
            break
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        
        U1 = [0]
        val = 1.0
        for k in range(1, m0 + 1):
            U1.append(int(val))
            val *= m1
        U1 = sorted(set(U1))
        s1 = score_set(U1) if len(U1) >= 2 else -1.0
        
        U2 = [0]
        val = 1.0
        for k in range(1, m0 + 1):
            U2.append(int(val))
            val *= m2
        U2 = sorted(set(U2))
        s2 = score_set(U2) if len(U2) >= 2 else -1.0
        
        if s1 < s2:
            lo = m1
        else:
            hi = m2
    
    b_opt = (lo + hi) / 2.0
    U = [0]
    val = 1.0
    for k in range(1, m0 + 1):
        U.append(int(val))
        val *= b_opt
    U = sorted(set(U))
    s_opt = score_set(U)
    
    # Try adjacent m values
    for m_adj in [m0 - 1, m0 + 1, m0 - 2, m0 + 2]:
        if m_adj < 2 or time.time() > deadline - 10:
            continue
        U = [0]
        val = 1.0
        for k in range(1, m_adj + 1):
            U.append(int(val))
            val *= b_opt
        U = sorted(set(U))
        if len(U) >= 2:
            s = score_set(U)
            if s > s_opt:
                s_opt = s
                m0 = m_adj
    
    return U, s_opt

def local_search_fast(U, deadline, max_val_limit=1000, max_n_limit=35):
    """
    Fast local search using incremental scoring.
    """
    U = sorted(set(U))
    current_sums, current_diffs = compute_sums_diffs(U)
    current_score = score_set(U)
    best_score = current_score
    best_set = list(U)
    
    U_set = set(U)
    
    no_improvement = 0
    max_no_improvement = 50
    
    while time.time() < deadline - 3:
        if no_improvement > max_no_improvement:
            break
        
        improved = False
        move_type = random.random()
        
        if move_type < 0.55:
            # ADD
            if len(U) < max_n_limit:
                # Generate candidates
                candidates = set()
                for a in U:
                    for b in U:
                        s = a + b
                        d = abs(a - b)
                        if 0 < s <= max_val_limit and s not in U_set:
                            candidates.add(s)
                        if 0 < d <= max_val_limit and d not in U_set:
                            candidates.add(d)
                # Powers of 2
                for k in range(1, 20):
                    val = 1 << k
                    if val <= max_val_limit and val not in U_set:
                        candidates.add(val)
                # Random
                for _ in range(15):
                    r = random.randint(1, min(100, max_val_limit))
                    if r not in U_set:
                        candidates.add(r)
                
                if candidates:
                    cand_list = list(candidates)
                    random.shuffle(cand_list)
                    for c in cand_list[:15]:
                        new_score, new_sums, new_diffs = incremental_add_score(U, c, current_sums, current_diffs)
                        if new_score > current_score:
                            U.append(c)
                            U.sort()
                            U_set.add(c)
                            current_sums = new_sums
                            current_diffs = new_diffs
                            current_score = new_score
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best_set = list(U)
                            break
                            
        elif move_type < 0.75:
            # REMOVE
            if len(U) > 3:
                removable_indices = list(range(1, len(U)))
                random.shuffle(removable_indices)
                for idx in removable_indices[:5]:
                    new_score, new_sums, new_diffs = incremental_remove_score(U, idx, current_sums, current_diffs)
                    if new_score > current_score:
                        removed_val = U[idx]
                        U = U[:idx] + U[idx+1:]
                        U_set.discard(removed_val)
                        current_sums = new_sums
                        current_diffs = new_diffs
                        current_score = new_score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_set = list(U)
                        break
                        
        else:
            # REPLACE
            if len(U) > 2:
                idx = random.randint(1, len(U) - 1)
                old_val = U[idx]
                # Generate replacement candidates
                candidates = set()
                for a in U:
                    for b in U:
                        s = a + b
                        d = abs(a - b)
                        if 0 < s <= max_val_limit and s not in U_set:
                            candidates.add(s)
                        if 0 < d <= max_val_limit and d not in U_set:
                            candidates.add(d)
                for k in range(1, 15):
                    val = 1 << k
                    if val <= max_val_limit and val not in U_set:
                        candidates.add(val)
                
                if candidates:
                    cand_list = list(candidates)
                    random.shuffle(cand_list)
                    for new_val in cand_list[:10]:
                        if new_val == old_val or new_val in U_set:
                            continue
                        # Remove old, add new
                        temp_U = U[:idx] + U[idx+1:]
                        temp_U_set = U_set - {old_val}
                        # Check if new_val is valid
                        if new_val in temp_U_set:
                            continue
                        # Score the replacement
                        temp_U.append(new_val)
                        temp_U.sort()
                        s = score_set(temp_U)
                        if s > current_score:
                            U = temp_U
                            U_set = set(U)
                            current_sums, current_diffs = compute_sums_diffs(U)
                            current_score = s
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best_set = list(U)
                            break
        
        if improved:
            no_improvement = 0
        else:
            no_improvement += 1
    
    return best_score, best_set

def construct():
    deadline = time.time() + 140  # Safety margin for 150s limit
    best_score = -1.0
    best_set = [0, 1]
    
    # Initial structured candidates
    structured_candidates = [
        [0, 1, 4, 9, 16, 25, 36, 49, 64],
        [0, 1, 2, 4, 8, 16, 32, 64, 128],
        [0, 1, 3, 7, 15, 31, 63, 127],
        [0, 1, 4, 10, 20, 35, 56],
        [0, 1, 3, 4, 9, 10, 12, 13],
        [0, 1, 3, 9, 27, 81, 243],
        [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
        [0, 1, 5, 10, 15, 20, 25, 30],
        [0, 1, 4, 10, 18, 28, 40, 54],
    ]
    
    for cand in structured_candidates:
        s = score_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)
    
    # 1. Continuous geometric search
    if time.time() < deadline - 30:
        geo_set, geo_score = continuous_geometric_search(deadline)
        if geo_set is not None and geo_score > best_score:
            best_score = geo_score
            best_set = geo_set[:]
            structured_candidates.append(geo_set)
    
    # 2. Greedy construction with incremental scoring
    if time.time() < deadline - 20:
        greedy_set = greedy_construction(deadline, max_val_limit=50000, max_n_limit=40)
        greedy_score = score_set(greedy_set)
        if greedy_score > best_score:
            best_score = greedy_score
            best_set = greedy_set[:]
            structured_candidates.append(greedy_set)
    
    # 3. Multiple local search restarts with incremental scoring
    max_val_limit = 800
    max_n_limit = 35
    
    restarts = 0
    while time.time() < deadline - 8:
        restarts += 1
        
        if restarts % 3 == 0:
            base = random.choice(structured_candidates)
            U = list(base)
        elif restarts % 3 == 1:
            # Random geometric
            b = random.uniform(1.8, 4.0)
            m = random.randint(4, 12)
            U = [0]
            val = 1.0
            for k in range(1, m + 1):
                U.append(int(val))
                val *= b
            U = sorted(set(U))
        else:
            # Random sparse
            n = random.randint(4, 12)
            elements = random.sample(range(1, max_val_limit + 1), n - 1)
            U = [0] + elements
            U.sort()
        
        local_score, local_set = local_search_fast(U, deadline, max_val_limit, max_n_limit)
        
        if local_score > best_score:
            best_score = local_score
            best_set = local_set
            structured_candidates.append(local_set)
    
    # Final polish on the best set
    if time.time() < deadline - 3:
        final_score, final_set = local_search_fast(list(best_set), deadline, max_val_limit, max_n_limit)
        if final_score > best_score:
            best_score = final_score
            best_set = final_set
    
    return best_set

# EVOLVE-BLOCK-END
