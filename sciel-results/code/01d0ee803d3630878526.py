"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from math import log
from itertools import combinations

def _compute_score(U):
    """Compute the score for a given set U using NumPy for speed."""
    U_arr = np.array(sorted(set(U)), dtype=np.int64)
    if len(U_arr) == 0:
        return 0.0
    if 0 not in U_arr:
        return 0.0
    n = len(U_arr)
    if n == 0:
        return 0.0
    max_u = U_arr[-1]
    if max_u == 0:
        return 1.0
    
    # Compute U+U and U-U
    # For small n, this is very fast
    sumset = np.unique(U_arr[:, None] + U_arr[None, :])
    diffset = np.unique(U_arr[:, None] - U_arr[None, :])
    
    len_sum = len(sumset)
    len_diff = len(diffset)
    
    if len_sum == 0:
        return 0.0
        
    ratio = len_diff / len_sum
    if ratio <= 0:
        return 0.0
    
    denominator = log(2 * max_u + 1)
    if denominator == 0:
        return 1.0
    
    score = 1 + log(ratio) / denominator
    return score

def _exhaustive_small_search(max_val=20, max_size=5):
    """Find the best small sets by exhaustive search."""
    best_set = [0, 1, 2, 5]
    best_score = _compute_score(best_set)
    
    range_val = range(max_val + 1)
    
    for size in range(3, max_size + 1):
        for comb in combinations(range_val, size):
            if 0 not in comb:
                continue
            s = list(comb)
            score = _compute_score(s)
            if score > best_score:
                best_score = score
                best_set = s
                
    return best_set, best_score

def _local_search(U, max_iterations=20000, timeout=10.0):
    """Local search to improve the set."""
    start_time = time.time()
    current = sorted(set(U))
    if 0 not in current:
        current.insert(0, 0)
    current_score = _compute_score(current)
    best = list(current)
    best_score = current_score
    
    for i in range(max_iterations):
        if time.time() - start_time > timeout:
            break
        
        U_set = set(current)
        max_u = max(U_set) if U_set else 0
        n = len(U_set)
        
        # Generate candidates for addition
        candidates = set()
        
        # 1. Small perturbations around existing elements
        for x in list(U_set):
            for offset in [-3, -2, -1, 1, 2, 3, 4, 5, 7, 8]:
                c = x + offset
                if c >= 0:
                    candidates.add(c)
        
        # 2. Structure-preserving: sums and diffs of pairs
        # Limit to first 30 elements to keep it fast
        elems = list(U_set)
        if n > 30:
            elems = elems[:30]
        for idx_a in range(len(elems)):
            for idx_b in range(len(elems)):
                s = elems[idx_a] + elems[idx_b]
                if s <= 2 * max_u + 50:
                    candidates.add(s)
                d = abs(elems[idx_a] - elems[idx_b])
                if d > 0 and d <= 2 * max_u + 50:
                    candidates.add(d)
        
        # 3. Powers and gaps
        p = 1
        while p <= 2 * max_u + 50:
            candidates.add(p)
            p *= 2
        p = 1
        while p <= 2 * max_u + 50:
            candidates.add(p)
            p *= 3
            
        # 4. Random candidates
        for _ in range(30):
            candidates.add(random.randint(0, 2 * max_u + 50))
            
        # 5. Specific "gap" candidates
        if max_u > 0:
            candidates.add(max_u + 1)
            candidates.add(max_u + 2)
            candidates.add(max_u + 3)
            candidates.add(max_u * 2)
            candidates.add(max_u * 2 + 1)
            
        candidate_list = list(candidates)
        random.shuffle(candidate_list)
        
        improved = False
        for cand in candidate_list:
            if time.time() - start_time > timeout:
                break
            if cand in U_set:
                continue
            new_set = U_set | {cand}
            new_score = _compute_score(new_set)
            if new_score > current_score + 1e-9:
                current = sorted(new_set)
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best = list(current)
                improved = True
                break
        
        if not improved:
            # Try removing an element
            removable = [x for x in current if x != 0]
            random.shuffle(removable)
            for cand in removable[:20]:
                if time.time() - start_time > timeout:
                    break
                new_set = U_set - {cand}
                if len(new_set) < 2:
                    continue
                new_score = _compute_score(new_set)
                if new_score > current_score + 1e-9:
                    current = sorted(new_set)
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best = list(current)
                    improved = True
                    break
        
        if not improved and i > 20:
            # Try a swap
            if random.random() < 0.5:
                removable = [x for x in current if x != 0]
                if removable:
                    rem = random.choice(removable)
                    new_elem = random.randint(0, 2 * max_u + 100)
                    new_set = (U_set - {rem}) | {new_elem}
                    if len(new_set) >= 2:
                        new_score = _compute_score(new_set)
                        if new_score > current_score + 1e-9:
                            current = sorted(new_set)
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best = list(current)
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # 1. Exhaustive search for best small sets
    best_seed, best_seed_score = _exhaustive_small_search(max_val=25, max_size=5)
    
    # 2. Local search from the best seed
    improved, score = _local_search(best_seed, max_iterations=50000, timeout=30.0)
    if score > best_seed_score:
        best_seed = improved
        best_seed_score = score
        
    # 3. Try a few other promising structures as seeds
    # Geometric progression
    geo = [0]
    v = 1
    while v <= 100:
        geo.append(v)
        v *= 2
    geo_score = _compute_score(geo)
    if geo_score > best_seed_score:
        best_seed = geo
        best_seed_score = geo_score
        improved, score = _local_search(best_seed, max_iterations=50000, timeout=20.0)
        if score > best_seed_score:
            best_seed = improved
            best_seed_score = score
            
    # 4. Final polish
    if time.time() < deadline - 5:
        remaining = deadline - time.time() - 2
        if remaining > 5:
            final_result, final_score = _local_search(best_seed, max_iterations=20000, timeout=remaining)
            if final_score > best_seed_score:
                best_seed = final_result
                best_seed_score = final_score
                
    # Final check
    final_score = _compute_score(best_seed)
    if final_score < 1.0:
        best_seed = [0, 1, 2, 5]
        
    return best_seed
# EVOLVE-BLOCK-END
