"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np
from itertools import combinations

_start_time = time.time()
_DEADLINE = 105  # safety margin before 150s limit

def _elapsed():
    return time.time() - _start_time

def _score_set(U):
    """Compute score = 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)."""
    if not U or 0 not in U:
        return -1.0
    n = len(U)
    max_u = max(U)
    if max_u == 0:
        return -1.0
    arr = np.array(U, dtype=np.int64)
    sums = arr[:, None] + arr[None, :]
    diffs = arr[:, None] - arr[None, :]
    num_sums = len(np.unique(sums))
    num_diffs = len(np.unique(diffs))
    if num_sums == 0:
        return -1.0
    ratio = num_diffs / num_sums
    if ratio <= 1:
        return -1.0
    log_denom = math.log(2 * max_u + 1)
    if log_denom <= 0:
        return -1.0
    return 1.0 + math.log(ratio) / log_denom

def _score_fast(U):
    """Faster scoring using set operations."""
    if not U or 0 not in U:
        return -1.0
    n = len(U)
    max_u = max(U)
    if max_u == 0:
        return -1.0
    U_set = set(U)
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    num_sums = len(sums)
    num_diffs = len(diffs)
    if num_sums == 0:
        return -1.0
    ratio = num_diffs / num_sums
    if ratio <= 1:
        return -1.0
    log_denom = math.log(2 * max_u + 1)
    if log_denom <= 0:
        return -1.0
    return 1.0 + math.log(ratio) / log_denom

def _prime_products(max_prime, min_product, max_product):
    """Generate products of distinct primes up to max_prime."""
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    primes = [p for p in primes if p <= max_prime]
    products = set()
    for r in range(1, len(primes) + 1):
        for combo in combinations(primes, r):
            prod = 1
            for p in combo:
                prod *= p
            if min_product <= prod <= max_product:
                products.add(prod)
    return sorted(products)

def _gap_multiplication_construct(num_gaps, max_gap):
    """Construct set using cumulative sums of prime-product gaps."""
    gaps = _prime_products(47, 2, max_gap)
    if not gaps:
        return [0]
    
    # Greedy selection of gaps to maximize ratio
    U = [0]
    current_sum = 0
    used_gaps = set()
    
    for _ in range(num_gaps):
        best_gap = None
        best_ratio = -1
        for g in gaps:
            if g in used_gaps:
                continue
            new_sum = current_sum + g
            if new_sum > max_gap * 10:  # Keep total sum reasonable
                continue
            test_U = U + [new_sum]
            ratio = _score_fast(test_U)
            if ratio > best_ratio:
                best_ratio = ratio
                best_gap = g
        
        if best_gap is None:
            break
        
        current_sum += best_gap
        used_gaps.add(best_gap)
        U.append(current_sum)
    
    return U

def _exhaustive_small_search(max_size=6, max_val=50):
    """Exhaustive search over small sets."""
    best_U = [0]
    best_score = _score_fast([0])
    
    # Try all subsets of {0,1,...,max_val} up to size max_size
    candidates = list(range(1, max_val + 1))
    
    for size in range(2, max_size + 1):
        if _elapsed() > _DEADLINE - 30:
            break
        for combo in combinations(candidates, size - 1):
            if _elapsed() > _DEADLINE - 30:
                break
            U = [0] + list(combo)
            s = _score_fast(U)
            if s > best_score:
                best_score = s
                best_U = U[:]
    
    return best_U, best_score

def _local_search_gap(initial, max_steps=5000):
    """Local search optimized for gap-based structures."""
    best_U = list(initial)
    best_score = _score_fast(best_U)
    current_U = list(best_U)
    current_score = best_score
    stall_count = 0
    
    for step in range(max_steps):
        if _elapsed() > _DEADLINE:
            break
        max_val = max(current_U) if current_U else 0
        improved = False
        cur_set = set(current_U)
        
        # Try adding a new element with gap-based candidates
        candidates = []
        # Gap-based: add max_val + prime_product
        for g in _prime_products(31, 1, max_val // 2 + 10):
            candidates.append(max_val + g)
        # Also try some random values
        for _ in range(10):
            candidates.append(random.randint(1, max_val * 2))
        random.shuffle(candidates)
        
        for new_val in candidates[:15]:
            if new_val <= 0 or new_val in cur_set:
                continue
            new_U = sorted(current_U + [new_val])
            new_score = _score_fast(new_U)
            if new_score > current_score + 1e-10:
                current_U = new_U
                current_score = new_score
                cur_set = set(current_U)
                if current_score > best_score:
                    best_U = list(current_U)
                    best_score = current_score
                improved = True
                stall_count = 0
                break
        
        if not improved:
            # Try removing an element
            if len(current_U) > 2:
                remove_candidates = list(range(1, len(current_U)))
                random.shuffle(remove_candidates)
                for idx in remove_candidates[:8]:
                    new_U = [x for i, x in enumerate(current_U) if i != idx]
                    new_score = _score_fast(new_U)
                    if new_score > current_score + 1e-10:
                        current_U = new_U
                        current_score = new_score
                        cur_set = set(current_U)
                        if current_score > best_score:
                            best_U = list(current_U)
                            best_score = current_score
                        improved = True
                        stall_count = 0
                        break
        
        if not improved:
            stall_count += 1
            if stall_count > 25:
                # Perturb: replace a random element
                if len(current_U) > 2:
                    idx = random.randint(1, len(current_U) - 1)
                    max_val = max(current_U)
                    new_val = random.randint(1, max_val * 3)
                    if new_val not in cur_set:
                        current_U[idx] = new_val
                        current_U.sort()
                        cur_set = set(current_U)
                        current_score = _score_fast(current_U)
                        stall_count = 0
                else:
                    break
            elif step > 80:
                break
    
    return best_U, best_score

def construct():
    global _start_time
    _start_time = time.time()
    
    best_U = [0, 1, 3]
    best_score = _score_fast(best_U)
    
    # Phase 1: Exhaustive search over small sets (fast, high quality)
    if _elapsed() < _DEADLINE - 60:
        U_ex, s_ex = _exhaustive_small_search(max_size=5, max_val=40)
        if s_ex > best_score:
            best_score = s_ex
            best_U = U_ex[:]
    
    # Phase 2: Gap-multiplication constructions
    if _elapsed() < _DEADLINE - 40:
        for num_gaps in range(3, 10):
            if _elapsed() > _DEADLINE - 40:
                break
            for max_gap in [10, 20, 50, 100]:
                if _elapsed() > _DEADLINE - 40:
                    break
                U_gap = _gap_multiplication_construct(num_gaps, max_gap)
                if len(U_gap) >= 2:
                    s_gap = _score_fast(U_gap)
                    if s_gap > best_score:
                        best_score = s_gap
                        best_U = U_gap[:]
    
    # Phase 3: Local search from best
    if _elapsed() < _DEADLINE - 10:
        U_ls, s_ls = _local_search_gap(best_U, max_steps=3000)
        if s_ls > best_score:
            best_score = s_ls
            best_U = U_ls
    
    # Phase 4: Local search from gap constructions
    if _elapsed() < _DEADLINE - 5:
        for num_gaps in [4, 5, 6]:
            if _elapsed() > _DEADLINE - 5:
                break
            U_seed = _gap_multiplication_construct(num_gaps, 50)
            if len(U_seed) >= 3:
                U_ls2, s_ls2 = _local_search_gap(U_seed, max_steps=1500)
                if s_ls2 > best_score:
                    best_score = s_ls2
                    best_U = U_ls2
    
    # Phase 5: Final polish
    if _elapsed() < _DEADLINE - 2:
        U_final, s_final = _local_search_gap(best_U, max_steps=1000)
        if s_final > best_score:
            best_score = s_final
            best_U = U_final
    
    return sorted(best_U)
