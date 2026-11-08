"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import itertools
from functools import lru_cache

# EVOLVE-BLOCK-START
import numpy as np

def _score_set(U):
    """Compute the lower bound 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)."""
    if not U:
        return 0.0
    U = list(set(U))
    if 0 not in U:
        U.append(0)
    max_u = max(U)
    if max_u <= 0:
        return 0.0
    denom = 2 * max_u + 1
    if denom <= 1:
        return 0.0
    log_denom = math.log(denom)
    if log_denom <= 0:
        return 0.0
    
    # Compute U+U and U-U
    U_set = set(U)
    sum_set = set()
    diff_set = set()
    for a in U:
        for b in U:
            sum_set.add(a + b)
            diff_set.add(a - b)
    
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 0:
        return 0.0
    return 1.0 + math.log(ratio) / log_denom

def _try_set(U):
    """Validate and score a candidate set."""
    try:
        if not U:
            return 0.0
        U = [int(x) for x in U]
        if any(x < 0 for x in U):
            return 0.0
        if 0 not in U:
            U.append(0)
        U = list(set(U))
        return _score_set(U)
    except:
        return 0.0

def _exhaustive_search_small(n_max=6, max_val=20):
    """Exhaustively search for small sets with n elements (including 0)."""
    best_score = 0.0
    best_set = [0, 1, 3]
    
    for n in range(2, n_max + 1):
        # Choose n-1 elements from 1..max_val, always include 0
        for combo in itertools.combinations(range(1, max_val + 1), n - 1):
            U = [0] + list(combo)
            s = _try_set(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
    return best_score, best_set

def _greedy_construct(max_u, target_size):
    """Greedy construction: add elements that maximize score improvement."""
    U = [0]
    # Start with a small seed
    candidates = list(range(1, max_u + 1))
    
    for _ in range(target_size - 1):
        best_elem = None
        best_score = _try_set(U)
        # Try adding each candidate
        for elem in candidates:
            if elem in U:
                continue
            new_U = U + [elem]
            s = _try_set(new_U)
            if s > best_score:
                best_score = s
                best_elem = elem
        if best_elem is None:
            break
        U.append(best_elem)
        candidates.remove(best_elem)
    return U

def _local_search(U, max_u, num_iters=500):
    """Local search: try adding, removing, or swapping elements."""
    U = list(set(U))
    if 0 not in U:
        U.append(0)
    current_score = _try_set(U)
    U_set = set(U)
    candidates = set(range(1, max_u + 1)) - U_set
    
    for _ in range(num_iters):
        improved = False
        
        # Try adding
        for elem in list(candidates):
            new_U = U + [elem]
            s = _try_set(new_U)
            if s > current_score + 1e-10:
                U = new_U
                U_set.add(elem)
                candidates.discard(elem)
                current_score = s
                improved = True
                break
        
        if improved:
            continue
        
        # Try removing (not 0)
        for elem in list(U_set):
            if elem == 0:
                continue
            new_U = [x for x in U if x != elem]
            s = _try_set(new_U)
            if s > current_score + 1e-10:
                U = new_U
                U_set.discard(elem)
                candidates.add(elem)
                current_score = s
                improved = True
                break
        
        if improved:
            continue
        
        # Try swapping
        for add_elem in list(candidates):
            for rem_elem in list(U_set):
                if rem_elem == 0:
                    continue
                new_U = [x for x in U if x != rem_elem] + [add_elem]
                s = _try_set(new_U)
                if s > current_score + 1e-10:
                    U = new_U
                    U_set.discard(rem_elem)
                    U_set.add(add_elem)
                    candidates.discard(add_elem)
                    candidates.add(rem_elem)
                    current_score = s
                    improved = True
                    break
            if improved:
                break
    
    return U, current_score

def construct():
    start_time = time.time()
    deadline = 110.0  # safety margin before 120s limit
    
    # Start with exhaustive search for small sets
    best_score, best_set = _exhaustive_search_small(n_max=5, max_val=15)
    
    # Try greedy construction with various parameters
    for max_u in [20, 50, 100, 200]:
        if time.time() - start_time > deadline:
            break
        for target_size in [5, 8, 12]:
            if time.time() - start_time > deadline:
                break
            U = _greedy_construct(max_u, target_size)
            s = _try_set(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
    
    # Local search from best found
    if time.time() - start_time < deadline:
        max_u_for_ls = max(best_set) + 20 if best_set else 50
        max_u_for_ls = min(max_u_for_ls, 300)
        U_ls, s_ls = _local_search(best_set, max_u_for_ls, num_iters=200)
        if s_ls > best_score:
            best_score = s_ls
            best_set = U_ls[:]
    
    # Additional local search iterations if time permits
    if time.time() - start_time < deadline - 10:
        max_u_for_ls2 = max(best_set) + 30 if best_set else 100
        max_u_for_ls2 = min(max_u_for_ls2, 500)
        U_ls2, s_ls2 = _local_search(best_set, max_u_for_ls2, num_iters=300)
        if s_ls2 > best_score:
            best_score = s_ls2
            best_set = U_ls2[:]
    
    # Ensure 0 is in the set
    if 0 not in best_set:
        best_set.append(0)
    
    return sorted(set(best_set))
# EVOLVE-BLOCK-END
