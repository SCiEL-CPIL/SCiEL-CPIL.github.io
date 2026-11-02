"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _compute_score(U):
    """Compute the score: 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)."""
    if not U or len(U) < 2:
        return 0.0
    U = sorted(set(U))
    max_u = U[-1]
    if max_u == 0:
        return 0.0
    # Compute U+U and U-U
    sum_set = set()
    diff_set = set()
    n = len(U)
    for i in range(n):
        for j in range(i, n):
            sum_set.add(U[i] + U[j])
        for j in range(n):
            diff_set.add(U[i] - U[j])
    # |U-U| is the size of the difference set (which includes negative values and 0)
    # |U+U| is the size of the sum set
    if len(sum_set) == 0:
        return 0.0
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 0:
        return 0.0
    denom = math.log(2 * max_u + 1)
    if denom <= 0:
        return 0.0
    return 1.0 + math.log(ratio) / denom

def _greedy_prime_step():
    """
    Greedy construction using prime-step gaps.
    U = {0, p1, p1+p2, p1+p2+p3, ...}
    At each step, choose the next prime p_k to maximize the instantaneous score.
    """
    U = [0]
    # Start with small primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    best_U = [0]
    best_score = _compute_score([0, 1]) if len([0, 1]) >= 2 else 0.0
    # Actually let's just start and build up
    best_U = [0]
    
    # We need at least 2 elements for a meaningful score, but let's build greedily
    current = [0]
    for p in primes:
        candidate = current + [current[-1] + p]
        s = _compute_score(candidate)
        if s > best_score or len(candidate) <= 2:
            best_score = s
            best_U = candidate
            current = candidate
    
    return best_U

def _local_search(U, deadline, max_iter=50000):
    """
    Local search: try replacing an element with a nearby value to improve score.
    """
    U = sorted(set(U))
    if len(U) < 2:
        return U
    
    best_U = U[:]
    best_score = _compute_score(best_U)
    
    start_time = time.time()
    rng = random.Random(42)
    
    for it in range(max_iter):
        if time.time() > deadline - 15:
            break
        
        # Pick a random index to modify (not 0)
        idx = rng.randint(1, len(best_U) - 1)
        old_val = best_U[idx]
        
        # Try a few perturbations
        improved = False
        for delta in rng.sample(range(-50, 51), min(20, 101)):
            if delta == 0:
                continue
            new_val = old_val + delta
            if new_val < 0:
                continue
            # Ensure uniqueness and ordering
            new_U = best_U[:]
            new_U[idx] = new_val
            new_U = sorted(set(new_U))
            # Check 0 is still present
            if 0 not in new_U:
                continue
            # Check no duplicates after sort
            if len(new_U) != len(new_U):
                continue
            
            s = _compute_score(new_U)
            if s > best_score + 1e-12:
                best_score = s
                best_U = new_U
                idx = rng.randint(1, len(best_U) - 1)
                old_val = best_U[idx]
                improved = True
                break
        
        if not improved and it % 1000 == 0:
            # Occasionally try a larger perturbation
            idx = rng.randint(1, len(best_U) - 1)
            old_val = best_U[idx]
            for delta in rng.sample(range(-200, 201), 30):
                new_val = old_val + delta
                if new_val < 0:
                    continue
                new_U = best_U[:]
                new_U[idx] = new_val
                new_U = sorted(set(new_U))
                if 0 not in new_U:
                    continue
                s = _compute_score(new_U)
                if s > best_score + 1e-12:
                    best_score = s
                    best_U = new_U
                    break
    
    return best_U

def _add_element(U, deadline, max_candidates=200):
    """Try adding a new element to U to improve score."""
    U = sorted(set(U))
    best_U = U[:]
    best_score = _compute_score(best_U)
    
    max_u = U[-1] if U else 0
    # Try adding elements in a range
    candidates = set()
    # Try values near existing elements and in gaps
    for u in U:
        for delta in range(-5, 6):
            v = u + delta
            if v > 0:  # 0 already present
                candidates.add(v)
    # Try some random values
    for _ in range(100):
        v = random.randint(1, max_u + 50)
        candidates.add(v)
    
    candidates = sorted(candidates)
    
    for v in candidates:
        if v in U:
            continue
        new_U = sorted(U + [v])
        s = _compute_score(new_U)
        if s > best_score + 1e-12:
            best_score = s
            best_U = new_U
    
    return best_U

def _remove_element(U, deadline):
    """Try removing an element from U to improve score."""
    U = sorted(set(U))
    if len(U) <= 2:
        return U
    
    best_U = U[:]
    best_score = _compute_score(best_U)
    
    for i in range(1, len(U)):  # Don't remove 0
        new_U = U[:i] + U[i+1:]
        s = _compute_score(new_U)
        if s > best_score + 1e-12:
            best_score = s
            best_U = new_U
            # Don't break; try further removals
            # But this changes the length, so restart
            break
    
    return best_U

def construct():
    deadline = time.time() + 130  # Safety margin from 150s
    
    # Start with greedy prime-step construction
    U = _greedy_prime_step()
    
    # Try different starting primes
    # Also try superincreasing-like sequences
    start_variants = []
    # Prime-based
    start_variants.append(_greedy_prime_step())
    
    # Try a few different initial sequences
    for init in [[0, 1], [0, 2], [0, 3], [0, 1, 2], [0, 1, 3], [0, 2, 5]]:
        if time.time() > deadline - 60:
            break
        # Extend greedily
        current = list(init)
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            candidate = current + [current[-1] + p]
            start_variants.append(candidate)
    
    # Pick the best starting variant
    best_start = None
    best_start_score = -1
    for sv in start_variants:
        s = _compute_score(sv)
        if s > best_start_score:
            best_start_score = s
            best_start = sv
    
    U = best_start if best_start else [0, 1, 3]
    
    # Local search
    if time.time() < deadline - 30:
        U = _local_search(U, deadline)
    
    # Try adding elements
    if time.time() < deadline - 20:
        U = _add_element(U, deadline)
    
    # Try removing elements
    if time.time() < deadline - 10:
        U = _remove_element(U, deadline)
    
    # Another local search pass
    if time.time() < deadline - 15:
        U = _local_search(U, deadline, max_iter=20000)
    
    # Ensure 0 is in the set and all are nonnegative
    U = sorted(set([x for x in U if x >= 0]))
    if 0 not in U:
        U = [0] + U
    
    return U

# EVOLVE-BLOCK-END
