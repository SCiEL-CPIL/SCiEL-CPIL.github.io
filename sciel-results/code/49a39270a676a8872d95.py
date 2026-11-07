"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

N = 128

def _check_valid(S):
    """Check if set S has no forbidden triple."""
    Sset = set(S)
    for a, b, c in combinations(S, 3):
        if (a + b) in Sset and (a + c) in Sset and (b + c) in Sset:
            return False
    return True

def _count_violations(S):
    """Count number of forbidden triples in set S."""
    Sset = set(S)
    count = 0
    for a, b, c in combinations(S, 3):
        if (a + b) in Sset and (a + c) in Sset and (b + c) in Sset:
            count += 1
    return count

def _greedy_construct():
    """Greedy construction: add elements one by one if they don't create violations."""
    S = set()
    # Try adding in various orders
    for x in range(1, N + 1):
        S.add(x)
        if _count_violations(S) > 0:
            S.remove(x)
    return S

def _greedy_construct_desc():
    """Greedy construction in descending order (similar to seed)."""
    S = set()
    for x in range(N, 0, -1):
        S.add(x)
        if _count_violations(S) > 0:
            S.remove(x)
    return S

def _local_search(S, time_limit=55.0):
    """Local search to improve the set size."""
    start_time = time.time()
    S = set(S)
    best_S = set(S)
    best_size = len(S)
    
    # Try to add elements not in S
    candidates = [x for x in range(1, N + 1) if x not in S]
    random.shuffle(candidates)
    
    # Phase 1: Try adding single elements
    for x in candidates:
        if time.time() - start_time > time_limit:
            break
        S.add(x)
        if _count_violations(S) == 0:
            best_S = set(S)
            best_size = len(S)
        else:
            S.remove(x)
    
    # Phase 2: Try swaps (remove one, add one)
    for _ in range(2000):
        if time.time() - start_time > time_limit:
            break
        if not best_S:
            break
        # Try to find a beneficial swap
        removed = random.choice(list(best_S))
        S = set(best_S)
        S.remove(removed)
        # Try to add a different element
        for x in candidates:
            if x == removed:
                continue
            S.add(x)
            if _count_violations(S) == 0:
                best_S = set(S)
                best_size = len(S)
                break
            S.remove(x)
    
    return best_S

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    # Start with the seed solution (known valid)
    S = set(range(N // 2 + 1, N + 1))
    best_S = set(S)
    best_size = len(S)
    
    # Try greedy constructions
    for _ in range(5):
        if time.time() > deadline:
            break
        S_greedy = _greedy_construct()
        if len(S_greedy) > best_size:
            best_S = S_greedy
            best_size = len(S_greedy)
    
    # Try greedy in descending order
    if time.time() <= deadline:
        S_greedy_desc = _greedy_construct_desc()
        if len(S_greedy_desc) > best_size:
            best_S = S_greedy_desc
            best_size = len(S_greedy_desc)
    
    # Local search from best solution
    if time.time() <= deadline:
        S_improved = _local_search(best_S, time_limit=max(0, deadline - time.time()))
        if len(S_improved) > best_size:
            best_S = S_improved
            best_size = len(S_improved)
    
    # Return as list of Python ints
    return [int(x) for x in sorted(best_S)]
# EVOLVE-BLOCK-END
