"""B_3[1] set construction: maximize cardinality with unique unordered triple sums."""
import time
import random
from itertools import combinations_with_replacement

N_MAX = 4096
TARGET = 80
DEADLINE_SECONDS = 85  # safety margin before 90s kill

def _triple_sums_set(S):
    """Return the set of all unordered triple sums (with repetition) for sorted list S."""
    n = len(S)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                sums.add(S[i] + S[j] + S[k])
    return sums

def _new_sums_involving(x, S):
    """Compute all triple sums that involve x, given existing sorted set S.
    Returns (new_sums_set, collision_bool).
    The new sums are those triples that include at least one x.
    We check if any of these are already in the existing sums set."""
    n = len(S)
    new_sums = set()
    
    # Triples involving x:
    # (x, x, x)
    new_sums.add(3 * x)
    # (x, x, s_i) for each s_i in S
    for s in S:
        new_sums.add(2 * x + s)
    # (x, s_i, s_j) for s_i <= s_j in S
    for i in range(n):
        for j in range(i, n):
            new_sums.add(x + S[i] + S[j])
    
    return new_sums

def _check_add(x, S, existing_sums):
    """Check if adding x to S keeps all triple sums unique.
    Returns (new_existing_sums, ok) where new_existing_sums is the updated set."""
    new_sums = _new_sums_involving(x, S)
    # Check for collisions with existing sums
    for s in new_sums:
        if s in existing_sums:
            return existing_sums, False
    # No collision; merge
    existing_sums = existing_sums | new_sums
    return existing_sums, True

def _greedy_linear(deadline, start_seed=None):
    """Greedy: iterate x from 1 to N_MAX, add if no collision."""
    if start_seed is not None:
        random.seed(start_seed)
    
    S = []
    existing_sums = set()
    
    for x in range(1, N_MAX + 1):
        if time.time() > deadline:
            break
        if len(S) >= TARGET:
            break
        new_sums, ok = _check_add(x, S, existing_sums)
        if ok:
            S.append(x)
            existing_sums = new_sums
    
    return S

def _greedy_random(deadline, seed):
    """Greedy with randomized order."""
    random.seed(seed)
    candidates = list(range(1, N_MAX + 1))
    random.shuffle(candidates)
    
    S = []
    existing_sums = set()
    
    for x in candidates:
        if time.time() > deadline:
            break
        if len(S) >= TARGET:
            break
        new_sums, ok = _check_add(x, S, existing_sums)
        if ok:
            S.append(x)
            existing_sums = new_sums
    
    return S

def _local_search_add(S, existing_sums, deadline):
    """Try to add more elements by trying values not yet in S.
    We scan through possible values and try to insert them."""
    S = list(S)
    S.sort()
    existing_sums = set(existing_sums)
    
    # Try to add more elements
    used = set(S)
    for x in range(1, N_MAX + 1):
        if time.time() > deadline:
            break
        if len(S) >= TARGET:
            break
        if x in used:
            continue
        new_sums, ok = _check_add(x, S, existing_sums)
        if ok:
            S.append(x)
            S.sort()
            used.add(x)
            existing_sums = new_sums
    
    return S

def _local_search_remove_add(S, existing_sums, deadline):
    """Try removing one element and adding a different one to potentially fit more."""
    S = list(S)
    S.sort()
    n = len(S)
    
    # Recompute sums from scratch for safety
    existing_sums = _triple_sums_set(S)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            if len(S) >= TARGET:
                break
            # Try removing S[i]
            S_without = S[:i] + S[i+1:]
            sums_without = _triple_sums_set(S_without)
            
            # Try adding various x
            for x in range(1, N_MAX + 1):
                if x in S_without:
                    continue
                new_sums, ok = _check_add(x, S_without, sums_without)
                if ok:
                    S = S_without + [x]
                    S.sort()
                    existing_sums = new_sums
                    improved = True
                    break
            if improved:
                break
        n = len(S)
    
    return S

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # Phase 1: Linear greedy (deterministic, fast)
    S = _greedy_linear(deadline)
    
    if len(S) >= TARGET:
        return S
    
    # Phase 2: Try several random greedy starts
    best_S = S
    best_len = len(S)
    
    seeds = [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    
    for seed in seeds:
        if time.time() > deadline:
            break
        S_rand = _greedy_random(deadline, seed)
        if len(S_rand) > best_len:
            best_S = S_rand
            best_len = len(S_rand)
            if best_len >= TARGET:
                return best_S
    
    # Phase 3: Local search on the best solution
    if time.time() < deadline:
        S_local = _local_search_add(best_S, _triple_sums_set(best_S), deadline)
        if len(S_local) > best_len:
            best_S = S_local
            best_len = len(S_local)
    
    # Phase 4: More aggressive local search if time permits
    if time.time() < deadline - 5:
        S_local2 = _local_search_remove_add(best_S, _triple_sums_set(best_S), deadline)
        if len(S_local2) > best_len:
            best_S = S_local2
            best_len = len(S_local2)
    
    return best_S

# EVOLVE-BLOCK-END
