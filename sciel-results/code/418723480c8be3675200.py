"""B_3[1] set construction: maximize cardinality with unique unordered triple sums."""
import time
import random

DEADLINE = 55.0  # safety margin before 60s

def _triple_sums_involving(x, S):
    """Compute all triple sums that include x, given existing sorted set S."""
    n = len(S)
    sums = [3 * x]
    for s in S:
        sums.append(2 * x + s)
        sums.append(x + 2 * s)
    for i in range(n):
        for j in range(i + 1, n):
            sums.append(x + S[i] + S[j])
    return sums

def _greedy_build(candidates, max_size=800):
    """Greedy: try each candidate in order, add if no collision."""
    S = []
    sum_set = set()
    for x in candidates:
        if x in sum_set:  # quick check: x itself might collide as part of triple
            pass
        new_sums = _triple_sums_involving(x, S)
        collision = False
        for s in new_sums:
            if s in sum_set:
                collision = True
                break
        if not collision:
            S.append(x)
            for s in new_sums:
                sum_set.add(s)
            if len(S) >= max_size:
                break
    return S

def _greedy_linear(start=1, end=4097):
    """Greedy scanning linearly from start to end."""
    return _greedy_build(range(start, end))

def _greedy_exponential(base=3):
    """Greedy with exponential spacing to pack more."""
    candidates = []
    x = 1
    while x <= 4096:
        candidates.append(x)
        x = x * base + 1
    return _greedy_build(candidates)

def _random_greedy(seed_val=42, n_candidates=5000):
    """Random candidate order greedy."""
    rng = random.Random(seed_val)
    candidates = list(range(1, 4097))
    rng.shuffle(candidates)
    # Try to build a good set
    S = []
    sum_set = set()
    for x in candidates:
        new_sums = _triple_sums_involving(x, S)
        collision = False
        for s in new_sums:
            if s in sum_set:
                collision = True
                break
        if not collision:
            S.append(x)
            for s in new_sums:
                sum_set.add(s)
    return S

def _verify(S):
    """Verify all unordered triples have unique sums."""
    from itertools import combinations_with_replacement
    sums = [sum(t) for t in combinations_with_replacement(S, 3)]
    return len(sums) == len(set(sums))

def construct():
    start_time = time.time()
    
    best = []
    
    # Strategy 1: Linear greedy from 1
    S1 = _greedy_linear(1, 4097)
    if len(S1) > len(best):
        best = S1
    if time.time() - start_time > DEADLINE - 5:
        return best
    
    # Strategy 2: Exponential greedy
    for base in [2, 3, 4, 5, 7, 10]:
        S2 = _greedy_exponential(base)
        if len(S2) > len(best):
            best = S2
        if time.time() - start_time > DEADLINE - 5:
            return best
    
    # Strategy 3: Random greedy with different seeds
    for seed_val in [1, 7, 42, 123, 999, 2024, 31415, 271828]:
        S3 = _random_greedy(seed_val)
        if len(S3) > len(best):
            best = S3
        if time.time() - start_time > DEADLINE - 3:
            break
    
    # Strategy 4: Hybrid - start with exponential, then fill gaps linearly
    S4_base = _greedy_exponential(3)
    S4 = list(S4_base)
    sum_set4 = set()
    for s in _triple_sums_involving(S4_base[-1], S4_base[:-1]):
        sum_set4.add(s)
    # Recompute all sums
    sum_set4 = set()
    for x in S4:
        for s in _triple_sums_involving(x, [y for y in S4 if y != x]):
            sum_set4.add(s)
    # Simpler: just rebuild
    S4 = []
    sum_set4 = set()
    # First add exponential candidates
    exp_candidates = []
    x = 1
    while x <= 4096:
        exp_candidates.append(x)
        x = x * 3 + 1
    for x in exp_candidates:
        new_sums = _triple_sums_involving(x, S4)
        if not any(s in sum_set4 for s in new_sums):
            S4.append(x)
            for s in new_sums:
                sum_set4.add(s)
    # Then fill in linearly
    for x in range(1, 4097):
        if x in S4:
            continue
        new_sums = _triple_sums_involving(x, S4)
        if not any(s in sum_set4 for s in new_sums):
            S4.append(x)
            for s in new_sums:
                sum_set4.add(s)
    if len(S4) > len(best):
        best = S4
    
    if time.time() - start_time > DEADLINE - 2:
        return best
    
    # Final verification of best
    if best and not _verify(best):
        # Fallback to a verified smaller set
        S_fallback = _greedy_linear(1, 4097)
        if _verify(S_fallback):
            best = S_fallback
    
    return best
