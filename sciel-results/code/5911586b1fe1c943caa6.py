"""Improved program: conflict-ordered greedy + multi-scale local search (1-for-k, 2-for-k, 3-for-k)."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _conflict(a, b):
    """Return True if (a*b) % (a+b) == 0 (i.e., a and b conflict)."""
    return (a * b) % (a + b) == 0

def _conflict_count(a, S):
    """Count how many elements in S conflict with a."""
    cnt = 0
    for b in S:
        if b != a and _conflict(a, b):
            cnt += 1
    return cnt

def _valid(S):
    """Check that S is a valid set (no conflicting pairs)."""
    S = list(S)
    n = len(S)
    for i in range(n):
        for j in range(i + 1, n):
            if _conflict(S[i], S[j]):
                return False
    return True

def _greedy_build():
    """
    Conflict-count-ordered greedy: repeatedly pick the element with fewest
    conflicts against the current set, breaking ties by value.
    """
    S = set()
    remaining = set(range(1, 257))
    while remaining:
        # Find element with minimum conflict count against S
        best_elem = None
        best_cnt = float('inf')
        for a in remaining:
            cnt = _conflict_count(a, S)
            if cnt < best_cnt or (cnt == best_cnt and (best_elem is None or a < best_elem)):
                best_cnt = cnt
                best_elem = a
        S.add(best_elem)
        remaining.discard(best_elem)
    return S

def _local_search_1for1(S, max_rounds=50):
    """Try removing one element and adding one other (1-for-1 swap)."""
    S = set(S)
    for _ in range(max_rounds):
        improved = False
        for a in list(S):
            S.remove(a)
            for b in range(1, 257):
                if b not in S and b != a and not any(_conflict(b, x) for x in S):
                    S.add(b)
                    improved = True
                    break
            if not improved:
                S.add(a)
            if improved:
                break
        if not improved:
            break
    return S

def _local_search_1for2(S, max_rounds=30):
    """Try removing one element and adding two others (1-for-2)."""
    S = set(S)
    for _ in range(max_rounds):
        improved = False
        for a in list(S):
            S.remove(a)
            # Try all pairs (b, c) not in S, b < c
            for b in range(1, 257):
                if b in S or b == a:
                    continue
                if any(_conflict(b, x) for x in S):
                    continue
                for c in range(b + 1, 257):
                    if c in S or c == a:
                        continue
                    if any(_conflict(c, x) for x in S) or _conflict(b, c):
                        continue
                    S.add(b)
                    S.add(c)
                    improved = True
                    break
                if improved:
                    break
            if not improved:
                S.add(a)
            if improved:
                break
        if not improved:
            break
    return S

def _local_search_2for3(S, max_rounds=20):
    """Try removing two elements and adding three others (2-for-3)."""
    S = set(S)
    for _ in range(max_rounds):
        improved = False
        elems = list(S)
        n = len(elems)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = elems[i], elems[j]
                S.remove(a)
                S.remove(b)
                # Try all triples (c, d, e)
                for c in range(1, 257):
                    if c in S or c == a or c == b:
                        continue
                    if any(_conflict(c, x) for x in S):
                        continue
                    for d in range(c + 1, 257):
                        if d in S or d == a or d == b:
                            continue
                        if any(_conflict(d, x) for x in S) or _conflict(c, d):
                            continue
                        for e in range(d + 1, 257):
                            if e in S or e == a or e == b:
                                continue
                            if any(_conflict(e, x) for x in S) or _conflict(c, e) or _conflict(d, e):
                                continue
                            S.add(c)
                            S.add(d)
                            S.add(e)
                            improved = True
                            break
                        if improved:
                            break
                    if improved:
                        break
                if not improved:
                    S.add(a)
                    S.add(b)
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return S

def _local_search_3for4(S, max_rounds=10):
    """Try removing three elements and adding four others (3-for-4)."""
    S = set(S)
    for _ in range(max_rounds):
        improved = False
        elems = list(S)
        n = len(elems)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    a, b, c = elems[i], elems[j], elems[k]
                    S.remove(a)
                    S.remove(b)
                    S.remove(c)
                    # Try all quadruples (d, e, f, g)
                    for d in range(1, 257):
                        if d in S or d in (a, b, c):
                            continue
                        if any(_conflict(d, x) for x in S):
                            continue
                        for e in range(d + 1, 257):
                            if e in S or e in (a, b, c):
                                continue
                            if any(_conflict(e, x) for x in S) or _conflict(d, e):
                                continue
                            for f in range(e + 1, 257):
                                if f in S or f in (a, b, c):
                                    continue
                                if any(_conflict(f, x) for x in S) or _conflict(d, f) or _conflict(e, f):
                                    continue
                                for g in range(f + 1, 257):
                                    if g in S or g in (a, b, c):
                                        continue
                                    if (any(_conflict(g, x) for x in S) or _conflict(d, g) or
                                        _conflict(e, g) or _conflict(f, g)):
                                        continue
                                    S.add(d)
                                    S.add(e)
                                    S.add(f)
                                    S.add(g)
                                    improved = True
                                    break
                                if improved:
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if not improved:
                        S.add(a)
                        S.add(b)
                        S.add(c)
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return S

def _randomized_rebuild(rng, S):
    """Randomized perturbation: remove a few random elements, rebuild greedily."""
    S = set(S)
    n_remove = min(3, len(S))
    to_remove = rng.sample(list(S), n_remove)
    for x in to_remove:
        S.discard(x)
    # Greedily add back
    remaining = set(range(1, 257)) - S
    while remaining:
        best_elem = None
        best_cnt = float('inf')
        for a in remaining:
            cnt = _conflict_count(a, S)
            if cnt < best_cnt or (cnt == best_cnt and (best_elem is None or rng.random() < 0.5)):
                best_cnt = cnt
                best_elem = a
        S.add(best_elem)
        remaining.discard(best_elem)
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin within 90s limit
    
    best = set()
    
    # Phase 1: Conflict-ordered greedy
    S = _greedy_build()
    best = set(S)
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return list(best)
    
    # Phase 2: Multi-scale local search
    # 1-for-1
    S = _local_search_1for1(S)
    if len(S) > len(best):
        best = set(S)
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return list(best)
    
    # 1-for-2
    S = _local_search_1for2(S)
    if len(S) > len(best):
        best = set(S)
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return list(best)
    
    # 2-for-3
    S = _local_search_2for3(S)
    if len(S) > len(best):
        best = set(S)
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return list(best)
    
    # 3-for-4
    S = _local_search_3for4(S)
    if len(S) > len(best):
        best = set(S)
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return list(best)
    
    # Phase 3: Iterated local search with randomization
    rng = random.Random(42)
    current = set(best)
    
    while time.time() - start_time < deadline - 5:
        # Perturb
        perturbed = _randomized_rebuild(rng, current)
        
        # Local search on perturbed
        perturbed = _local_search_1for1(perturbed, max_rounds=30)
        perturbed = _local_search_1for2(perturbed, max_rounds=20)
        perturbed = _local_search_2for3(perturbed, max_rounds=15)
        perturbed = _local_search_3for4(perturbed, max_rounds=8)
        
        if len(perturbed) > len(current):
            current = perturbed
            if len(current) > len(best):
                best = set(current)
        
        # Check time
        if time.time() - start_time > deadline - 5:
            break
    
    # Final polish
    if time.time() - start_time < deadline - 2:
        final = _local_search_1for1(best, max_rounds=50)
        final = _local_search_1for2(final, max_rounds=30)
        final = _local_search_2for3(final, max_rounds=20)
        if len(final) > len(best):
            best = final
    
    # Ensure nonempty
    if not best:
        best = {1}
    
    return list(best)

# EVOLVE-BLOCK-END
