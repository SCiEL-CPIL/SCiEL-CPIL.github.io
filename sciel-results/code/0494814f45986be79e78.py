"""Optimized program: greedy with local search improvement."""
# EVOLVE-BLOCK-START
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def _is_bad_triple(a, b, c):
    return _lcm(a, b) == _lcm(a, c) == _lcm(b, c)

def _has_bad_triple_in_set(S):
    S_list = list(S)
    for i in range(len(S_list)):
        for j in range(i+1, len(S_list)):
            for k in range(j+1, len(S_list)):
                if _is_bad_triple(S_list[i], S_list[j], S_list[k]):
                    return True
    return False

def _greedy_select(order):
    selected = []
    for x in order:
        ok = True
        for a, b in zip(selected, selected[1:]):
            pass
        # Check all pairs in selected
        for i in range(len(selected)):
            for j in range(i+1, len(selected)):
                if _is_bad_triple(selected[i], selected[j], x):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            selected.append(x)
    return selected

def _local_search(S, n=128, max_iters=5000):
    S = set(S)
    current_size = len(S)
    
    for _ in range(max_iters):
        improved = False
        # Try to add a vertex
        for x in range(1, n+1):
            if x in S:
                continue
            ok = True
            S_list = list(S)
            for i in range(len(S_list)):
                for j in range(i+1, len(S_list)):
                    if _is_bad_triple(S_list[i], S_list[j], x):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                S.add(x)
                current_size += 1
                improved = True
                break
        
        if not improved:
            # Try swap: remove one, add one
            S_list = list(S)
            for i in range(len(S_list)):
                for x in range(1, n+1):
                    if x in S:
                        continue
                    S.remove(S_list[i])
                    S.add(x)
                    if not _has_bad_triple_in_set(S):
                        improved = True
                        break
                    S.remove(x)
                    S.add(S_list[i])
                if improved:
                    break
            if improved:
                continue
        
        if not improved:
            break
    
    return list(S)

def construct():
    n = 128
    # Try multiple orderings
    best = []
    
    # Natural order
    S1 = _greedy_select(range(1, n+1))
    S1 = _local_search(S1, n, max_iters=2000)
    if len(S1) > len(best):
        best = S1
    
    # Reverse order
    S2 = _greedy_select(range(n, 0, -1))
    S2 = _local_search(S2, n, max_iters=2000)
    if len(S2) > len(best):
        best = S2
    
    # Powers of 2 first, then others
    order = [2**k for k in range(0, 8) if 2**k <= n]
    order += [x for x in range(1, n+1) if x not in order]
    S3 = _greedy_select(order)
    S3 = _local_search(S3, n, max_iters=2000)
    if len(S3) > len(best):
        best = S3
    
    return best
# EVOLVE-BLOCK-END
