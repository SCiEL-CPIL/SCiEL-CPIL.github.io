import time
from itertools import combinations

def _smooth_numbers(limit):
    s = set()
    a = 1
    while a <= limit:
        b = a
        while b <= limit:
            c = b
            while c <= limit:
                s.add(c)
                c *= 5
            b *= 3
        a *= 2
    return sorted(s)

_SMOOTH = _smooth_numbers(128)
_SMOOTH_SET = set(_SMOOTH)
_N = len(_SMOOTH)
_SMOOTH_IDX = {v: i for i, v in enumerate(_SMOOTH)}

# Precompute incompatibility: for each index, set of incompatible indices
_incompat = [set() for _ in range(_N)]
for i in range(_N):
    for j in range(i + 1, _N):
        a, b = _SMOOTH[i], _SMOOTH[j]
        if a % b == 0 or b % a == 0:
            _incompat[i].add(j)
            _incompat[j].add(i)

# Precompute compatible pairs for fast lookup
_compatible_pair = [[True] * _N for _ in range(_N)]
for i in range(_N):
    for j in range(_N):
        if i != j and j in _incompat[i]:
            _compatible_pair[i][j] = False

# Precompute sum tables for k=2, k=3, k=4 to speed up search
# For k=2: sum2[i][j] = SMOOTH[i] + SMOOTH[j] for i < j
# We'll use a dict mapping sum -> list of (i, j) pairs
_sum2_pairs = {}
for i in range(_N):
    for j in range(i + 1, _N):
        s = _SMOOTH[i] + _SMOOTH[j]
        if s <= 128:
            if s not in _sum2_pairs:
                _sum2_pairs[s] = []
            _sum2_pairs[s].append((i, j))

# For k=3: sum3[i][j][k] = SMOOTH[i] + SMOOTH[j] + SMOOTH[k] for i < j < k
# We'll use a dict mapping sum -> list of (i, j, k) triples
_sum3_triples = {}
for i in range(_N):
    for j in range(i + 1, _N):
        for k in range(j + 1, _N):
            s = _SMOOTH[i] + _SMOOTH[j] + _SMOOTH[k]
            if s <= 128:
                if s not in _sum3_triples:
                    _sum3_triples[s] = []
                _sum3_triples[s].append((i, j, k))

# For k=4: sum4[i][j][k][l] = SMOOTH[i] + SMOOTH[j] + SMOOTH[k] + SMOOTH[l] for i < j < k < l
# This might be too large, so we'll use a more targeted approach
# Let's precompute for k=4 as well if the number of combinations is manageable
_sum4_quads = {}
for i in range(_N):
    for j in range(i + 1, _N):
        for k in range(j + 1, _N):
            for l in range(k + 1, _N):
                s = _SMOOTH[i] + _SMOOTH[j] + _SMOOTH[k] + _SMOOTH[l]
                if s <= 128:
                    if s not in _sum4_quads:
                        _sum4_quads[s] = []
                    _sum4_quads[s].append((i, j, k, l))

def _is_antichain_indices(indices):
    """Check if the given indices form an antichain."""
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            if indices[i] in _incompat[indices[j]]:
                return False
    return True

def _find_min_antichain(target, max_terms=30, deadline=None):
    """Find minimum number of distinct smooth numbers forming antichain summing to target."""
    if deadline is not None and time.time() > deadline:
        return None
    
    # k=1
    if target in _SMOOTH_SET:
        return [target]
    
    # k=2: use precomputed pairs
    if target in _sum2_pairs:
        for (i, j) in _sum2_pairs[target]:
            if i not in _incompat[j]:
                return [_SMOOTH[i], _SMOOTH[j]]
    
    # k=3: use precomputed triples
    if target in _sum3_triples:
        for (i, j, k) in _sum3_triples[target]:
            if _is_antichain_indices([i, j, k]):
                return [_SMOOTH[i], _SMOOTH[j], _SMOOTH[k]]
    
    # k=4: use precomputed quads
    if target in _sum4_quads:
        for (i, j, k, l) in _sum4_quads[target]:
            if _is_antichain_indices([i, j, k, l]):
                return [_SMOOTH[i], _SMOOTH[j], _SMOOTH[k], _SMOOTH[l]]
    
    # k=5 via backtracking with pruning
    if deadline is None or time.time() < deadline:
        found = [None]
        
        def backtrack5(start, remaining, count, chosen):
            if deadline is not None and time.time() > deadline:
                return
            if count == 5:
                if remaining == 0:
                    found[0] = list(chosen)
                    return
                return
            if remaining <= 0:
                return
            # Prune: need (5-count) numbers each >= _SMOOTH[start]
            min_possible = _SMOOTH[start] * (5 - count)
            if min_possible > remaining:
                return
            for idx in range(start, _N):
                if deadline is not None and time.time() > deadline:
                    return
                v = _SMOOTH[idx]
                if v > remaining:
                    break
                # Check compatibility
                compat = True
                for ci in chosen:
                    if idx in _incompat[ci]:
                        compat = False
                        break
                if not compat:
                    continue
                chosen.append(idx)
                backtrack5(idx + 1, remaining - v, count + 1, chosen)
                chosen.pop()
                if found[0] is not None:
                    return
        
        backtrack5(0, target, 0, [])
        if found[0] is not None:
            return [_SMOOTH[i] for i in found[0]]
    
    # k=6 via backtracking
    if deadline is None or time.time() < deadline:
        found = [None]
        
        def backtrack6(start, remaining, count, chosen):
            if deadline is not None and time.time() > deadline:
                return
            if count == 6:
                if remaining == 0:
                    found[0] = list(chosen)
                    return
                return
            if remaining <= 0:
                return
            min_possible = _SMOOTH[start] * (6 - count)
            if min_possible > remaining:
                return
            for idx in range(start, _N):
                if deadline is not None and time.time() > deadline:
                    return
                v = _SMOOTH[idx]
                if v > remaining:
                    break
                compat = True
                for ci in chosen:
                    if idx in _incompat[ci]:
                        compat = False
                        break
                if not compat:
                    continue
                chosen.append(idx)
                backtrack6(idx + 1, remaining - v, count + 1, chosen)
                chosen.pop()
                if found[0] is not None:
                    return
        
        backtrack6(0, target, 0, [])
        if found[0] is not None:
            return [_SMOOTH[i] for i in found[0]]
    
    # Fallback: try larger k with simple search
    if deadline is None or time.time() < deadline:
        for k in range(7, 12):
            smooth_leq = [v for v in _SMOOTH if v <= target]
            if len(smooth_leq) < k:
                continue
            for terms in combinations(smooth_leq, k):
                if sum(terms) != target:
                    continue
                valid = True
                for p in range(len(terms)):
                    for q in range(p + 1, len(terms)):
                        if terms[p] % terms[q] == 0 or terms[q] % terms[p] == 0:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    return list(terms)
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    results = []
    for n in range(1, 129):
        if time.time() > deadline:
            # Emergency fallback
            for m in range(n, 129):
                ans = _find_min_antichain(m, deadline)
                if ans is None:
                    # Try to find any valid representation
                    if m in _SMOOTH_SET:
                        ans = [m]
                    else:
                        # Find any antichain summing to m
                        found = False
                        for k in range(2, 10):
                            smooth_leq = [v for v in _SMOOTH if v <= m]
                            if len(smooth_leq) >= k:
                                for terms in combinations(smooth_leq, k):
                                    if sum(terms) == m:
                                        valid = True
                                        for p in range(len(terms)):
                                            for q in range(p + 1, len(terms)):
                                                if terms[p] % terms[q] == 0 or terms[q] % terms[p] == 0:
                                                    valid = False
                                                    break
                                            if not valid:
                                                break
                                        if valid:
                                            ans = list(terms)
                                            found = True
                                            break
                            if found:
                                break
                        if not found:
                            ans = [m] if m in _SMOOTH_SET else [1, m - 1]
                results.append(ans)
            return results
        
        ans = _find_min_antichain(n, deadline)
        if ans is None:
            # Last resort fallback
            if n in _SMOOTH_SET:
                ans = [n]
            else:
                # Try to find any valid representation
                found = False
                for k in range(2, 15):
                    smooth_leq = [v for v in _SMOOTH if v <= n]
                    if len(smooth_leq) >= k:
                        for terms in combinations(smooth_leq, k):
                            if sum(terms) == n:
                                valid = True
                                for p in range(len(terms)):
                                    for q in range(p + 1, len(terms)):
                                        if terms[p] % terms[q] == 0 or terms[q] % terms[p] == 0:
                                            valid = False
                                            break
                                    if not valid:
                                        break
                                if valid:
                                    ans = list(terms)
                                    found = True
                                    break
                    if found:
                        break
                if not found:
                    ans = [n] if n in _SMOOTH_SET else [1, n - 1]
        results.append(ans)
    
    return results
