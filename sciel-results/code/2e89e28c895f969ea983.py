import time
import random

N = 40
_DEADLINE = time.time() + 85

def _find_diff_basis(target_size, max_val, seed=42):
    """Find a set S of size target_size from 1..max_val-1 such that
    pairwise differences cover 1..max_val-2. Must include 1 and max_val-1."""
    if time.time() > _DEADLINE - 3:
        return None
    
    # Check if possible
    if target_size * (target_size - 1) // 2 < max_val - 2:
        return None

    target_mask = (1 << (max_val - 2)) - 1
    
    def compute_covered(S):
        s_list = sorted(S)
        mask = 0
        for i in range(len(s_list)):
            for j in range(i + 1, len(s_list)):
                mask |= 1 << (s_list[j] - s_list[i] - 1)
        return mask
    
    def count_covered(mask):
        return bin(mask & target_mask).count('1')
        
    fixed = {1, max_val - 1}
    remaining_size = target_size - len(fixed)
    candidates = [x for x in range(2, max_val - 1) if x not in fixed]
    
    best = None
    best_score = -1
    
    rng = random.Random(seed)
    
    # Greedy initialization
    def greedy_init():
        S = set(fixed)
        covered = compute_covered(S)
        for _ in range(remaining_size):
            if time.time() > _DEADLINE - 3:
                break
            best_candidate = None
            best_gain = -1
            for c in candidates:
                if c in S:
                    continue
                S.add(c)
                new_covered = compute_covered(S)
                gain = count_covered(new_covered) - count_covered(covered)
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = c
                S.remove(c)
            if best_candidate is not None:
                S.add(best_candidate)
                covered = compute_covered(S)
            else:
                avail = [c for c in candidates if c not in S]
                if avail:
                    S.add(rng.choice(avail))
                    covered = compute_covered(S)
        return S, covered

    S, covered = greedy_init()
    score = count_covered(covered)
    if score > best_score:
        best_score = score
        best = set(S)
        
    if covered & target_mask == target_mask:
        return sorted(S)
        
    # Local Search
    improved = True
    while improved and time.time() < _DEADLINE - 3:
        improved = False
        s_list = sorted(S)
        covered = compute_covered(S)
        score = count_covered(covered)
        if covered & target_mask == target_mask:
            return sorted(S)
            
        # 1-swap
        for idx in range(len(s_list)):
            old = s_list[idx]
            if old in fixed:
                continue
            for nv in candidates:
                if nv in S and nv != old:
                    continue
                S.remove(old)
                S.add(nv)
                new_covered = compute_covered(S)
                new_score = count_covered(new_covered)
                if new_score > score:
                    score = new_score
                    covered = new_covered
                    improved = True
                    if score > best_score:
                        best_score = score
                        best = set(S)
                    if covered & target_mask == target_mask:
                        return sorted(S)
                    break
                else:
                    S.remove(nv)
                    S.add(old)
                    
    if best is not None and compute_covered(best) & target_mask == target_mask:
        return sorted(best)
    return None

def construct():
    S = None
    # Try to find size 10 (optimal for this structure)
    # Size 9 is impossible as 9*8/2 = 36 < 38
    for seed in [42, 123, 999, 1, 2, 3, 7, 11, 13]:
        if time.time() > _DEADLINE - 10:
            break
        S = _find_diff_basis(10, N, seed=seed)
        if S is not None:
            break
    
    if S is None:
        # Try size 11
        for seed in [42, 123, 999]:
            if time.time() > _DEADLINE - 5:
                break
            S = _find_diff_basis(11, N, seed=seed)
            if S is not None:
                break

    if S is None:
        # Fallback to a known valid structure if search fails
        # Construct a valid S of size 10 manually or use a dense fallback
        # A simple valid S for N=40:
        # We need differences to cover 1..38.
        # Try: 1, 2, 4, 8, 16, 32, ... powers of 2?
        # Diffs of powers of 2 cover many, but not all.
        # Let's just use the fallback dense graph if search fails, but it's bad.
        # Better: Return a valid graph with more edges.
        S = list(range(1, N))
        
    S_set = set(S)
    edges = []
    edges_set = set()
    
    def add_edge(a, b):
        if a == b:
            return
        if a > b:
            a, b = b, a
        if (a, b) not in edges_set:
            edges_set.add((a, b))
            edges.append((a, b))
            
    for i in range(1, N - 1):
        add_edge(i, i + 1)
    for s in S:
        add_edge(0, s)
        
    cycles = []
    valid = True
    for L in range(3, N + 1):
        diff = L - 2
        found = False
        for u in sorted(S_set):
            v = u + diff
            if v in S_set:
                cycle = [0] + list(range(u, v + 1))
                ok = True
                for i in range(len(cycle) - 1):
                    a, b = cycle[i], cycle[i + 1]
                    if a > b:
                        a, b = b, a
                    if (a, b) not in edges_set:
                        ok = False
                        break
                if ok:
                    a, b = cycle[-1], cycle[0]
                    if a > b:
                        a, b = b, a
                    if (a, b) not in edges_set:
                        ok = False
                if ok:
                    cycles.append(cycle)
                    found = True
                    break
        if not found:
            valid = False
            break
            
    if not valid:
        e2 = [(0, i) for i in range(1, N)] + [(i, i + 1) for i in range(1, N - 1)]
        c2 = []
        for L in range(3, N + 1):
            c2.append(list(range(L)))
        return {'edges': e2, 'cycles': c2}
        
    return {'edges': edges, 'cycles': cycles}
