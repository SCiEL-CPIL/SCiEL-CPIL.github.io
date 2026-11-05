import time
import random

N = 40
_DEADLINE = time.time() + 85

def _compute_covered_mask(S, max_val):
    """Compute bitmask of covered differences."""
    s_list = sorted(S)
    mask = 0
    for i in range(len(s_list)):
        for j in range(i + 1, len(s_list)):
            d = s_list[j] - s_list[i]
            if 1 <= d <= max_val - 2:
                mask |= (1 << (d - 1))
    return mask

def _count_covered(mask, target_mask):
    return bin(mask & target_mask).count('1')

def _find_diff_basis(target_size, max_val, seed=42, time_budget=None):
    """Find a set S of size target_size from 1..max_val-1 such that
    pairwise differences cover 1..max_val-2. Must include 1 and max_val-1."""
    if time_budget is None:
        time_budget = _DEADLINE - 3
    
    if time.time() > time_budget:
        return None
    
    target_mask = (1 << (max_val - 2)) - 1
    fixed = {1, max_val - 1}
    remaining_size = target_size - len(fixed)
    candidates = [x for x in range(2, max_val - 1) if x not in fixed]
    
    rng = random.Random(seed)
    
    # Try multiple random initializations with greedy completion
    best_S = None
    best_score = -1
    
    for attempt in range(5):
        if time.time() > time_budget:
            break
        
        S = set(fixed)
        if attempt == 0:
            # Greedy: pick elements that maximize new coverage
            covered = _compute_covered_mask(S, max_val)
            for _ in range(remaining_size):
                if time.time() > time_budget:
                    break
                best_c = None
                best_gain = -1
                for c in candidates:
                    if c in S:
                        continue
                    S.add(c)
                    new_covered = _compute_covered_mask(S, max_val)
                    gain = _count_covered(new_covered, target_mask) - _count_covered(covered, target_mask)
                    if gain > best_gain:
                        best_gain = gain
                        best_c = c
                    S.remove(c)
                if best_c is not None:
                    S.add(best_c)
                    covered = _compute_covered_mask(S, max_val)
                else:
                    avail = [c for c in candidates if c not in S]
                    if avail:
                        S.add(rng.choice(avail))
                        covered = _compute_covered_mask(S, max_val)
        else:
            # Random initialization
            for c in rng.sample(candidates, min(remaining_size, len(candidates))):
                S.add(c)
        
        score = _count_covered(_compute_covered_mask(S, max_val), target_mask)
        if score > best_score:
            best_score = score
            best_S = set(S)
        
        if score == max_val - 2:
            return sorted(S)
    
    if best_S is None:
        return None
    
    S = best_S
    # Local search with 1-swap moves
    improved = True
    while improved and time.time() < time_budget:
        improved = False
        s_list = sorted(S)
        covered = _compute_covered_mask(S, max_val)
        score = _count_covered(covered, target_mask)
        
        if score == max_val - 2:
            return sorted(S)
        
        # 1-swap moves
        for idx in range(len(s_list)):
            if time.time() > time_budget:
                break
            old = s_list[idx]
            if old in fixed:
                continue
            for nv in candidates:
                if nv in S:
                    continue
                S.remove(old)
                S.add(nv)
                new_covered = _compute_covered_mask(S, max_val)
                new_score = _count_covered(new_covered, target_mask)
                if new_score > score:
                    score = new_score
                    improved = True
                    if score > best_score:
                        best_score = score
                        best_S = set(S)
                    if score == max_val - 2:
                        return sorted(S)
                    break
                else:
                    S.remove(nv)
                    S.add(old)
    
    if best_S is not None and _count_covered(_compute_covered_mask(best_S, max_val), target_mask) == max_val - 2:
        return sorted(best_S)
    return None

def construct():
    S = None
    
    # Minimum size: need C(k,2) >= 38, so k >= 10
    for size in [10, 11, 12]:
        if time.time() > _DEADLINE - 10:
            break
        for seed in [42, 123, 999, 1, 2, 3, 7, 13]:
            if time.time() > _DEADLINE - 5:
                break
            S = _find_diff_basis(size, N, seed=seed)
            if S is not None:
                break
        if S is not None:
            break
    
    if S is None:
        # Fallback: use all vertices 1..39
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
