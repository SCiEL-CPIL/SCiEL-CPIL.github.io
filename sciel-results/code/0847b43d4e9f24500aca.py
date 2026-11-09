import time
import random
import numpy as np
from itertools import combinations

N = 192
MAX_K = 3
MAX_SIZE = 48
MAX_SUM = 3 * N  # 576

def _precompute():
    """Precompute divisibility and pair sums for fast conflict checking."""
    # divs[a] = set of multiples of a up to MAX_SUM
    divs = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for m in range(a, MAX_SUM + 1, a):
            divs[a].add(m)
    
    # pair_sums: for each pair (i,j) with i<j, store i+j
    # We'll use a 2D array for quick lookup
    pair_sum = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            pair_sum[i][j] = i + j
            pair_sum[j][i] = i + j
    
    return divs, pair_sum

DIVS, PAIR_SUM = _precompute()

def _conflict_count(x, selected_set):
    """Count how many elements in selected_set conflict with x.
    Conflict: x divides (a+b+c) for some 3 distinct others, OR a divides (x+b+c) for some a in selected, b,c in selected\\{a}."""
    sel = list(selected_set)
    m = len(sel)
    if m < 3:
        return 0
    
    count = 0
    
    # x divides sum of any 3 from selected
    for i in range(m):
        for j in range(i + 1, m):
            for k in range(j + 1, m):
                s = sel[i] + sel[j] + sel[k]
                if s in DIVS[x]:
                    count += 1
    
    # For each a in selected, a divides (x + b + c) where b,c are two others (not a)
    for i, a in enumerate(sel):
        others = [sel[j] for j in range(m) if j != i]
        if len(others) >= 2:
            for p in range(len(others)):
                for q in range(p + 1, len(others)):
                    s = others[p] + others[q] + x
                    if s in DIVS[a]:
                        count += 1
    
    return count

def _can_add(x, selected):
    """Check if x can be added to selected (list) without violating constraints."""
    m = len(selected)
    # 1) x divides sum of any 3 from selected?
    if m >= 3:
        for i in range(m):
            for j in range(i + 1, m):
                for k in range(j + 1, m):
                    s = selected[i] + selected[j] + selected[k]
                    if s in DIVS[x]:
                        return False
    # 2) for each a in selected, a divides sum of 3 from (selected ∪ {x}) \ {a}
    for i, a in enumerate(selected):
        others = [selected[j] for j in range(m) if j != i]
        if len(others) >= 2:
            for p in range(len(others)):
                for q in range(p + 1, len(others)):
                    s = others[p] + others[q] + x
                    if s in DIVS[a]:
                        return False
    return True

def _greedy_build(order, limit=MAX_SIZE):
    """Greedy: add elements in given order if they don't conflict."""
    selected = []
    for x in order:
        if len(selected) >= limit:
            break
        if _can_add(x, selected):
            selected.append(x)
    return selected

def _greedy_degree_based(limit=MAX_SIZE):
    """Greedy that prioritizes elements with fewer potential conflicts.
    At each step, among all remaining candidates, pick the one with minimum conflict count."""
    selected = []
    remaining = set(range(1, N + 1))
    
    while len(selected) < limit and remaining:
        best_x = None
        best_conf = float('inf')
        for x in remaining:
            if _can_add(x, selected):
                # Estimate future conflicts: how many pairs in selected would create issues
                # Simpler: just count conflicts with current set (all 0 if valid)
                # Use a heuristic: prefer larger numbers (fewer divisors)
                conf = 0
                m = len(selected)
                if m >= 3:
                    for i in range(m):
                        for j in range(i + 1, m):
                            for k in range(j + 1, m):
                                s = selected[i] + selected[j] + selected[k]
                                if s in DIVS[x]:
                                    conf += 1
                for i, a in enumerate(selected):
                    others = [selected[j] for j in range(m) if j != i]
                    if len(others) >= 2:
                        for p in range(len(others)):
                            for q in range(p + 1, len(others)):
                                s = others[p] + others[q] + x
                                if s in DIVS[a]:
                                    conf += 1
                if conf < best_conf:
                    best_conf = conf
                    best_x = x
                if conf == 0:
                    break
        if best_x is None:
            break
        selected.append(best_x)
        remaining.discard(best_x)
    
    return selected

def _local_search(selected, time_limit):
    """Local search with 1-add, 1-swap, 2-swap, and remove-one-add-two moves."""
    start = time.time()
    best = list(selected)
    best_size = len(best)
    
    def _try_add_any(cur):
        """Find any element that can be added to cur."""
        cands = [x for x in range(1, N + 1) if x not in cur]
        for x in cands:
            if _can_add(x, cur):
                return x
        return None
    
    def _try_swap(cur):
        """Try removing one element and adding a different one."""
        for i in range(len(cur)):
            temp = cur[:i] + cur[i + 1:]
            cands = [x for x in range(1, N + 1) if x not in temp]
            for x in cands:
                if _can_add(x, temp):
                    return temp + [x]
        return None
    
    def _try_two_swap(cur):
        """Try removing two elements and adding two different ones."""
        n = len(cur)
        cands = [x for x in range(1, N + 1) if x not in cur]
        for i in range(n):
            for j in range(i + 1, n):
                temp = cur[:i] + cur[i + 1:j] + cur[j + 1:]
                for a in cands:
                    if not _can_add(a, temp):
                        continue
                    for b in cands:
                        if b == a:
                            continue
                        if _can_add(b, temp + [a]):
                            return temp + [a, b]
        return None
    
    while time.time() - start < time_limit:
        improved = False
        
        # 1-add
        x = _try_add_any(best)
        if x is not None:
            best.append(x)
            improved = True
        else:
            # 1-swap
            new_best = _try_swap(best)
            if new_best is not None and len(new_best) > len(best):
                best = new_best
                improved = True
            elif new_best is not None:
                # Same size but different composition - try it as starting point
                # Only accept if it leads to improvement later; for now, skip
                pass
            else:
                # 2-swap
                if len(best) > 6:
                    new_best = _try_two_swap(best)
                    if new_best is not None and len(new_best) > len(best):
                        best = new_best
                        improved = True
                # Remove-one-add-two (net +1)
                if len(best) > 4 and not improved:
                    cands = [x for x in range(1, N + 1) if x not in best]
                    found = False
                    for i in range(len(best)):
                        if time.time() - start > time_limit:
                            found = True
                            break
                        temp = best[:i] + best[i + 1:]
                        for a in cands:
                            if time.time() - start > time_limit:
                                found = True
                                break
                            if not _can_add(a, temp):
                                continue
                            for b in cands:
                                if b == a:
                                    continue
                                if _can_add(b, temp + [a]):
                                    best = temp + [a, b]
                                    improved = True
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break
                if not improved:
                    break
        if len(best) > best_size:
            best_size = len(best)
    
    return best

def _is_valid(selected):
    """Verify the entire set is valid."""
    m = len(selected)
    if m < 4:
        return True
    for idx, a in enumerate(selected):
        others = [selected[j] for j in range(m) if j != idx]
        for i in range(len(others)):
            for j in range(i + 1, len(others)):
                for k in range(j + 1, len(others)):
                    s = others[i] + others[j] + others[k]
                    if s % a == 0:
                        return False
    return True

def construct():
    start = time.time()
    deadline = start + 85.0
    best_result = []
    
    # Strategy 1: Greedy large-first
    r = _greedy_build(list(range(N, 0, -1)))
    best_result = r
    
    # Strategy 2: Greedy small-first
    r = _greedy_build(list(range(1, N + 1)))
    if len(r) > len(best_result):
        best_result = r
    
    # Strategy 3: Degree-based greedy
    if time.time() < deadline - 30:
        r = _greedy_degree_based()
        if len(r) > len(best_result):
            best_result = r
    
    # Strategy 4: Interval-based greedy (various ranges)
    for lo, hi in [(60, 192), (40, 192), (80, 192), (50, 150), (70, 170), (30, 180), (100, 192)]:
        if time.time() > deadline - 30:
            break
        order = list(range(hi, lo - 1, -1))
        r = _greedy_build(order)
        if len(r) > len(best_result):
            best_result = r
    
    # Strategy 5: Randomized greedy with multiple seeds
    for seed in range(300):
        if time.time() > deadline - 25:
            break
        random.seed(seed)
        order = list(range(1, N + 1))
        random.shuffle(order)
        r = _greedy_build(order)
        if len(r) > len(best_result):
            best_result = r
    
    # Strategy 6: Weighted random - bias toward mid-range
    for seed in range(100):
        if time.time() > deadline - 20:
            break
        random.seed(10000 + seed)
        # Create weighted order: mid-range numbers more likely
        weights = []
        for x in range(1, N + 1):
            # Peak around 100-150
            w = 1.0 / (1.0 + abs(x - 120) * 0.05)
            weights.append(w)
        order = list(range(1, N + 1))
        # Shuffle with bias
        for i in range(len(order) - 1, 0, -1):
            j = random.choices(range(i + 1), weights=weights[:i + 1])[0]
            order[i], order[j] = order[j], order[i]
        r = _greedy_build(order)
        if len(r) > len(best_result):
            best_result = r
    
    # Local search on best
    if time.time() < deadline - 8:
        remaining = deadline - time.time() - 3
        best_result = _local_search(best_result, remaining)
    
    # Local search on other promising starts
    for seed in range(80):
        if time.time() > deadline - 5:
            break
        random.seed(20000 + seed)
        order = list(range(1, N + 1))
        random.shuffle(order)
        r = _greedy_build(order)
        if len(r) >= len(best_result) - 3:
            remaining = min(deadline - time.time() - 2, 3.0)
            if remaining > 0.5:
                r = _local_search(r, remaining)
                if len(r) > len(best_result):
                    best_result = r
    
    # Final polish
    if time.time() < deadline - 2:
        remaining = deadline - time.time() - 1
        if remaining > 0.3:
            best_result = _local_search(best_result, remaining)
    
    # Verify validity
    if not _is_valid(best_result):
        # Fallback: rebuild
        best_result = _greedy_build(list(range(N, 0, -1)))
    
    return best_result
