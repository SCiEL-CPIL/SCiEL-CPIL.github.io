import time
import random
import math

def _precompute(n_max=180):
    """
    Precompute forbidden triples and conflict pairs.
    For each a, find (b, c) with b < c such that 1/a = 1/b + 1/c.
    Equation: (b-a)(c-a) = a^2.
    Let d be a divisor of a^2, d < a^2/d. Then b = a + d, c = a + a^2/d.
    We require b, c in [1, n_max] and b != c (d != a^2/d => d != a).
    """
    conflicts = [[] for _ in range(n_max + 1)]
    triples = []
    
    for a in range(1, n_max + 1):
        a2 = a * a
        # Find divisors of a2
        divisors = []
        for d in range(1, int(math.isqrt(a2)) + 1):
            if a2 % d == 0:
                divisors.append(d)
                if d != a2 // d:
                    divisors.append(a2 // d)
        
        seen_pairs = set()
        for d in divisors:
            if d == a:
                continue
            e = a2 // d
            if d >= e:
                continue
            b = a + d
            c = a + e
            if 1 <= b <= n_max and 1 <= c <= n_max and b != c:
                pair = (min(b, c), max(b, c))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    conflicts[a].append(pair)
                    triples.append((a, b, c))
    
    # Also build reverse conflicts: for each element x, list of pairs (y, z) 
    # such that {x, y, z} is a forbidden triple.
    # This is more efficient for checking if adding x creates a conflict.
    conflict_pairs = [[] for _ in range(n_max + 1)]
    for a, b, c in triples:
        conflict_pairs[a].append((b, c))
        conflict_pairs[b].append((a, c))
        conflict_pairs[c].append((a, b))
    
    return conflict_pairs, triples

def _is_valid(s, conflict_pairs):
    """Check if set s is valid."""
    s_set = set(s)
    for a in s_set:
        for b, c in conflict_pairs[a]:
            if b in s_set and c in s_set:
                return False
    return True

def _greedy_build(conflict_pairs, order):
    """Greedy build: add elements in the given order if they don't create conflicts."""
    s = set()
    for a in order:
        # Check if adding a creates a conflict with existing pairs in s
        ok = True
        for b, c in conflict_pairs[a]:
            if b in s and c in s:
                ok = False
                break
        
        if ok:
            s.add(a)
    return s

def _local_search(s, conflict_pairs, n_max, deadline, rng):
    """
    Local search to improve the set.
    Tries: 1-add, 1-swap (remove 1, add 1), 2-swap (remove 2, add 2).
    """
    s = set(s)
    improved = True
    iter_count = 0
    max_iter = 1000
    
    while improved and iter_count < max_iter and time.time() < deadline - 2.0:
        improved = False
        iter_count += 1
        
        # 1. Try adding single elements
        missing = [x for x in range(1, n_max + 1) if x not in s]
        rng.shuffle(missing)
        for x in missing:
            if time.time() > deadline - 2.0:
                break
            candidate = set(s)
            candidate.add(x)
            if _is_valid(candidate, conflict_pairs):
                s = candidate
                improved = True
                break
        
        if improved:
            continue
        
        # 2. Try 1-swap: remove one, add one
        s_list = list(s)
        rng.shuffle(s_list)
        for z in s_list:
            if time.time() > deadline - 2.0:
                break
            candidate = set(s)
            candidate.remove(z)
            for x in missing:
                if time.time() > deadline - 2.0:
                    break
                if x in candidate:
                    continue
                candidate.add(x)
                if _is_valid(candidate, conflict_pairs):
                    s = candidate
                    improved = True
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # 3. Try 2-swap: remove two, add two
        if len(s) >= 2 and len(missing) >= 2:
            s_list = list(s)
            rng.shuffle(s_list)
            for i in range(min(50, len(s_list))):
                if time.time() > deadline - 2.0:
                    break
                z1 = s_list[i]
                for j in range(i + 1, min(i + 5, len(s_list))):
                    if time.time() > deadline - 2.0:
                        break
                    z2 = s_list[j]
                    candidate = set(s)
                    candidate.remove(z1)
                    candidate.remove(z2)
                    missing_list = list(missing)
                    rng.shuffle(missing_list)
                    for x1 in missing_list[:20]:
                        if time.time() > deadline - 2.0:
                            break
                        if x1 in candidate:
                            continue
                        candidate.add(x1)
                        for x2 in missing_list[:20]:
                            if time.time() > deadline - 2.0:
                                break
                            if x2 == x1 or x2 in candidate:
                                continue
                            candidate.add(x2)
                            if _is_valid(candidate, conflict_pairs):
                                s = candidate
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
    
    return s

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    n_max = 180
    conflict_pairs, triples = _precompute(n_max)
    
    rng = random.Random(42)
    
    best_set = set()
    best_size = 0
    
    # Phase 1: Generate initial candidates using greedy with different orders
    candidates = []
    
    # Ascending
    candidates.append(_greedy_build(conflict_pairs, list(range(1, n_max + 1))))
    # Descending
    candidates.append(_greedy_build(conflict_pairs, list(range(n_max, 0, -1))))
    
    # Random orders
    for _ in range(50):
        order = list(range(1, n_max + 1))
        rng.shuffle(order)
        candidates.append(_greedy_build(conflict_pairs, order))
        
    # Degree-based order
    degs = {a: len(conflict_pairs[a]) for a in range(1, n_max + 1)}
    order_asc = sorted(range(1, n_max + 1), key=lambda x: degs[x])
    order_desc = sorted(range(1, n_max + 1), key=lambda x: -degs[x])
    candidates.append(_greedy_build(conflict_pairs, order_asc))
    candidates.append(_greedy_build(conflict_pairs, order_desc))
    
    # Weighted random orders (prefer low degree?)
    for _ in range(20):
        weights = [1.0 / (degs[x] + 1) for x in range(1, n_max + 1)]
        # Normalize weights
        total_w = sum(weights)
        weights = [w / total_w for w in weights]
        
        # Sample without replacement using weights
        order = []
        remaining = list(range(1, n_max + 1))
        w_rem = weights[:]
        for _ in range(n_max):
            # Pick one
            r = rng.random()
            cum = 0
            idx = 0
            for i, w in enumerate(w_rem):
                cum += w
                if r <= cum:
                    idx = i
                    break
            order.append(remaining[idx])
            remaining.pop(idx)
            w_rem.pop(idx)
        candidates.append(_greedy_build(conflict_pairs, order))
        
    # Phase 2: Local search on each candidate
    for cand in candidates:
        if time.time() > deadline - 5.0:
            break
        improved_set = _local_search(cand, conflict_pairs, n_max, deadline, rng)
        if len(improved_set) > best_size:
            best_size = len(improved_set)
            best_set = improved_set.copy()
    
    # Phase 3: Final local search on best set with more iterations
    if time.time() < deadline - 3.0:
        best_set = _local_search(best_set, conflict_pairs, n_max, deadline, rng)
    
    return sorted(best_set)
