import time
import random
import math

def _precompute_conflicts(n_max=180):
    """
    Precompute for each a in [1, n_max] the list of pairs (b, c) with b < c
    such that 1/a = 1/b + 1/c.
    Equation: (b-a)(c-a) = a^2.
    Let d be a divisor of a^2. Then b-a = d and c-a = a^2/d.
    So b = a + d, c = a + a^2/d.
    We require b, c in [1, n_max] and b != c (d != a^2/d => d != a).
    """
    conflicts = {a: [] for a in range(1, n_max + 1)}
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
            b = a + d
            c = a + a2 // d
            if 1 <= b <= n_max and 1 <= c <= n_max and b != c:
                pair = (min(b, c), max(b, c))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    conflicts[a].append(pair)
    return conflicts

def _is_valid(s, conflicts):
    """Check if set s is valid."""
    s_set = set(s)
    for a in s_set:
        for b, c in conflicts[a]:
            if b in s_set and c in s_set:
                return False
    return True

def _greedy_build(conflicts, order):
    """Greedy build: add elements in the given order if they don't create conflicts."""
    s = set()
    for a in order:
        # Check if adding a creates a conflict with existing pairs in s
        ok = True
        for b, c in conflicts[a]:
            if b in s and c in s:
                ok = False
                break
        
        if ok:
            # Check reverse conflicts: does a conflict with any pair in s for some e in s?
            # This is equivalent to checking if S U {a} is valid.
            s.add(a)
            if not _is_valid(s, conflicts):
                s.remove(a)
    return s

def _repair_set(s, conflicts):
    """
    Remove elements to make the set valid.
    Returns a valid subset of s.
    """
    s = set(s)
    # We will remove elements that are involved in conflicts.
    # A simple strategy: while there is a conflict, remove one of the elements.
    # To maximize size, we should remove the element that is "least valuable" or involved in most conflicts.
    # Here we just remove one randomly or the one with highest degree in the conflict graph induced by s.
    
    changed = True
    while changed:
        changed = False
        # Find all conflicts in s
        conflicts_found = []
        for a in s:
            for b, c in conflicts[a]:
                if b in s and c in s:
                    conflicts_found.append((a, b, c))
        
        if not conflicts_found:
            break
        
        # Count how many times each element appears in conflicts
        freq = {}
        for a, b, c in conflicts_found:
            for x in (a, b, c):
                freq[x] = freq.get(x, 0) + 1
        
        # Remove the element with the highest frequency
        # Tie-break randomly or by value
        max_freq = max(freq.values())
        candidates = [x for x, f in freq.items() if f == max_freq]
        # Random choice to break symmetry
        to_remove = random.choice(candidates)
        s.remove(to_remove)
        changed = True
        
    return s

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    n_max = 180
    conflicts = _precompute_conflicts(n_max)
    
    rng = random.Random(42)
    
    best_set = set()
    
    # Phase 1: Generate initial candidates
    candidates = []
    
    # Ascending
    candidates.append(_greedy_build(conflicts, list(range(1, n_max + 1))))
    # Descending
    candidates.append(_greedy_build(conflicts, list(range(n_max, 0, -1))))
    
    # Random orders
    for _ in range(50):
        order = list(range(1, n_max + 1))
        rng.shuffle(order)
        candidates.append(_greedy_build(conflicts, order))
        
    # Degree-based order
    degs = {a: len(conflicts[a]) for a in range(1, n_max + 1)}
    order_asc = sorted(range(1, n_max + 1), key=lambda x: degs[x])
    order_desc = sorted(range(1, n_max + 1), key=lambda x: -degs[x])
    candidates.append(_greedy_build(conflicts, order_asc))
    candidates.append(_greedy_build(conflicts, order_desc))
    
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
        candidates.append(_greedy_build(conflicts, order))
        
    best_set = max(candidates, key=len)
    
    # Phase 2: Local Search
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        
        # 1. Try adding single elements
        missing = [x for x in range(1, n_max + 1) if x not in best_set]
        rng.shuffle(missing)
        for x in missing:
            if time.time() > deadline - 5.0:
                break
            candidate = set(best_set)
            candidate.add(x)
            if _is_valid(candidate, conflicts):
                best_set = candidate
                improved = True
                break # Restart loop as set changed
        
        # 2. Try 2-for-1 swaps (Add 2, Remove 1)
        if time.time() > deadline - 5.0:
            break
            
        attempts = 0
        max_attempts = 2000
        while attempts < max_attempts and time.time() < deadline - 5.0:
            attempts += 1
            missing = [x for x in range(1, n_max + 1) if x not in best_set]
            if len(missing) < 2:
                break
            x1, x2 = rng.sample(missing, 2)
            z = rng.choice(list(best_set))
            
            candidate = set(best_set)
            candidate.remove(z)
            candidate.add(x1)
            candidate.add(x2)
            
            if _is_valid(candidate, conflicts):
                best_set = candidate
                improved = True
                break # Restart local search
        
        # 3. Try 2-for-2 swaps (Add 2, Remove 2) to rearrange
        if time.time() > deadline - 5.0:
            break
            
        attempts = 0
        max_attempts = 1000
        while attempts < max_attempts and time.time() < deadline - 5.0:
            attempts += 1
            missing = [x for x in range(1, n_max + 1) if x not in best_set]
            if len(missing) < 2:
                break
            x1, x2 = rng.sample(missing, 2)
            z1, z2 = rng.sample(list(best_set), 2)
            
            candidate = set(best_set)
            candidate.remove(z1)
            candidate.remove(z2)
            candidate.add(x1)
            candidate.add(x2)
            
            if _is_valid(candidate, conflicts):
                # Only accept if it helps? Or just accept to explore.
                # Since size is same, we accept to potentially escape local minima.
                best_set = candidate
                # Don't set improved = True to avoid infinite loop of same-size swaps
                # But we might want to trigger a restart?
                # Let's just break and let the loop continue with adds.
                break

    return sorted(best_set)
