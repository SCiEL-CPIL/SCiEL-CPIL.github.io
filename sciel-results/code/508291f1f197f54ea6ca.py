import random
from math import gcd
import time

N = 128

def compute_conflict_sets():
    """
    For each x, precompute the set of frozenset pairs {u,v} such that
    {x,u,v} is a bad triple (all pairwise GCDs equal).
    """
    conflicts = {i: set() for i in range(1, N + 1)}
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g_ab = gcd(a, b)
            for c in range(b + 1, N + 1):
                if gcd(a, c) == g_ab and gcd(b, c) == g_ab:
                    conflicts[a].add(frozenset((b, c)))
                    conflicts[b].add(frozenset((a, c)))
                    conflicts[c].add(frozenset((a, b)))
    return conflicts

CONFLICTS = compute_conflict_sets()

# Precompute for each x, the list of (u, v) pairs for fast iteration
CONFLICT_PAIRS = {x: [tuple(p) for p in CONFLICTS[x]] for x in range(1, N + 1)}

# Build a bitmask for each element: bit i is set if element (i+1) is in the set
# For fast pair checking: for each x, precompute a list of (mask_u, mask_v) pairs
# Actually, let's use a different approach: for each x, store the conflict pairs as a set of sorted tuples

# For each x, create a set of (min(u,v), max(u,v)) for O(1) lookup
CONFLICT_TUPLE_SETS = {}
for x in range(1, N + 1):
    ts = set()
    for pair in CONFLICTS[x]:
        u, v = min(pair), max(pair)
        ts.add((u, v))
    CONFLICT_TUPLE_SETS[x] = ts

def can_add_fast(x, current_set):
    """
    Check if x can be added to current_set (a set of ints).
    Uses precomputed conflict tuple sets.
    """
    for u, v in CONFLICT_TUPLE_SETS[x]:
        if u in current_set and v in current_set:
            return False
    return True

def prime_factorization(n):
    """Return dict of prime -> exponent for n."""
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def structural_order():
    """
    Order numbers by a heuristic that tends to produce larger valid sets.
    Numbers with fewer/smaller prime factors tend to have more conflicts.
    Sort by: (number of distinct prime factors, product of primes, value)
    """
    def sort_key(x):
        pf = prime_factorization(x)
        num_distinct = len(pf)
        prod = 1
        for p in pf:
            prod *= p
        return (num_distinct, prod, x)
    return sorted(range(1, N + 1), key=sort_key)

def greedy_build(order):
    s = set()
    for x in order:
        if can_add_fast(x, s):
            s.add(x)
    return s

def local_search(s, deadline, max_iters=200):
    """
    Improve set s with multiple move types.
    """
    improved = True
    iter_count = 0
    while improved and time.time() < deadline and iter_count < max_iters:
        improved = False
        iter_count += 1
        
        # 1. Try to add any single element (most beneficial)
        candidates = [x for x in range(1, N + 1) if x not in s]
        random.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline:
                return s
            if can_add_fast(x, s):
                s.add(x)
                improved = True
                break
        
        if improved or time.time() > deadline:
            continue
            
        # 2. Try 1-for-1 swap: remove one, add one
        list_s = list(s)
        random.shuffle(list_s)
        
        for u in list_s:
            if time.time() > deadline:
                return s
            
            s_without_u = s - {u}
            
            addable = [x for x in range(1, N + 1) if x not in s]
            random.shuffle(addable)
            
            for x in addable:
                if time.time() > deadline:
                    return s
                if can_add_fast(x, s_without_u):
                    s.remove(u)
                    s.add(x)
                    improved = True
                    break
            
            if improved:
                break
        
        if improved or time.time() > deadline:
            continue

        # 3. Try 2-for-1: remove two, add one (net -1, but might enable future adds)
        # Only try this if set size > 20 to avoid degenerate cases
        if len(s) > 20:
            list_s2 = list(s)
            random.shuffle(list_s2)
            for i in range(min(10, len(list_s2) - 1)):
                if time.time() > deadline:
                    return s
                u, v = list_s2[i], list_s2[i + 1]
                if u == v:
                    continue
                s_without_uv = s - {u, v}
                
                addable = [x for x in range(1, N + 1) if x not in s]
                random.shuffle(addable)
                
                for x in addable:
                    if time.time() > deadline:
                        return s
                    if can_add_fast(x, s_without_uv):
                        # Check if this actually helps: can we add more after?
                        s_temp = s_without_uv | {x}
                        # Try to add more
                        more_added = False
                        for y in range(1, N + 1):
                            if y not in s_temp and can_add_fast(y, s_temp):
                                s_temp.add(y)
                                more_added = True
                        if more_added or len(s_temp) >= len(s):
                            s.clear()
                            s.update(s_temp)
                            improved = True
                            break
                if improved:
                    break

    return s

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    best_set = set()
    
    # Initial greedy with structural order
    s = greedy_build(structural_order())
    s = local_search(s, deadline)
    if len(s) > len(best_set):
        best_set = s.copy()
    
    # Try natural order
    if time.time() < deadline:
        s = greedy_build(list(range(1, N + 1)))
        s = local_search(s, deadline)
        if len(s) > len(best_set):
            best_set = s.copy()
    
    # Try reverse natural order
    if time.time() < deadline:
        s = greedy_build(list(range(N, 0, -1)))
        s = local_search(s, deadline)
        if len(s) > len(best_set):
            best_set = s.copy()
    
    # Try order by conflict count (fewest conflicts first)
    if time.time() < deadline:
        conflict_counts = {x: len(CONFLICTS[x]) for x in range(1, N + 1)}
        order_conf = sorted(range(1, N + 1), key=lambda x: conflict_counts[x])
        s = greedy_build(order_conf)
        s = local_search(s, deadline)
        if len(s) > len(best_set):
            best_set = s.copy()
    
    # Try order by conflict count (most conflicts first)
    if time.time() < deadline:
        conflict_counts = {x: len(CONFLICTS[x]) for x in range(1, N + 1)}
        order_conf_desc = sorted(range(1, N + 1), key=lambda x: -conflict_counts[x])
        s = greedy_build(order_conf_desc)
        s = local_search(s, deadline)
        if len(s) > len(best_set):
            best_set = s.copy()
    
    # Random restarts
    while time.time() < deadline:
        random_order = list(range(1, N + 1))
        random.shuffle(random_order)
        
        s = greedy_build(random_order)
        s = local_search(s, deadline)
        
        if len(s) > len(best_set):
            best_set = s.copy()
    
    return list(best_set)

