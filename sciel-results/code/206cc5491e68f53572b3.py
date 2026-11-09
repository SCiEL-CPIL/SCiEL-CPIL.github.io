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
        # Conflict exists if there is (b, c) in conflicts[a] such that b, c in s
        # OR if there is some e in s such that (a, b) is a conflicting pair for e
        # The second case is equivalent to: for some e in s, (min(a,b), max(a,b)) in conflicts[e] for some b in s.
        # It's safer and fast enough to just check full validity or a smarter incremental check.
        # Let's do a quick check:
        ok = True
        for b, c in conflicts[a]:
            if b in s and c in s:
                ok = False
                break
        
        if ok:
            # Check reverse conflicts: does a conflict with any pair in s for some e in s?
            # This is equivalent to checking if S U {a} is valid.
            # Since N is small, let's just add and check.
            s.add(a)
            if not _is_valid(s, conflicts):
                s.remove(a)
    return s

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    n_max = 180
    conflicts = _precompute_conflicts(n_max)
    
    rng = random.Random(42)
    
    # 1. Generate multiple initial sets using different greedy orders
    candidates = []
    
    # Ascending
    candidates.append(_greedy_build(conflicts, list(range(1, n_max + 1))))
    # Descending
    candidates.append(_greedy_build(conflicts, list(range(n_max, 0, -1))))
    
    # Random orders
    for _ in range(20):
        order = list(range(1, n_max + 1))
        rng.shuffle(order)
        candidates.append(_greedy_build(conflicts, order))
        
    # Degree-based order: elements with fewer conflicts first?
    # Or more conflicts first? Usually, packing problems benefit from placing constrained items first.
    # Let's try sorting by number of conflicts ascending and descending.
    degs = {a: len(conflicts[a]) for a in range(1, n_max + 1)}
    order_asc = sorted(range(1, n_max + 1), key=lambda x: degs[x])
    order_desc = sorted(range(1, n_max + 1), key=lambda x: -degs[x])
    candidates.append(_greedy_build(conflicts, order_asc))
    candidates.append(_greedy_build(conflicts, order_desc))

    best_set = max(candidates, key=len)
    best_size = len(best_set)
    
    # 2. Local Search
    # We try to improve best_set using 1-swaps and 2-swaps.
    # To speed up, we can maintain a list of "blocked" additions or just re-check.
    # Given N=180, re-checking is fine.
    
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
                best_size += 1
                improved = True
                missing.remove(x) # x is now in best_set
                break # Restart loop as set changed
                
        # 2. Try 2-for-1 swaps (Add 2, Remove 1)
        if time.time() > deadline - 5.0:
            break
            
        # To avoid O(N^3) complexity, we sample or limit iterations
        # We try random pairs of missing elements and random elements to remove
        attempts = 0
        max_attempts = 1000
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
                best_size += 1
                improved = True
                break # Restart local search
                
        # 3. Try 1-for-1 swaps to escape plateaus (keep size same, might enable future adds)
        # This is optional and slower. We only do it if we are stuck for a while?
        # For now, let's rely on 2-for-1.
        
        # 4. Try 2-for-2 swaps?
        # Add 2, Remove 2. Size unchanged. Can help rearrange.
        # This is expensive. Skip for now unless stuck.
        
    return sorted(best_set)
