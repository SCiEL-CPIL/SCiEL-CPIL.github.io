"""Improved constructive search for the 'no element divides sum of 3 distinct others' problem.

Strategy:
1. Try to build a large set of high values (97..192) using a smart greedy that prioritizes
   elements that are hard to be divisors (large primes, high values).
2. Apply local search (add/swap/remove) to repair violations and maximize cardinality.
3. Use multiple random restarts and ordering heuristics.
4. Time guard: keep a strict deadline and return best found.
"""
import time
import random
from itertools import combinations

def _violations_count(x, selected):
    """Return number of violations if x were added to selected.
    
    A violation is a pair (a, {b,c,d}) where a in S ∪ {x}, and b,c,d are 3 distinct
    elements from S ∪ {x} \\ {a}, and a | (b+c+d).
    """
    if not selected:
        return 0
    
    # Case 1: x is the divisor
    cnt = 0
    n = len(selected)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if (selected[i] + selected[j] + selected[k]) % x == 0:
                    cnt += 1
                    if cnt > 0:
                        break
                # early exit if we just need >0
            if cnt > 0:
                break
        if cnt > 0:
            break
    if cnt > 0:
        return cnt  # at least 1
    
    # Case 2: x is a summand, a in selected is divisor
    for idx_a, a in enumerate(selected):
        # Need b, c from selected\{a} such that b+c+x ≡ 0 mod a
        for i in range(n):
            if i == idx_a:
                continue
            for j in range(i+1, n):
                if j == idx_a:
                    continue
                if (selected[i] + selected[j] + x) % a == 0:
                    return 1
    
    return 0

def _can_add(x, selected):
    """Check if x can be added to selected without violating constraints."""
    return _violations_count(x, selected) == 0

def _greedy(order, deadline):
    """Greedy construction following given order, respecting deadline."""
    selected = []
    for x in order:
        if time.time() > deadline:
            break
        if _can_add(x, selected):
            selected.append(x)
    return selected

def _local_search(selected, pool, deadline, max_iters=200):
    """Try to improve selected by adding elements from pool or swapping.
    
    Returns the best set found.
    """
    selected = list(selected)
    selected_set = set(selected)
    best = selected
    best_len = len(selected)
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        improved = False
        
        # Try to add an element
        for x in pool:
            if x in selected_set:
                continue
            if _can_add(x, selected):
                selected.append(x)
                selected_set.add(x)
                if len(selected) > best_len:
                    best = list(selected)
                    best_len = len(selected)
                improved = True
                break
        
        if improved:
            continue
        
        # Try swaps: remove one from selected, add one from pool
        for idx, y in enumerate(selected):
            if time.time() > deadline:
                break
            for x in pool:
                if x in selected_set:
                    continue
                # Try replacing y with x
                candidate = selected[:idx] + selected[idx+1:]
                if _can_add(x, candidate):
                    candidate.append(x)
                    if len(candidate) >= best_len:
                        selected = candidate
                        selected_set = set(selected)
                        if len(selected) > best_len:
                            best = list(selected)
                            best_len = len(selected)
                        improved = True
                        break
            if improved:
                break
        
        if not improved:
            break
    
    return best

def _is_prime(x):
    if x < 2:
        return False
    if x < 4:
        return True
    if x % 2 == 0 or x % 3 == 0:
        return False
    i = 5
    while i * i <= x:
        if x % i == 0 or x % (i + 2) == 0:
            return False
        i += 6
    return True

def _num_divisors(x):
    cnt = 0
    for d in range(1, int(x**0.5) + 1):
        if x % d == 0:
            cnt += 1
            if d != x // d:
                cnt += 1
    return cnt

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    best = []
    best_len = 0
    
    def _update_best(s):
        nonlocal best, best_len
        if len(s) > best_len:
            best = list(s)
            best_len = len(s)
            return True
        return False
    
    # --- Phase 1: Multiple greedy orderings ---
    
    # Ordering 1: descending
    if time.time() < deadline:
        order = list(range(192, 0, -1))
        result = _greedy(order, deadline)
        _update_best(result)
    
    # Ordering 2: ascending
    if time.time() < deadline:
        order = list(range(1, 193))
        result = _greedy(order, deadline)
        _update_best(result)
    
    # Ordering 3: primes first (descending), then composites
    if time.time() < deadline:
        primes = [x for x in range(192, 0, -1) if _is_prime(x)]
        composites = [x for x in range(192, 0, -1) if not _is_prime(x)]
        order = primes + composites
        result = _greedy(order, deadline)
        _update_best(result)
    
    # Ordering 4: high numbers with few divisors first
    if time.time() < deadline:
        nums = list(range(192, 0, -1))
        order = sorted(nums, key=lambda x: (_num_divisors(x), -x))
        result = _greedy(order, deadline)
        _update_best(result)
    
    # Ordering 5: odd first, then even (descending)
    if time.time() < deadline:
        odds = [x for x in range(192, 0, -1) if x % 2 == 1]
        evens = [x for x in range(192, 0, -1) if x % 2 == 0]
        order = odds + evens
        result = _greedy(order, deadline)
        _update_best(result)
    
    # Ordering 6: numbers > 96 first (descending), then rest
    if time.time() < deadline:
        high = list(range(192, 96, -1))
        low = list(range(96, 0, -1))
        order = high + low
        result = _greedy(order, deadline)
        _update_best(result)
    
    # Ordering 7: coprime to 6 first
    if time.time() < deadline:
        c6 = [x for x in range(192, 0, -1) if x % 2 != 0 and x % 3 != 0]
        rest = [x for x in range(192, 0, -1) if x % 2 == 0 or x % 3 == 0]
        order = c6 + rest
        result = _greedy(order, deadline)
        _update_best(result)
    
    # --- Phase 2: Local search from best greedy results ---
    if time.time() < deadline:
        # Use the best greedy result as starting point
        starting = list(best)
        pool = [x for x in range(1, 193) if x not in set(starting)]
        result = _local_search(starting, pool, deadline, max_iters=500)
        _update_best(result)
    
    # --- Phase 3: Random restarts with local search ---
    if time.time() < deadline:
        rng = random.Random(12345)
        for restart in range(20):
            if time.time() > deadline:
                break
            # Random ordering with bias toward high numbers
            order = list(range(1, 193))
            # Bias: sort by random weight + value/192
            weights = [rng.random() + x / 192.0 for x in order]
            order.sort(key=lambda x: weights[order.index(x)] if False else 0)
            # Simpler: just shuffle but weight high numbers
            rng.shuffle(order)
            result = _greedy(order, deadline)
            
            if len(result) > best_len:
                _update_best(result)
            
            # Local search
            if time.time() < deadline and len(result) >= best_len - 2:
                pool = [x for x in range(1, 193) if x not in set(result)]
                result2 = _local_search(result, pool, deadline, max_iters=200)
                _update_best(result2)
    
    # --- Phase 4: Targeted high-interval construction with repair ---
    if time.time() < deadline:
        # Start with {97, 98, ..., 192} and try to repair
        candidate = list(range(97, 193))  # 96 elements
        # Find violations and remove elements
        removed = set()
        max_removals = 50
        for _ in range(max_removals):
            if time.time() > deadline:
                break
            # Find first violation
            viol = None
            n = len(candidate)
            found = False
            for a_idx, a in enumerate(candidate):
                if found:
                    break
                for i in range(n):
                    if i == a_idx or found:
                        continue
                    for j in range(i+1, n):
                        if j == a_idx or found:
                            continue
                        for k in range(j+1, n):
                            if k == a_idx or found:
                                continue
                            if (candidate[i] + candidate[j] + candidate[k]) % a == 0:
                                viol = (a, candidate[i], candidate[j], candidate[k])
                                found = True
                                break
            if viol is None:
                break
            # Remove the element that appears in the most violations (approx: remove divisor)
            a, b, c, d = viol
            # Remove a (the divisor) - it's constraining many triples
            if a in candidate:
                candidate.remove(a)
                removed.add(a)
        
        if len(candidate) > best_len:
            _update_best(candidate)
        
        # Try to add back removed elements or others
        if time.time() < deadline:
            pool = [x for x in range(1, 193) if x not in set(candidate)]
            result = _local_search(candidate, pool, deadline, max_iters=300)
            _update_best(result)
    
    # --- Phase 5: Final polish - try to add any remaining elements ---
    if time.time() < deadline:
        candidate = list(best)
        pool = [x for x in range(1, 193) if x not in set(candidate)]
        result = _local_search(candidate, pool, deadline, max_iters=500)
        _update_best(result)
    
    return best
