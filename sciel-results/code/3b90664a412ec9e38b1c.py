import time
import random
from fractions import Fraction

_N = 256
_KILL = 90.0
_MARGIN = 5.0

# Precompute reciprocals for fast comparison
_RECIP = [0.0] + [1.0 / i for i in range(1, _N + 1)]


def _get_blocked(S):
    """Compute the blocked set for a given set S.
    blocked contains all x that would form a 3-AP with two elements of S.
    """
    blocked = set()
    S_list = list(S)
    m = len(S_list)
    for i in range(m):
        a = S_list[i]
        for j in range(i + 1, m):
            b = S_list[j]
            # a, b, x => x = 2b - a
            x = 2 * b - a
            if 1 <= x <= _N:
                blocked.add(x)
            # a, x, b => x = (a+b)//2 if even
            s = a + b
            if s % 2 == 0:
                x = s // 2
                if 1 <= x <= _N:
                    blocked.add(x)
            # x, a, b => x = 2a - b
            x = 2 * a - b
            if 1 <= x <= _N:
                blocked.add(x)
    return blocked


def _add_to_blocked(blocked, S, x):
    """Update blocked set when adding x to S. Mutates blocked in place."""
    for a in S:
        if a == x:
            continue
        # x, a, 2a-x
        q = 2 * a - x
        if 1 <= q <= _N:
            blocked.add(q)
        # a, x, 2x-a
        q = 2 * x - a
        if 1 <= q <= _N:
            blocked.add(q)
        # (a+x)/2 if even
        s = a + x
        if s % 2 == 0:
            q = s // 2
            if 1 <= q <= _N:
                blocked.add(q)


def _greedy_ascending():
    """Greedy: add elements in order of smallest x (largest reciprocal) if valid."""
    S = set()
    blocked = set()
    for x in range(1, _N + 1):
        if x not in blocked:
            S.add(x)
            _add_to_blocked(blocked, S, x)
    return S


def _greedy_descending():
    """Greedy: add elements in order of largest x (smallest reciprocal) if valid."""
    S = set()
    blocked = set()
    for x in range(_N, 0, -1):
        if x not in blocked:
            S.add(x)
            _add_to_blocked(blocked, S, x)
    return S


def _random_init(rng, size_range=(30, 80)):
    """Generate a random valid initial set."""
    S = set()
    blocked = set()
    elements = list(range(1, _N + 1))
    rng.shuffle(elements)
    
    target_size = rng.randint(*size_range)
    for x in elements:
        if len(S) >= target_size:
            break
        if x not in blocked:
            S.add(x)
            _add_to_blocked(blocked, S, x)
    return S


def _local_search(S, deadline, rng):
    """Local search: greedy additions + improving swaps using blocked set."""
    S = set(S)
    blocked = _get_blocked(S)
    weight = sum(_RECIP[x] for x in S)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Step 1: Greedy Additions - add smallest valid x not in S and not in blocked
        added_any = False
        for x in range(1, _N + 1):
            if x not in S and x not in blocked:
                S.add(x)
                weight += _RECIP[x]
                _add_to_blocked(blocked, S, x)
                added_any = True
                improved = True
                # Continue scanning; adding x only blocks more elements
        
        if added_any:
            continue  # Restart to check for more additions
        
        # Step 2: Swaps - try to replace a in S with b not in S where 1/b > 1/a
        best_swap_gain = -1.0
        best_a = None
        best_b = None
        
        S_list = sorted(S, reverse=True)  # Try removing large elements first
        
        for a in S_list:
            if time.time() > deadline:
                break
            
            # Only consider b < a for improvement
            # Temporarily remove a to get blocked_without_a
            S_temp = S - {a}
            blocked_temp = _get_blocked(S_temp)
            
            # Find best b: smallest b not in S_temp and not in blocked_temp
            best_b_candidate = None
            for b in range(1, a):
                if b in S_temp:
                    continue
                if b in blocked_temp:
                    continue
                best_b_candidate = b
                break  # Smallest b gives largest 1/b
            
            if best_b_candidate is not None:
                gain = _RECIP[best_b_candidate] - _RECIP[a]
                if gain > best_swap_gain:
                    best_swap_gain = gain
                    best_a = a
                    best_b = best_b_candidate
        
        if best_a is not None and best_swap_gain > 1e-12:
            S.remove(best_a)
            S.add(best_b)
            weight += best_swap_gain
            blocked = _get_blocked(S)
            improved = True
        
        # Step 3: Double swaps - remove two, add one or two (to break local optima)
        if not improved and time.time() < deadline:
            S_list_all = sorted(S)
            found_double = False
            for i in range(len(S_list_all)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(S_list_all)):
                    if time.time() > deadline:
                        break
                    a = S_list_all[i]
                    b = S_list_all[j]
                    S_temp = S - {a, b}
                    blocked_temp = _get_blocked(S_temp)
                    
                    # Try adding the smallest valid element
                    for c in range(1, _N + 1):
                        if c in S_temp or c in blocked_temp:
                            continue
                        gain = _RECIP[c] - _RECIP[a] - _RECIP[b]
                        if gain > 0:
                            S.remove(a)
                            S.remove(b)
                            S.add(c)
                            weight += gain
                            blocked = _get_blocked(S)
                            improved = True
                            found_double = True
                            break
                    if found_double:
                        break
                if found_double:
                    break
            
            # Also try adding two elements after removing one
            if not found_double:
                for a in S_list_all:
                    if time.time() > deadline:
                        break
                    S_temp = S - {a}
                    blocked_temp = _get_blocked(S_temp)
                    
                    # Find two smallest valid elements
                    found = []
                    for c in range(1, _N + 1):
                        if len(found) >= 2:
                            break
                        if c in S_temp or c in blocked_temp:
                            continue
                        found.append(c)
                        # Update blocked_temp after adding c
                        _add_to_blocked(blocked_temp, S_temp, c)
                    
                    if len(found) == 2:
                        gain = _RECIP[found[0]] + _RECIP[found[1]] - _RECIP[a]
                        if gain > 0:
                            S.remove(a)
                            S.add(found[0])
                            S.add(found[1])
                            weight += gain
                            blocked = _get_blocked(S)
                            improved = True
                            found_double = True
                            break
    
    return S


def _reciprocal_sum_exact(S):
    """Compute exact reciprocal sum as Fraction."""
    total = Fraction(0, 1)
    for x in S:
        total += Fraction(1, x)
    return total


def construct():
    start_time = time.time()
    deadline = start_time + _KILL - _MARGIN
    rng = random.Random(20240517)
    
    best_set = None
    best_weight = -1.0
    
    # Generate diverse initial solutions
    inits = []
    
    # 1. Greedy ascending
    S1 = _greedy_ascending()
    inits.append(S1)
    
    # 2. Greedy descending
    S2 = _greedy_descending()
    inits.append(S2)
    
    # 3-10. Random initializations
    for i in range(8):
        S_rand = _random_init(rng, size_range=(20, 100))
        inits.append(S_rand)
    
    # Run local search on each initial solution
    for S_init in inits:
        if time.time() > deadline:
            break
        
        result = _local_search(S_init, deadline, rng)
        w = sum(_RECIP[x] for x in result)
        
        if w > best_weight:
            best_weight = w
            best_set = result.copy()
    
    # Additional local search restarts from best solution with perturbations
    for restart in range(50):
        if time.time() > deadline:
            break
        
        S = set(best_set)
        
        # Randomly remove some elements
        n_remove = rng.randint(1, min(5, len(S)))
        to_remove = rng.sample(list(S), n_remove)
        for x in to_remove:
            S.discard(x)
        
        # Add some random elements
        available = [x for x in range(1, _N + 1) if x not in S]
        if available:
            n_add = rng.randint(0, min(3, len(available)))
            to_add = rng.sample(available, n_add)
            for x in to_add:
                if x not in _get_blocked(S):
                    S.add(x)
        
        result = _local_search(S, deadline, rng)
        w = sum(_RECIP[x] for x in result)
        
        if w > best_weight:
            best_weight = w
            best_set = result.copy()
    
    return list(best_set)
