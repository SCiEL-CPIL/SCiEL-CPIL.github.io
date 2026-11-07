"""Suite task: large integer sets with no two-other-product divisibility."""
import time
import random
from math import isqrt

NMAX = 256
UNIVERSE = list(range(2, NMAX + 1))


def _precompute_conflicts():
    """For each x, compute set of y such that x divides y*z for some z != y.
    Also for each pair (y,z) with y!=z, compute set of x that divide y*z.
    """
    # conflict[x] = set of y such that there exists z != y with x | y*z
    # pair_conflict[(y,z)] = set of x such that x | y*z (for y != z)
    # For efficiency, we'll compute: for each x, which y "block" x
    # x is blocked by y if there exists z != y with x | y*z
    
    # Alternative: for each x, x conflicts with y if x | y*z for some z != y
    # This means: x/gcd(x,y) | z where z != y
    # Let d = x // gcd(x, y). We need d | z for some z in [2,256], z != y.
    # If d == 1: x | y, so x divides y. Then for any z, x | y*z? No, x|y means y = kx, so y*z = kx*z, so x | y*z. Yes!
    # If d > 1: we need some z != y with d | z. If d <= 256, there exists such z (e.g., z=d if d!=y, or z=2d if 2d!=y and 2d<=256, etc.)
    # Actually we need z in [2,256] and z != y.
    
    # Let's think differently. For the greedy approach, when considering adding x to set S:
    # x conflicts with existing {b,c} if x | b*c
    # b conflicts with new {x,c} if b | x*c  
    # c conflicts with new {x,b} if c | x*b
    
    # For speed, precompute for each x: divisors_info
    # x | b*c iff x/gcd(x,b) | c (when considering specific b)
    
    # Let's precompute: for each x, for each b, the "required" divisor d = x//gcd(x,b)
    # Then x | b*c iff d | c
    
    import math
    gcd = math.gcd
    
    # For the conflict-degree approach: 
    # When we have set S, element x in S has "conflict potential" = number of pairs (b,c) in S\{x} 
    # where x|b*c or b|x*c or c|x*b
    # But for greedy removal, we want to remove elements that "block" the most other elements.
    
    # Precompute: for each x, the set of all y such that x and y are in "conflict" 
    # meaning there exists some z where one divides product of other two
    # Actually for the removal heuristic, let's precompute:
    # block[x] = set of y such that y cannot coexist with x in any valid set
    #   i.e., there exists z where x|y*z or y|x*z or (x,y) pair blocks z
    # This is too broad. Let's use a different approach.
    
    # Precompute divisibility: divs[x] = set of multiples of x in [2,256]
    # And for the key check: x | b*c can be checked as (b*c) % x == 0
    
    # For speed in _valid_addition, precompute for each x and each b: 
    # req[x][b] = x // gcd(x, b), so x | b*c iff req[x][b] | c
    req = {}
    for x in range(2, NMAX + 1):
        req[x] = {}
        for b in range(2, NMAX + 1):
            if x == b:
                continue
            d = x // gcd(x, b)
            req[x][b] = d
    
    return req


def _valid_addition_fast(x, S_list, req):
    """Check if x can be added to set S_list without violating the constraint."""
    n = len(S_list)
    for i in range(n):
        b = S_list[i]
        for j in range(n):
            if i == j:
                continue
            c = S_list[j]
            # x | b*c ?
            d = req[x][b]
            if c % d == 0:
                return False
            # b | x*c ?
            d2 = req[b][x]
            if c % d2 == 0:
                return False
            # c | x*b ?
            d3 = req[c][x]
            if b % d3 == 0:
                return False
    return True


def _greedy_remove(req, deadline, order='largest'):
    """Start with all numbers, remove those with highest conflict degree."""
    S = set(range(2, NMAX + 1))
    
    if order == 'largest':
        # Process removal candidates in some order
        pass
    
    # Compute conflict degree for each element in S
    # Element x has conflict if there exist b,c in S\{x} with x|b*c or b|x*c or c|x*b
    # For greedy removal: remove x if it's involved in any conflict
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Find all elements that are in at least one conflict
        to_remove = set()
        S_list = list(S)
        n = len(S_list)
        
        # For efficiency, only check a subset if too large
        # Check each element x: is there b,c in S\{x} such that x|b*c?
        for x in S_list:
            if x in to_remove:
                continue
            found = False
            for i in range(n):
                b = S_list[i]
                if b == x or b in to_remove:
                    continue
                for j in range(n):
                    if j == i:
                        continue
                    c = S_list[j]
                    if c == x or c in to_remove:
                        continue
                    # Check x | b*c
                    d = req[x][b]
                    if c % d == 0:
                        found = True
                        break
                    # Check b | x*c
                    d2 = req[b][x]
                    if c % d2 == 0:
                        found = True
                        break
                    # Check c | x*b
                    d3 = req[c][x]
                    if b % d3 == 0:
                        found = True
                        break
                if found:
                    break
            if found:
                to_remove.add(x)
        
        if to_remove:
            S -= to_remove
            changed = True
    
    return list(S)


def _greedy_add_largest(req, deadline):
    """Greedy: add numbers from largest to smallest if valid."""
    chosen = []
    chosen_set = set()
    for x in range(NMAX, 1, -1):
        if time.time() > deadline:
            break
        if _valid_addition_fast(x, chosen, req):
            chosen.append(x)
            chosen_set.add(x)
    return chosen


def _greedy_add_smallest(req, deadline):
    """Greedy: add numbers from smallest to largest if valid."""
    chosen = []
    chosen_set = set()
    for x in range(2, NMAX + 1):
        if time.time() > deadline:
            break
        if _valid_addition_fast(x, chosen, req):
            chosen.append(x)
            chosen_set.add(x)
    return chosen


def _greedy_add_random(req, deadline, seed=42):
    """Greedy: add numbers in random order if valid."""
    rng = random.Random(seed)
    candidates = list(range(2, NMAX + 1))
    rng.shuffle(candidates)
    chosen = []
    chosen_set = set()
    for x in candidates:
        if time.time() > deadline:
            break
        if _valid_addition_fast(x, chosen, req):
            chosen.append(x)
            chosen_set.add(x)
    return chosen


def _local_search(S, req, deadline, max_iters=500, seed=99999):
    """Local search: try adding, removing, swapping elements."""
    S = list(S)
    S_set = set(S)
    best = S[:]
    best_size = len(S)
    rng = random.Random(seed)
    
    for _it in range(max_iters):
        if time.time() > deadline:
            break
        
        op = rng.random()
        
        if op < 0.4 and len(S) < NMAX:
            # Try adding
            candidates = [x for x in range(2, NMAX + 1) if x not in S_set]
            if candidates:
                rng.shuffle(candidates)
                added = False
                for x in candidates[:30]:
                    if _valid_addition_fast(x, S, req):
                        S.append(x)
                        S_set.add(x)
                        if len(S) > best_size:
                            best = S[:]
                            best_size = len(S)
                        added = True
                        break
                if added:
                    continue
        
        elif op < 0.7 and len(S) > 2:
            # Try removing one and adding one (swap)
            i = rng.randrange(len(S))
            removed = S[i]
            S_temp = S[:i] + S[i+1:]
            S_temp_set = set(S_temp)
            candidates = [x for x in range(2, NMAX + 1) if x not in S_temp_set]
            if candidates:
                rng.shuffle(candidates)
                improved = False
                for x in candidates[:40]:
                    if _valid_addition_fast(x, S_temp, req):
                        S = S_temp + [x]
                        S_set = set(S)
                        if len(S) > best_size:
                            best = S[:]
                            best_size = len(S)
                        improved = True
                        break
                if not improved:
                    S = S_temp + [removed]
                    S_set = set(S)
        
        else:
            # Try removing a random element to allow re-growth
            if S:
                i = rng.randrange(len(S))
                removed = S[i]
                S.pop(i)
                S_set.discard(removed)
                # Try to add back something better
                candidates = [x for x in range(2, NMAX + 1) if x not in S_set]
                if candidates:
                    rng.shuffle(candidates)
                    for x in candidates[:30]:
                        if _valid_addition_fast(x, S, req):
                            S.append(x)
                            S_set.add(x)
                            if len(S) > best_size:
                                best = S[:]
                                best_size = len(S)
                            break
    
    return best


def _simulated_annealing(S, req, deadline, max_iters=4000, T_start=100.0, T_end=0.5, seed=55555):
    """Simulated annealing for the set optimization."""
    S = list(S)
    S_set = set(S)
    best = S[:]
    best_size = len(S)
    rng = random.Random(seed)
    
    current_temp = T_start
    cooling = (T_end / T_start) ** (1.0 / max(1, max_iters))
    
    for _it in range(max_iters):
        if time.time() > deadline:
            break
        current_temp *= cooling
        
        op = rng.random()
        
        if op < 0.35 and len(S) < NMAX:
            # Add
            candidates = [x for x in range(2, NMAX + 1) if x not in S_set]
            if candidates:
                x = rng.choice(candidates)
                if _valid_addition_fast(x, S, req):
                    S.append(x)
                    S_set.add(x)
                    if len(S) > best_size:
                        best = S[:]
                        best_size = len(S)
                    continue
        
        elif op < 0.65 and len(S) > 2:
            # Swap: remove one, add one
            i = rng.randrange(len(S))
            removed = S[i]
            S_temp = S[:i] + S[i+1:]
            S_temp_set = set(S_temp)
            candidates = [x for x in range(2, NMAX + 1) if x not in S_temp_set]
            if candidates:
                x = rng.choice(candidates)
                if _valid_addition_fast(x, S_temp, req):
                    S = S_temp + [x]
                    S_set = set(S)
                    if len(S) > best_size:
                        best = S[:]
                        best_size = len(S)
                else:
                    # Accept worse with some probability? No, we can't have invalid sets.
                    # Just revert
                    pass
        
        elif op < 0.85 and len(S) > 3:
            # Remove two, try to add two different ones
            i, j = rng.sample(range(len(S)), 2)
            i, j = min(i, j), max(i, j)
            S_temp = S[:i] + S[i+1:j] + S[j+1:]
            S_temp_set = set(S_temp)
            candidates = [x for x in range(2, NMAX + 1) if x not in S_temp_set]
            if len(candidates) >= 2:
                x = rng.choice(candidates)
                if _valid_addition_fast(x, S_temp, req):
                    S_temp2 = S_temp + [x]
                    S_temp2_set = S_temp_set | {x}
                    remaining = [y for y in candidates if y != x]
                    if remaining:
                        y = rng.choice(remaining)
                        if _valid_addition_fast(y, S_temp2, req):
                            S = S_temp2 + [y]
                            S_set = set(S)
                            if len(S) > best_size:
                                best = S[:]
                                best_size = len(S)
        
        else:
            # Remove one (to escape local optima)
            if S:
                i = rng.randrange(len(S))
                removed = S[i]
                S.pop(i)
                S_set.discard(removed)
                # Try to add something
                candidates = [x for x in range(2, NMAX + 1) if x not in S_set]
                if candidates:
                    rng.shuffle(candidates)
                    for x in candidates[:20]:
                        if _valid_addition_fast(x, S, req):
                            S.append(x)
                            S_set.add(x)
                            if len(S) > best_size:
                                best = S[:]
                                best_size = len(S)
                            break
    
    return best


def _multi_start_local_search(req, deadline, seeds=None):
    """Run local search from multiple random starts."""
    if seeds is None:
        seeds = [1, 7, 13, 42, 99, 256, 777, 1234]
    
    best = None
    best_size = 0
    
    for seed in seeds:
        if time.time() > deadline:
            break
        # Random greedy start
        rng = random.Random(seed)
        candidates = list(range(2, NMAX + 1))
        rng.shuffle(candidates)
        S = []
        S_set = set()
        for x in candidates:
            if time.time() > deadline:
                break
            if _valid_addition_fast(x, S, req):
                S.append(x)
                S_set.add(x)
        
        # Local search refinement
        S_refined = _local_search(S, req, deadline, max_iters=200, seed=seed * 1000)
        if S_refined and len(S_refined) > best_size:
            best = S_refined
            best_size = len(S_refined)
    
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 90s kill, 5s safety margin
    
    req = _precompute_conflicts()
    
    best = None
    best_size = 0
    
    def consider(c):
        nonlocal best, best_size
        if c and len(c) > best_size:
            best = c
            best_size = len(c)
    
    # 1. Greedy add (largest first) - tends to keep large numbers
    consider(_greedy_add_largest(req, deadline))
    
    # 2. Greedy add (smallest first)
    if time.time() < deadline - 10:
        consider(_greedy_add_smallest(req, deadline))
    
    # 3. Greedy remove from full set
    if time.time() < deadline - 15:
        consider(_greedy_remove(req, deadline - 15))
    
    # 4. Multiple random greedy starts with local search
    if time.time() < deadline - 20:
        consider(_multi_start_local_search(req, deadline - 20, seeds=[1, 7, 13, 42, 99]))
    
    # 5. Simulated annealing from best so far
    if best and time.time() < deadline - 10:
        for sa_seed in [55555, 12345, 77777]:
            if time.time() > deadline - 5:
                break
            consider(_simulated_annealing(best, req, deadline - 5, max_iters=3000, seed=sa_seed))
    
    # 6. More local search iterations on best
    if best and time.time() < deadline - 5:
        consider(_local_search(best, req, deadline - 5, max_iters=1000, seed=424242))
    
    # 7. Final: try to add any remaining valid elements
    if best and time.time() < deadline - 2:
        S = list(best)
        S_set = set(S)
        for x in range(2, NMAX + 1):
            if x not in S_set:
                if _valid_addition_fast(x, S, req):
                    S.append(x)
                    S_set.add(x)
        consider(S)
    
    if best is None:
        # Fallback: all primes
        def is_prime(n):
            if n < 2:
                return False
            if n < 4:
                return True
            if n % 2 == 0 or n % 3 == 0:
                return False
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i + 2) == 0:
                    return False
                i += 6
            return True
        best = [x for x in range(2, NMAX + 1) if is_prime(x)]
    
    return [int(x) for x in best]
