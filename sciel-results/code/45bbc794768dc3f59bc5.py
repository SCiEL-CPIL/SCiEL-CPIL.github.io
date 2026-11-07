import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin before 90s kill

    N = 256
    candidates = list(range(2, N + 1))

    # Precompute prime factorizations for fast divisibility checks
    def factorize(n):
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

    factorizations = [None] * (N + 1)
    for i in range(2, N + 1):
        factorizations[i] = factorize(i)

    def divides_product(a, b, c):
        """Check if a divides b*c using prime factorizations."""
        fa = factorizations[a]
        fb = factorizations[b]
        fc = factorizations[c]
        
        # For each prime in a's factorization, check if b*c has enough
        for p, exp_a in fa.items():
            exp_bc = fb.get(p, 0) + fc.get(p, 0)
            if exp_bc < exp_a:
                return False
        return True

    # Build conflict graph: x and y are incompatible if there exists z such that x|yz
    # Actually, the constraint is: for all distinct a,b,c in set, a does NOT divide b*c
    # This is a 3-uniform hypergraph constraint, not a 2-uniform one.
    # But we can still use greedy + local search with fast checking.

    def is_valid_addition(a, current_set):
        """Check if adding 'a' to 'current_set' creates any bad triple."""
        if len(current_set) < 2:
            return True
        
        cs = current_set
        n = len(cs)
        
        for i in range(n):
            b = cs[i]
            for j in range(i + 1, n):
                c = cs[j]
                # Check a | b*c
                if divides_product(a, b, c):
                    return False
                # Check b | a*c
                if divides_product(b, a, c):
                    return False
                # Check c | a*b
                if divides_product(c, a, b):
                    return False
        return True

    def greedy_search(order):
        s = []
        for a in order:
            if is_valid_addition(a, s):
                s.append(a)
        return s

    def local_search_improve(s):
        """Try to improve set s by adding elements or swapping."""
        s = list(s)
        current_size = len(s)
        improved = True
        
        while improved and time.time() < deadline - 5.0:
            improved = False
            
            # Try adding elements that weren't in the set
            for a in candidates:
                if a in s:
                    continue
                if is_valid_addition(a, s):
                    s.append(a)
                    improved = True
                    break
            
            # Try swapping: remove one element, add another
            if not improved:
                for i, b in enumerate(s):
                    s_temp = s[:i] + s[i+1:]
                    for a in candidates:
                        if a in s_temp:
                            continue
                        if is_valid_addition(a, s_temp):
                            s = s_temp + [a]
                            improved = True
                            break
                    if improved:
                        break
        
        return s

    best_set = []

    # Strategy 1: Upper Half
    upper_half = list(range(N // 2 + 1, N + 1))
    valid_upper = True
    uh_set = set(upper_half)
    for i in range(len(upper_half)):
        a = upper_half[i]
        for j in range(i + 1, len(upper_half)):
            b = upper_half[j]
            for k in range(j + 1, len(upper_half)):
                c = upper_half[k]
                if divides_product(a, b, c) or divides_product(b, a, c) or divides_product(c, a, b):
                    valid_upper = False
                    break
            if not valid_upper:
                break
        if not valid_upper:
            break
            
    if valid_upper:
        best_set = upper_half[:]
    else:
        strategies = [
            sorted(candidates, reverse=True),
            sorted(candidates, key=lambda x: len(factorizations[x])),
            sorted(candidates, key=lambda x: (len(factorizations[x]), -x))
        ]
        for order in strategies:
            s = greedy_search(order)
            if len(s) > len(best_set):
                best_set = s

    # Local search on the best set found
    if time.time() < deadline - 10.0:
        improved = local_search_improve(best_set)
        if len(improved) > len(best_set):
            best_set = improved

    # Randomized restarts with local search if time permits
    while time.time() < deadline - 5.0:
        order = candidates[:]
        random.shuffle(order)
        s = greedy_search(order)
        
        if time.time() < deadline - 5.0:
            s = local_search_improve(s)
        
        if len(s) > len(best_set):
            best_set = s

    return [int(x) for x in best_set]
