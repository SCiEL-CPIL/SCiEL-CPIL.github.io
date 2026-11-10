import time
import numpy as np
from math import gcd, isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin before 90s kill

    MAX_N = 256

    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, isqrt(n) + 1):
            if sieve[i]:
                for j in range(i * i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(MAX_N)
    num_primes = len(primes)
    prime_to_idx = {p: i for i, p in enumerate(primes)}

    # Precompute prime factors for each number
    prime_factors = {}
    for n in range(1, MAX_N + 1):
        factors = set()
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                factors.add(p)
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            factors.add(temp)
        prime_factors[n] = factors

    # Precompute compatibility: compat[a][b] = gcd(a,b)==1
    # Use a 2D boolean array for fast lookup
    compat = [[True] * (MAX_N + 1) for _ in range(MAX_N + 1)]
    for a in range(1, MAX_N + 1):
        for b in range(a + 1, MAX_N + 1):
            if gcd(a, b) == 1:
                compat[a][b] = True
                compat[b][a] = True
            else:
                compat[a][b] = False
                compat[b][a] = False

    def is_compatible_with_all(n, S):
        """Check if n is coprime to all elements in S."""
        for x in S:
            if x != n and not compat[n][x]:
                return False
        return True

    def is_valid_set(S):
        """Verify all pairs in S are coprime."""
        lst = list(S)
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if not compat[lst[i]][lst[j]]:
                    return False
        return True

    # 1. Try ILP approach
    best_set = None
    best_sum = 0

    try:
        from scipy.optimize import milp, LinearConstraint, Bounds

        # Build constraint matrix A (num_primes x MAX_N)
        A = np.zeros((num_primes, MAX_N), dtype=np.int8)
        for n in range(1, MAX_N + 1):
            for p in prime_factors[n]:
                A[prime_to_idx[p], n - 1] = 1

        # Objective: maximize sum(n * x_n) => minimize -sum(n * x_n)
        c = -np.arange(1, MAX_N + 1, dtype=float)

        constraints = LinearConstraint(A, lb=np.zeros(num_primes), ub=np.ones(num_primes))
        integrality = np.ones(MAX_N)
        bounds = Bounds(lb=np.zeros(MAX_N), ub=np.ones(MAX_N))

        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)

        if res.status == 0:
            x = res.x
            selected = [int(i + 1) for i in range(MAX_N) if x[i] > 0.5]
            if len(selected) > 0:
                if is_valid_set(set(selected)):
                    best_set = set(selected)
                    best_sum = sum(selected)
    except Exception:
        pass

    # 2. Greedy initialization with multiple strategies
    def greedy_init(order):
        """Greedy: start with 1, then add numbers in given order if compatible."""
        S = {1}
        for n in order:
            if n not in S and is_compatible_with_all(n, S):
                S.add(n)
        return S

    # Try different greedy orderings
    greedy_candidates = []
    
    # Order 1: descending
    greedy_candidates.append(greedy_init(range(256, 1, -1)))
    
    # Order 2: descending, but skip 1 initially (we add 1 at start)
    greedy_candidates.append(greedy_init(range(256, 1, -1)))
    
    # Order 3: Try to prioritize numbers with fewer prime factors
    def num_prime_factors(n):
        return len(prime_factors[n])
    
    # Sort by (num_prime_factors, -value) - prefer numbers with fewer factors and higher value
    order3 = sorted(range(256, 1, -1), key=lambda n: (num_prime_factors(n), -n))
    greedy_candidates.append(greedy_init(order3))
    
    # Order 4: Descending by value, but only add if it doesn't block too many future options
    # For now, just try a few more orderings
    
    # Pick the best greedy candidate
    for cand in greedy_candidates:
        cand_sum = sum(cand)
        if best_set is None or cand_sum > best_sum:
            best_set = cand
            best_sum = cand_sum

    # 3. Local search with multiple move types
    def local_search(S, S_sum, time_limit):
        """Perform local search on set S, return improved set and sum."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            S_list = sorted(S, reverse=True)
            
            # Move type 1: Try adding a number if it fits
            for b in range(256, 1, -1):
                if b in S:
                    continue
                if is_compatible_with_all(b, S):
                    S.add(b)
                    S_sum += b
                    improved = True
                    break
            
            # Move type 2: Try 1-for-1 swaps: remove a, add b if b > a and compatible
            for a in list(S_list):
                if a == 1:
                    continue
                for b in range(256, a, -1):
                    if b in S:
                        continue
                    # Check if b is compatible with all elements except a
                    ok = True
                    for x in S:
                        if x != a and not compat[b][x]:
                            ok = False
                            break
                    if ok:
                        new_sum = S_sum - a + b
                        if new_sum > S_sum:
                            S.discard(a)
                            S.add(b)
                            S_sum = new_sum
                            improved = True
                            break
                if improved:
                    break
            
            # Move type 3: Try 2-for-1 swaps: remove two small numbers, add one large
            if len(S) >= 2:
                S_sorted = sorted(S)
                for i in range(len(S_sorted)):
                    a = S_sorted[i]
                    if a == 1:
                        continue
                    for j in range(i + 1, len(S_sorted)):
                        b = S_sorted[j]
                        if b == 1:
                            continue
                        if a + b < 256:
                            continue  # need to add something larger
                        for c in range(256, max(a, b), -1):
                            if c in S:
                                continue
                            # Check c compatible with all except a and b
                            ok = True
                            for x in S:
                                if x != a and x != b and not compat[c][x]:
                                    ok = False
                                    break
                            if ok:
                                new_sum = S_sum - a - b + c
                                if new_sum > S_sum:
                                    S.discard(a)
                                    S.discard(b)
                                    S.add(c)
                                    S_sum = new_sum
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
            
            # Move type 4: Try 1-for-2 swaps: remove one number, add two smaller ones
            for a in list(S):
                if a == 1:
                    continue
                # Try to find two numbers b, c such that b+c > a and both compatible with S\{a}
                found = False
                for b in range(256, 1, -1):
                    if b in S or b == a:
                        continue
                    # Check b compatible with S\{a}
                    ok_b = True
                    for x in S:
                        if x != a and not compat[b][x]:
                            ok_b = False
                            break
                    if not ok_b:
                        continue
                    for c in range(256, b, -1):
                        if c in S or c == a or c == b:
                            continue
                        if b + c <= a:
                            continue  # need b+c > a
                        # Check c compatible with S\{a} and with b
                        ok_c = True
                        for x in S:
                            if x != a and not compat[c][x]:
                                ok_c = False
                                break
                        if not ok_c or not compat[b][c]:
                            continue
                        new_sum = S_sum - a + b + c
                        if new_sum > S_sum:
                            S.discard(a)
                            S.add(b)
                            S.add(c)
                            S_sum = new_sum
                            improved = True
                            found = True
                            break
                    if found:
                        break
                if improved:
                    break

        return S, S_sum

    # Run local search
    if best_set is not None:
        best_set, best_sum = local_search(best_set, best_sum, deadline)

    # 4. Final polish: exhaustive check of all single additions and swaps
    if time.time() < deadline - 5:
        # Check all possible single additions
        for b in range(256, 1, -1):
            if b in best_set:
                continue
            if is_compatible_with_all(b, best_set):
                best_set.add(b)
                best_sum += b
        
        # Check all possible 1-for-1 swaps more carefully
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for a in list(best_set):
                if a == 1:
                    continue
                for b in range(256, a, -1):
                    if b in best_set:
                        continue
                    ok = all(compat[b][x] for x in best_set if x != a)
                    if ok:
                        new_sum = best_sum - a + b
                        if new_sum > best_sum:
                            best_set.discard(a)
                            best_set.add(b)
                            best_sum = new_sum
                            improved = True
                            break
                if improved:
                    break

    # Final verification
    result = sorted(best_set)
    if not result:
        result = [1]

    # Ensure all are Python ints
    return [int(x) for x in result]
