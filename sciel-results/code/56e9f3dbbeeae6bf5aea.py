import math
import time
import sys

def construct():
    start = time.time()
    deadline = start + 85.0
    N = 256

    # 1. Sieve of Eratosthenes for primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for p in range(2, int(math.isqrt(N)) + 1):
        if is_prime[p]:
            for m in range(p * p, N + 1, p):
                is_prime[m] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    num_primes = len(primes)

    # 2. Precompute prime factors for each number 1..N
    prime_factors = [set() for _ in range(N + 1)]
    for i in range(2, N + 1):
        x = i
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                prime_factors[i].add(p)
                while x % p == 0:
                    x //= p
        if x > 1:
            prime_factors[i].add(x)
    prime_factors[1] = set()

    # 3. MILP formulation: maximize sum(n*x_n) s.t. for each prime p, sum_{n: p|n} x_n <= 1, x_n in {0,1}
    best_set = None

    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np

        A = np.zeros((num_primes, N + 1), dtype=np.int8)
        for idx, p in enumerate(primes):
            for multiple in range(p, N + 1, p):
                A[idx, multiple] = 1

        c = np.zeros(N + 1, dtype=np.float64)
        for n in range(1, N + 1):
            c[n] = -float(n)

        cons = LinearConstraint(A, lb=0, ub=1)
        bounds = Bounds(0, 1)
        integrality = np.ones(N + 1, dtype=np.int8)

        res = milp(c, constraints=cons, integrality=integrality, bounds=bounds, options={"time_limit": 40})

        if res.success:
            selected = []
            for n in range(1, N + 1):
                if res.x[n] > 0.5:
                    selected.append(int(n))
            if selected:
                valid = True
                for i in range(len(selected)):
                    for j in range(i + 1, len(selected)):
                        if math.gcd(selected[i], selected[j]) != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    best_set = set(selected)
    except Exception:
        best_set = None

    # 4. Enhanced local search with more swap moves and restarts
    if best_set is None:
        best_set = set()

    best_sum = sum(best_set)
    used_primes = set()
    for n in best_set:
        if n > 1:
            used_primes |= prime_factors[n]

    # Greedy init if empty
    if not best_set:
        candidates = list(range(N, 0, -1))
        best_set = set()
        used_primes = set()
        for n in candidates:
            if n == 1:
                best_set.add(1)
                continue
            pf = prime_factors[n]
            if len(pf & used_primes) == 0:
                best_set.add(n)
                used_primes |= pf
        best_sum = sum(best_set)

    # Local search: try add, 1-for-1, 1-for-2, 2-for-1, 2-for-2, 3-for-2, 2-for-3
    unchanged = 0
    max_unchanged = 200
    while unchanged < max_unchanged and time.time() < deadline - 3.0:
        improved = False
        current_set = set(best_set)
        current_used = set(used_primes)
        current_sum = best_sum

        # Move 1: Add largest possible
        for n in range(N, 0, -1):
            if n not in current_set and n > 1:
                if len(prime_factors[n] & current_used) == 0:
                    new_sum = current_sum + n
                    if new_sum > best_sum:
                        best_set.add(n)
                        used_primes |= prime_factors[n]
                        best_sum = new_sum
                        improved = True
                        break
            elif n == 1 and 1 not in current_set:
                new_sum = current_sum + 1
                if new_sum > best_sum:
                    best_set.add(1)
                    best_sum = new_sum
                    improved = True
                    break

        if improved:
            unchanged = 0
            continue

        # Move 2: 1-for-1 swap
        for n_out in list(best_set):
            if n_out == 1:
                continue
            freed = prime_factors[n_out]
            remaining = current_used - freed
            for n_in in range(N, 0, -1):
                if n_in not in best_set and n_in != 1:
                    if len(prime_factors[n_in] & remaining) == 0:
                        if n_in > n_out:
                            best_set.discard(n_out)
                            best_set.add(n_in)
                            used_primes = remaining | prime_factors[n_in]
                            best_sum += n_in - n_out
                            improved = True
                            break
                elif n_in == 1 and 1 not in best_set:
                    if len(prime_factors[n_in] & remaining) == 0:
                        if n_in > n_out:
                            best_set.discard(n_out)
                            best_set.add(n_in)
                            used_primes = remaining
                            best_sum += n_in - n_out
                            improved = True
                            break
            if improved:
                break

        if improved:
            unchanged = 0
            continue

        # Move 3: 1-for-2 swap
        for n_out in list(best_set):
            if n_out == 1:
                continue
            freed = prime_factors[n_out]
            remaining = current_used - freed
            for n_in1 in range(N, 0, -1):
                if n_in1 not in best_set and n_in1 != 1:
                    if len(prime_factors[n_in1] & remaining) == 0:
                        rem2 = remaining | prime_factors[n_in1]
                        for n_in2 in range(n_in1 - 1, 0, -1):
                            if n_in2 not in best_set and n_in2 != n_in1:
                                if n_in2 == 1:
                                    if len(prime_factors[n_in2] & rem2) == 0:
                                        if n_in1 + n_in2 > n_out:
                                            best_set.discard(n_out)
                                            best_set.add(n_in1)
                                            best_set.add(n_in2)
                                            used_primes = rem2 | prime_factors[n_in2]
                                            best_sum += n_in1 + n_in2 - n_out
                                            improved = True
                                            break
                                else:
                                    if len(prime_factors[n_in2] & rem2) == 0:
                                        if n_in1 + n_in2 > n_out:
                                            best_set.discard(n_out)
                                            best_set.add(n_in1)
                                            best_set.add(n_in2)
                                            used_primes = rem2 | prime_factors[n_in2]
                                            best_sum += n_in1 + n_in2 - n_out
                                            improved = True
                                            break
                            if improved:
                                break
                if improved:
                    break
            if improved:
                break

        if improved:
            unchanged = 0
            continue

        # Move 4: 2-for-1 swap
        lst = [x for x in best_set if x != 1]
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                a, b = lst[i], lst[j]
                freed = prime_factors[a] | prime_factors[b]
                remaining = current_used - freed
                for n_in in range(N, 0, -1):
                    if n_in not in best_set and n_in != 1:
                        if len(prime_factors[n_in] & remaining) == 0:
                            if n_in > a + b:
                                best_set.discard(a)
                                best_set.discard(b)
                                best_set.add(n_in)
                                used_primes = remaining | prime_factors[n_in]
                                best_sum += n_in - a - b
                                improved = True
                                break
                    elif n_in == 1 and 1 not in best_set:
                        if len(prime_factors[n_in] & remaining) == 0:
                            if n_in > a + b:
                                best_set.discard(a)
                                best_set.discard(b)
                                best_set.add(n_in)
                                used_primes = remaining
                                best_sum += n_in - a - b
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

        if improved:
            unchanged = 0
            continue

        # Move 5: 2-for-2 swap
        lst = [x for x in best_set if x != 1]
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                a, b = lst[i], lst[j]
                freed = prime_factors[a] | prime_factors[b]
                remaining = current_used - freed
                for n_in1 in range(N, 0, -1):
                    if n_in1 not in best_set and n_in1 != 1:
                        if len(prime_factors[n_in1] & remaining) == 0:
                            rem2 = remaining | prime_factors[n_in1]
                            for n_in2 in range(n_in1 - 1, 0, -1):
                                if n_in2 not in best_set and n_in2 != n_in1:
                                    if n_in2 == 1:
                                        if len(prime_factors[n_in2] & rem2) == 0:
                                            if n_in1 + n_in2 > a + b:
                                                best_set.discard(a)
                                                best_set.discard(b)
                                                best_set.add(n_in1)
                                                best_set.add(n_in2)
                                                used_primes = rem2 | prime_factors[n_in2]
                                                best_sum += n_in1 + n_in2 - a - b
                                                improved = True
                                                break
                                    else:
                                        if len(prime_factors[n_in2] & rem2) == 0:
                                            if n_in1 + n_in2 > a + b:
                                                best_set.discard(a)
                                                best_set.discard(b)
                                                best_set.add(n_in1)
                                                best_set.add(n_in2)
                                                used_primes = rem2 | prime_factors[n_in2]
                                                best_sum += n_in1 + n_in2 - a - b
                                                improved = True
                                                break
                                if improved:
                                    break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

        if improved:
            unchanged = 0
            continue

        # Move 6: 3-for-2 swap
        lst = [x for x in best_set if x != 1]
        if len(lst) >= 2:
            for i in range(len(lst)):
                for j in range(i + 1, len(lst)):
                    a, b = lst[i], lst[j]
                    freed = prime_factors[a] | prime_factors[b]
                    remaining = current_used - freed
                    for n_in1 in range(N, 0, -1):
                        if n_in1 not in best_set and n_in1 != 1:
                            if len(prime_factors[n_in1] & remaining) == 0:
                                rem2 = remaining | prime_factors[n_in1]
                                for n_in2 in range(n_in1 - 1, 0, -1):
                                    if n_in2 not in best_set and n_in2 != n_in1:
                                        if n_in2 == 1:
                                            if len(prime_factors[n_in2] & rem2) == 0:
                                                rem3 = rem2 | prime_factors[n_in2]
                                                for n_in3 in range(n_in2 - 1, 0, -1):
                                                    if n_in3 not in best_set and n_in3 != n_in1 and n_in3 != n_in2:
                                                        if n_in3 == 1:
                                                            if len(prime_factors[n_in3] & rem3) == 0:
                                                                if n_in1 + n_in2 + n_in3 > a + b:
                                                                    best_set.discard(a)
                                                                    best_set.discard(b)
                                                                    best_set.add(n_in1)
                                                                    best_set.add(n_in2)
                                                                    best_set.add(n_in3)
                                                                    used_primes = rem3 | prime_factors[n_in3]
                                                                    best_sum += n_in1 + n_in2 + n_in3 - a - b
                                                                    improved = True
                                                                    break
                                                        else:
                                                            if len(prime_factors[n_in3] & rem3) == 0:
                                                                if n_in1 + n_in2 + n_in3 > a + b:
                                                                    best_set.discard(a)
                                                                    best_set.discard(b)
                                                                    best_set.add(n_in1)
                                                                    best_set.add(n_in2)
                                                                    best_set.add(n_in3)
                                                                    used_primes = rem3 | prime_factors[n_in3]
                                                                    best_sum += n_in1 + n_in2 + n_in3 - a - b
                                                                    improved = True
                                                                    break
                                                    if improved:
                                                        break
                                                if improved:
                                                    break
                                        else:
                                            if len(prime_factors[n_in2] & rem2) == 0:
                                                rem3 = rem2 | prime_factors[n_in2]
                                                for n_in3 in range(n_in2 - 1, 0, -1):
                                                    if n_in3 not in best_set and n_in3 != n_in1 and n_in3 != n_in2:
                                                        if n_in3 == 1:
                                                            if len(prime_factors[n_in3] & rem3) == 0:
                                                                if n_in1 + n_in2 + n_in3 > a + b:
                                                                    best_set.discard(a)
                                                                    best_set.discard(b)
                                                                    best_set.add(n_in1)
                                                                    best_set.add(n_in2)
                                                                    best_set.add(n_in3)
                                                                    used_primes = rem3 | prime_factors[n_in3]
                                                                    best_sum += n_in1 + n_in2 + n_in3 - a - b
                                                                    improved = True
                                                                    break
                                                        else:
                                                            if len(prime_factors[n_in3] & rem3) == 0:
                                                                if n_in1 + n_in2 + n_in3 > a + b:
                                                                    best_set.discard(a)
                                                                    best_set.discard(b)
                                                                    best_set.add(n_in1)
                                                                    best_set.add(n_in2)
                                                                    best_set.add(n_in3)
                                                                    used_primes = rem3 | prime_factors[n_in3]
                                                                    best_sum += n_in1 + n_in2 + n_in3 - a - b
                                                                    improved = True
                                                                    break
                                                    if improved:
                                                        break
                                                if improved:
                                                    break
                                        if improved:
                                            break
                            if improved:
                                break
                    if improved:
                        break
                if improved:
                    break

        if improved:
            unchanged = 0
            continue

        # Move 7: 2-for-3 swap
        lst = [x for x in best_set if x != 1]
        if len(lst) >= 3:
            for i in range(len(lst)):
                for j in range(i + 1, len(lst)):
                    for k in range(j + 1, len(lst)):
                        a, b, c = lst[i], lst[j], lst[k]
                        freed = prime_factors[a] | prime_factors[b] | prime_factors[c]
                        remaining = current_used - freed
                        for n_in1 in range(N, 0, -1):
                            if n_in1 not in best_set and n_in1 != 1:
                                if len(prime_factors[n_in1] & remaining) == 0:
                                    rem2 = remaining | prime_factors[n_in1]
                                    for n_in2 in range(n_in1 - 1, 0, -1):
                                        if n_in2 not in best_set and n_in2 != n_in1:
                                            if n_in2 == 1:
                                                if len(prime_factors[n_in2] & rem2) == 0:
                                                    if n_in1 + n_in2 > a + b + c:
                                                        best_set.discard(a)
                                                        best_set.discard(b)
                                                        best_set.discard(c)
                                                        best_set.add(n_in1)
                                                        best_set.add(n_in2)
                                                        used_primes = rem2 | prime_factors[n_in2]
                                                        best_sum += n_in1 + n_in2 - a - b - c
                                                        improved = True
                                                        break
                                            else:
                                                if len(prime_factors[n_in2] & rem2) == 0:
                                                    if n_in1 + n_in2 > a + b + c:
                                                        best_set.discard(a)
                                                        best_set.discard(b)
                                                        best_set.discard(c)
                                                        best_set.add(n_in1)
                                                        best_set.add(n_in2)
                                                        used_primes = rem2 | prime_factors[n_in2]
                                                        best_sum += n_in1 + n_in2 - a - b - c
                                                        improved = True
                                                        break
                                        if improved:
                                            break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

        if not improved:
            unchanged += 1

    # Final validation
    result = [int(x) for x in best_set]
    result = list(set(result))
    result = [x for x in result if 1 <= x <= N]

    # Verify pairwise coprimality
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            assert math.gcd(result[i], result[j]) == 1
            assert isinstance(result[i], int)

    return result
