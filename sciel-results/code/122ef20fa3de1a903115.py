import time
import numpy as np
from math import gcd, isqrt
import random

def construct():
    start_time = time.time()
    deadline = start_time + 78  # 12s margin for safety

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

    # Pre-compute divisibility: which primes divide each number
    divisors = [[] for _ in range(MAX_N + 1)]
    for n in range(1, MAX_N + 1):
        for p in primes:
            if n % p == 0:
                divisors[n].append(p)

    # Pre-compute conflict matrix: conflict[i][j] = True if gcd(i+1, j+1) != 1
    # Use bitsets for fast conflict checking
    MAX_IDX = MAX_N - 1  # 0-indexed
    conflict_bits = [0] * (MAX_N + 1)
    for n in range(1, MAX_N + 1):
        bits = 0
        for m in range(1, MAX_N + 1):
            if n != m and gcd(n, m) != 1:
                bits |= (1 << (m - 1))
        conflict_bits[n] = bits

    # 1. Try ILP approach with better parameters
    best_result = None
    best_sum = 0
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds

        # Build constraint matrix: for each prime, sum of selected numbers divisible by it <= 1
        A = np.zeros((num_primes, MAX_N), dtype=float)
        for n in range(1, MAX_N + 1):
            for p in divisors[n]:
                A[prime_to_idx[p], n - 1] = 1.0

        # Objective: Maximize sum(n * x_n) => Minimize -sum(n * x_n)
        c = -np.arange(1, MAX_N + 1, dtype=float)

        # Constraints: A @ x <= 1 for each prime
        constraints = LinearConstraint(A, lb=np.zeros(num_primes), ub=np.ones(num_primes))

        # Integrality: all variables integer
        integrality = np.ones(MAX_N)

        # Bounds: 0 <= x <= 1
        bounds = Bounds(lb=np.zeros(MAX_N), ub=np.ones(MAX_N))

        # Use HiGHS solver with increased time limit and gap tolerance
        res = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds,
            options={
                'time_limit': 40,
                'mip_rel_gap': 0.0,  # Try to find exact optimum
                'mip_abs_gap': 0.0,
                'presolve': 'on',
                'parallel': 'on',
            }
        )

        if res.status == 0:
            x = res.x
            selected = [int(i + 1) for i in range(MAX_N) if x[i] > 0.5]
            
            # Verify
            if len(selected) > 0:
                valid = True
                for i in range(len(selected)):
                    for j in range(i + 1, len(selected)):
                        if gcd(selected[i], selected[j]) != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    best_result = [int(n) for n in selected]
                    best_sum = sum(best_result)
    except Exception:
        pass

    # 2. If ILP didn't give good result, use improved local search
    if best_result is None or time.time() < deadline:
        # Start with greedy: pick largest coprime numbers
        S = []
        for n in range(256, 0, -1):
            if all(gcd(n, x) == 1 for x in S):
                S.append(n)
        
        best_set = set(S)
        cur_sum = sum(S)
        
        # If ILP gave a better result, start from there
        if best_result is not None and best_sum > cur_sum:
            best_set = set(best_result)
            cur_sum = best_sum
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            S_list = sorted(list(best_set), reverse=True)
            
            # Try to add any missing number
            for b in range(256, 1, -1):
                if b in best_set:
                    continue
                if all(gcd(b, x) == 1 for x in best_set):
                    best_set.add(b)
                    cur_sum += b
                    improved = True
            
            # 1-for-1 swaps
            for a in S_list:
                if a == 1:
                    continue
                for b in range(256, a, -1):
                    if b in best_set:
                        continue
                    if all(gcd(b, x) == 1 for x in best_set if x != a):
                        new_sum = cur_sum - a + b
                        if new_sum > cur_sum:
                            best_set.remove(a)
                            best_set.add(b)
                            cur_sum = new_sum
                            improved = True
            
            # 2-for-1 swaps: remove two small, add one big
            if time.time() < deadline:
                S_list = sorted(list(best_set))
                for i in range(len(S_list)):
                    if time.time() >= deadline:
                        break
                    a1 = S_list[i]
                    if a1 == 1:
                        continue
                    for j in range(i + 1, len(S_list)):
                        a2 = S_list[j]
                        if a2 == 1:
                            continue
                        for b in range(256, a1 + a2, -1):
                            if b in best_set or b == a1 or b == a2:
                                continue
                            if all(gcd(b, x) == 1 for x in best_set if x not in (a1, a2)):
                                best_set.remove(a1)
                                best_set.remove(a2)
                                best_set.add(b)
                                cur_sum = cur_sum - a1 - a2 + b
                                improved = True
                                break
            
            # 2-for-2 swaps: remove two, add two different
            if time.time() < deadline:
                S_list = sorted(list(best_set))
                for i in range(len(S_list)):
                    if time.time() >= deadline:
                        break
                    a1 = S_list[i]
                    if a1 == 1:
                        continue
                    for j in range(i + 1, len(S_list)):
                        a2 = S_list[j]
                        if a2 == 1:
                            continue
                        # Try to find two numbers b1, b2 that can replace a1, a2
                        for b1 in range(256, a1, -1):
                            if b1 in best_set or b1 == a1 or b1 == a2:
                                continue
                            if not all(gcd(b1, x) == 1 for x in best_set if x not in (a1, a2)):
                                continue
                            for b2 in range(256, max(a1, a2), -1):
                                if b2 in best_set or b2 == a1 or b2 == a2 or b2 == b1:
                                    continue
                                if not all(gcd(b2, x) == 1 for x in best_set if x not in (a1, a2)):
                                    continue
                                if gcd(b1, b2) != 1:
                                    continue
                                new_sum = cur_sum - a1 - a2 + b1 + b2
                                if new_sum > cur_sum:
                                    best_set.remove(a1)
                                    best_set.remove(a2)
                                    best_set.add(b1)
                                    best_set.add(b2)
                                    cur_sum = new_sum
                                    improved = True
                                    break
                            if improved:
                                break
                    if improved:
                        break

    # Final polish: try to add 1 if it fits
    if 1 not in best_set:
        if all(gcd(1, x) == 1 for x in best_set):
            best_set.add(1)
            cur_sum += 1

    return [int(x) for x in sorted(best_set)]
