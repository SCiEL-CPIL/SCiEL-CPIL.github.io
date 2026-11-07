import math
import time
from itertools import combinations
from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np

def construct():
    start = time.time()
    deadline = start + 80  # safety margin under 90s

    N = 256

    # Sieve for primes
    def sieve(limit):
        is_p = [True] * (limit + 1)
        is_p[0] = is_p[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_p[i]:
                for j in range(i * i, limit + 1, i):
                    is_p[j] = False
        return [i for i in range(2, limit + 1) if is_p[i]]

    primes = sieve(N)
    prime_set = set(primes)
    prime_index = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)

    # Prime powers: largest p^k <= N for each prime p
    prime_powers = {}
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        prime_powers[p] = power

    # Precompute prime factorization (as frozenset of distinct primes) for all numbers 1..N
    F = [None] * (N + 1)
    F[1] = frozenset()
    for i in range(2, N + 1):
        x = i
        factors = set()
        d = 2
        while d * d <= x:
            if x % d == 0:
                factors.add(d)
                while x % d == 0:
                    x //= d
            else:
                d += 1
        if x > 1:
            factors.add(x)
        F[i] = frozenset(factors)

    # Base solution: 1 + all prime powers
    base_numbers = [1] + [prime_powers[p] for p in primes]
    base_sum = sum(base_numbers)

    # Generate candidate moves: composites with positive gain
    # For each distinct prime set, keep only the largest composite
    best_c_for_ps = {}
    for c in range(2, N + 1):
        if c in prime_set:
            continue
        ps = F[c]
        if len(ps) < 2:
            continue
        if ps not in best_c_for_ps or c > best_c_for_ps[ps]:
            best_c_for_ps[ps] = c

    moves = []
    for ps, c in best_c_for_ps.items():
        base_cost = sum(prime_powers[p] for p in ps)
        gain = c - base_cost
        if gain > 0:
            moves.append((ps, c, gain))

    n_moves = len(moves)

    # Build ILP: maximize sum(gain_i * x_i) subject to each prime used at most once
    # x_i in {0, 1} for each move i
    # Constraint: for each prime p, sum(x_i for moves i containing p) <= 1

    # Objective: maximize sum of gains
    c_obj = np.array([-m[2] for m in moves], dtype=float)  # minimize negative gain

    # Constraints: for each prime, at most one move can use it
    A_rows = []
    for p in primes:
        row = np.zeros(n_moves)
        for i, (ps, c, gain) in enumerate(moves):
            if p in ps:
                row[i] = 1.0
        A_rows.append(row)

    A = np.array(A_rows) if A_rows else np.zeros((1, n_moves))
    b_ub = np.ones(len(primes)) if len(primes) > 0 else np.ones(1)

    constraints = LinearConstraint(A, 0, b_ub)
    integrality = np.ones(n_moves)
    bounds = Bounds(0, 1)

    if time.time() > deadline - 5:
        return [int(x) for x in base_numbers]

    try:
        res = milp(c=c_obj, constraints=constraints, integrality=integrality, bounds=bounds)
    except Exception:
        res = None

    best_selected_moves = []
    if res is not None and res.success:
        for i in range(n_moves):
            if res.x[i] > 0.5:
                best_selected_moves.append(moves[i])

    # Construct final solution
    final_sol = set(base_numbers)

    used_primes_in_moves = set()
    for ps, c, gain in best_selected_moves:
        for p in ps:
            used_primes_in_moves.add(p)
            final_sol.remove(prime_powers[p])

    for ps, c, gain in best_selected_moves:
        final_sol.add(c)

    result = [int(x) for x in final_sol]

    # Verification
    def verify(lst):
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if math.gcd(lst[i], lst[j]) != 1:
                    return False
        return True

    if not verify(result):
        result = [int(x) for x in base_numbers]

    return result
