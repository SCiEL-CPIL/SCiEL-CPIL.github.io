"""Combined exact MILP + local search for pairwise-coprime max-sum task."""
import math
import time
import sys

def construct():
    start = time.time()
    deadline = start + 85.0  # 5s safety margin under 90s limit

    N = 256

    # Precompute prime factors for each number 1..N
    # F[x] = frozenset of distinct prime factors of x
    F = [frozenset()] * (N + 1)
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

    # Collect all primes that appear as factors
    all_primes = set()
    for i in range(2, N + 1):
        all_primes.update(F[i])
    all_primes = sorted(all_primes)
    P = len(all_primes)

    # Try MILP first for exact solution
    best_sum = 0
    best_subset = []

    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np

        # Variables: x_i in {0,1} for i in 1..N
        # Maximize sum(i * x_i)
        # Subject to: for each prime p, sum(x_i for i where p in F[i]) <= 1
        #             x_i in {0,1}

        n_vars = N  # x_1, x_2, ..., x_N (index i-1 for number i)

        # Objective: maximize sum(i * x_i) => minimize -sum(i * x_i)
        c = np.array([-float(i) for i in range(1, N + 1)])

        # Constraints: for each prime p, sum of x_i where p divides i <= 1
        # Build constraint matrix
        rows = []
        for p in all_primes:
            row = np.zeros(N)
            for i in range(1, N + 1):
                if p in F[i]:
                    row[i - 1] = 1.0
            rows.append(row)

        if rows:
            A = np.array(rows)
            lb = np.zeros(P)
            ub = np.ones(P)
            constraints = LinearConstraint(A, lb, ub)
        else:
            constraints = LinearConstraint(np.zeros((0, N)), np.zeros(0), np.zeros(0))

        # Bounds: 0 <= x_i <= 1
        integrality = np.ones(N)  # all binary
        bounds = Bounds(0, 1)

        res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)

        if res.status == 0 and res.x is not None:
            subset = [i + 1 for i in range(N) if res.x[i] > 0.5]
            s = sum(subset)
            if s > best_sum:
                best_sum = s
                best_subset = subset

    except Exception:
        pass

    # If MILP didn't work or gave poor result, use greedy + local search as fallback
    if not best_subset or best_sum < 100:
        # Greedy descending
        selected = []
        selected_factors = set()
        for a in range(N, 0, -1):
            if F[a].isdisjoint(selected_factors):
                selected.append(a)
                selected_factors.update(F[a])

        s = sum(selected)
        if s > best_sum:
            best_sum = s
            best_subset = selected

    # Local search: try to improve by swapping elements
    S = set(best_subset)
    improved = True
    max_iters = 50

    while improved and max_iters > 0:
        improved = False
        max_iters -= 1

        if time.time() > deadline - 5:
            break

        S_factors = set()
        for x in S:
            S_factors.update(F[x])

        # Try adding any number that doesn't conflict
        added = False
        for x in range(1, N + 1):
            if x not in S and F[x].isdisjoint(S_factors):
                S.add(x)
                S_factors.update(F[x])
                improved = True
                added = True
                break

        if added:
            continue

        # Try 1-exchange: remove a, add best set of replacements
        for a in list(S):
            if time.time() > deadline - 5:
                break

            rest = S - {a}
            rest_factors = set()
            for x in rest:
                rest_factors.update(F[x])

            # Find candidates coprime to rest_factors
            candidates = [x for x in range(1, N + 1) if x not in rest and F[x].isdisjoint(rest_factors)]

            if not candidates:
                continue

            # Greedy pick best subset from candidates
            cand_factors = set()
            cand_subset = []
            for x in sorted(candidates, reverse=True):
                if F[x].isdisjoint(cand_factors):
                    cand_subset.append(x)
                    cand_factors.update(F[x])

            if sum(cand_subset) > a:
                S = rest.union(cand_subset)
                improved = True
                break

        # Try 2-exchange: remove two, add best set
        if not improved:
            S_list = list(S)
            for i in range(len(S_list)):
                if time.time() > deadline - 5:
                    break
                for j in range(i + 1, len(S_list)):
                    a, b = S_list[i], S_list[j]
                    rest = S - {a, b}
                    rest_factors = set()
                    for x in rest:
                        rest_factors.update(F[x])

                    candidates = [x for x in range(1, N + 1) if x not in rest and F[x].isdisjoint(rest_factors)]

                    if not candidates:
                        continue

                    cand_factors = set()
                    cand_subset = []
                    for x in sorted(candidates, reverse=True):
                        if F[x].isdisjoint(cand_factors):
                            cand_subset.append(x)
                            cand_factors.update(F[x])

                    if sum(cand_subset) > a + b:
                        S = rest.union(cand_subset)
                        improved = True
                        break
                if improved:
                    break

    new_sum = sum(S)
    if new_sum > best_sum:
        best_sum = new_sum
        best_subset = list(S)

    # Final verification
    result = sorted(best_subset)

    # Verify pairwise coprimality
    valid = True
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if math.gcd(result[i], result[j]) != 1:
                valid = False
                break
        if not valid:
            break

    if not valid:
        # Fallback to just primes + 1
        result = [1]
        for p in all_primes:
            if p <= N:
                result.append(p)

    # Ensure distinct, in range, proper types
    final = []
    seen = set()
    for x in result:
        if isinstance(x, int) and not isinstance(x, bool) and 1 <= x <= N and x not in seen:
            seen.add(x)
            final.append(int(x))

    if not final:
        final = [1]

    return final
