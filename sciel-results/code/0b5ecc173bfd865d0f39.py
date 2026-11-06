import sys
from math import factorial
from functools import lru_cache
import time

def construct():
    start_time = time.time()
    timeout = 55.0

    target = factorial(12)  # 479001600
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    total_exp = (10, 5, 2, 1, 1)

    # Generate all divisors of 12! that are > 12 and < target
    divisors = []
    for e2 in range(11):
        for e3 in range(6):
            for e5 in range(3):
                for e7 in range(2):
                    for e11 in range(2):
                        val = (2 ** e2) * (3 ** e3) * (5 ** e5) * (7 ** e7) * (11 ** e11)
                        if 12 < val < target:
                            divisors.append((val, (e2, e3, e5, e7, e11)))

    divisors.sort(key=lambda x: x[0])
    n_divs = len(divisors)
    divisor_vals = [d[0] for d in divisors]
    divisor_exps = [d[1] for d in divisors]

    MAX_FACTORS = 12

    # Precompute suffix sum of exponents for pruning
    suffix_exp = [[0]*5 for _ in range(n_divs + 1)]
    for i in range(n_divs - 1, -1, -1):
        for j in range(5):
            suffix_exp[i][j] = suffix_exp[i+1][j] + divisor_exps[i][j]

    best_solution = None
    best_max_factor = target + 1

    def dfs(start_idx, rem_exp, count, path):
        nonlocal best_solution, best_max_factor

        if time.time() - start_time > timeout:
            return

        if count >= MAX_FACTORS:
            return

        if all(r == 0 for r in rem_exp):
            if count > 0:
                current_max = max(path)
                if current_max < best_max_factor:
                    best_max_factor = current_max
                    best_solution = path[:]
            return

        # Pruning: check if remaining exponents can be satisfied by available candidates
        for j in range(5):
            if suffix_exp[start_idx][j] < rem_exp[j]:
                return

        for i in range(start_idx, n_divs):
            val = divisor_vals[i]
            exp = divisor_exps[i]

            # Check if this divisor's exponents fit in rem_exp
            ok = True
            for j in range(5):
                if exp[j] > rem_exp[j]:
                    ok = False
                    break
            if not ok:
                continue

            # Pruning: if we have a best solution and val >= best_max_factor, stop
            if best_solution is not None and val >= best_max_factor:
                break

            new_rem = tuple(rem_exp[j] - exp[j] for j in range(5))
            path.append(val)
            dfs(i + 1, new_rem, count + 1, path)
            path.pop()

    dfs(0, total_exp, 0, [])

    if best_solution is not None:
        return best_solution
    return [target]

# EVOLVE-BLOCK-END
