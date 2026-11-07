import math
import time
import numpy as np
import random

def construct():
    start = time.time()
    deadline = start + 85.0
    N = 256

    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for p in range(2, int(math.isqrt(N)) + 1):
        if is_prime[p]:
            for m in range(p * p, N + 1, p):
                is_prime[m] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]

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

    pf_frozen = [frozenset(prime_factors[i]) for i in range(N + 1)]

    def is_valid_solution(sol_list):
        if not isinstance(sol_list, (list, tuple)) or len(sol_list) == 0:
            return False
        seen = set()
        for x in sol_list:
            if not isinstance(x, int) or isinstance(x, bool):
                return False
            if x < 1 or x > N or x in seen:
                return False
            seen.add(x)
        lst = list(sol_list)
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if math.gcd(lst[i], lst[j]) != 1:
                    return False
        return True

    best_result = None
    milp_optimal = False

    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        indices = list(range(2, N + 1))
        num_vars = len(indices)
        num_primes = len(primes)

        A = np.zeros((num_primes, num_vars), dtype=np.int8)
        for idx, p in enumerate(primes):
            for v_idx, n in enumerate(indices):
                if n % p == 0:
                    A[idx, v_idx] = 1

        c = np.array([-float(n) for n in indices])
        cons = LinearConstraint(A, lb=0, ub=1)
        bounds = Bounds(lb=0, ub=1)
        integrality = np.ones(num_vars, dtype=np.int8)

        time_left = max(1.0, deadline - time.time() - 15.0)
        res = milp(c, constraints=cons, integrality=integrality, bounds=bounds,
                   options={'time_limit': time_left})

        if res.x is not None:
            sol_set = {1}
            for v_idx, val in enumerate(res.x):
                if val > 0.5:
                    sol_set.add(indices[v_idx])
            if is_valid_solution(list(sol_set)):
                best_result = list(sol_set)
                if res.status == 0:
                    milp_optimal = True
    except Exception:
        pass

    if best_result is None:
        best_set = {1}
        used_primes = set()
        for n in range(N, 1, -1):
            if not (prime_factors[n] & used_primes):
                best_set.add(n)
                used_primes |= prime_factors[n]
        best_result = list(best_set)

    best_set = set(best_result)
    best_sum = sum(best_set)

    def local_search(sol_set, max_iters=500):
        nonlocal best_sum
        cur_set = set(sol_set)
        cur_sum = sum(cur_set)
        used = set()
        for s in cur_set:
            used |= prime_factors[s]

        for _ in range(max_iters):
            if time.time() > deadline - 3.0:
                break
            improved = False

            for n in range(N, 1, -1):
                if n not in cur_set and not (pf_frozen[n] & used):
                    cur_set.add(n)
                    used |= pf_frozen[n]
                    cur_sum += n
                    improved = True
                    break

            if not improved:
                bg, bm = 0, None
                for no in list(cur_set):
                    if no == 1: continue
                    rem = used - pf_frozen[no]
                    for ni in range(N, 1, -1):
                        if ni not in cur_set and not (pf_frozen[ni] & rem):
                            g = ni - no
                            if g > bg:
                                bg, bm = g, (no, ni)
                            if ni <= no: break
                if bm:
                    no, ni = bm
                    cur_set.remove(no); cur_set.add(ni)
                    used = (used - pf_frozen[no]) | pf_frozen[ni]
                    cur_sum += bg
                    improved = True

            if not improved:
                bg, bm = 0, None
                lst = [x for x in cur_set if x != 1]
                for i in range(len(lst)):
                    for j in range(i + 1, len(lst)):
                        a, b = lst[i], lst[j]
                        rem = used - (pf_frozen[a] | pf_frozen[b])
                        for ni in range(N, 1, -1):
                            if ni not in cur_set and not (pf_frozen[ni] & rem):
                                g = ni - a - b
                                if g > bg:
                                    bg, bm = g, (a, b, ni)
                                if ni <= a + b: break
                if bm:
                    a, b, ni = bm
                    cur_set.remove(a); cur_set.remove(b); cur_set.add(ni)
                    used = (used - pf_frozen[a] - pf_frozen[b]) | pf_frozen[ni]
                    cur_sum += bg
                    improved = True

            if not improved:
                bg, bm = 0, None
                for no in list(cur_set):
                    if no == 1: continue
                    rem = used - pf_frozen[no]
                    for n1 in range(N, 1, -1):
                        if n1 not in cur_set and not (pf_frozen[n1] & rem):
                            blocked = rem | pf_frozen[n1]
                            for n2 in range(N, 1, -1):
                                if n2 != n1 and n2 not in cur_set and not (pf_frozen[n2] & blocked):
                                    g = n1 + n2 - no
                                    if g > bg:
                                        bg, bm = g, (no, n1, n2)
                                    if n1 + n2 <= no: break
                        if n1 * 2 <= no: break
                if bm:
                    no, n1, n2 = bm
                    cur_set.remove(no); cur_set.add(n1); cur_set.add(n2)
                    used = (used - pf_frozen[no]) | pf_frozen[n1] | pf_frozen[n2]
                    cur_sum += bg
                    improved = True

            if not improved:
                break

        if cur_sum > best_sum:
            best_sum = cur_sum
            best_set = cur_set
        return cur_set

    local_search(best_set)

    random.seed(42)
    restarts = 0
    while time.time() < deadline - 5.0 and restarts < 100:
        restarts += 1
        perturbed = set(best_set)
        k = random.randint(2, 6)
        removable = [x for x in perturbed if x != 1]
        if len(removable) < k:
            continue
        to_remove = random.sample(removable, k)
        for x in to_remove:
            perturbed.remove(x)
        used = set()
        for s in perturbed:
            used |= prime_factors[s]
        for n in range(N, 1, -1):
            if n not in perturbed and not (prime_factors[n] & used):
                perturbed.add(n)
                used |= prime_factors[n]
        local_search(perturbed, max_iters=200)

    final = list(best_set)
    if 1 not in final:
        final.append(1)

    if not is_valid_solution(final):
        final = [1]
        used = set()
        for n in range(N, 1, -1):
            if not (prime_factors[n] & used):
                final.append(n)
                used |= prime_factors[n]
        if not is_valid_solution(final):
            final = [1]

    return [int(x) for x in final]
