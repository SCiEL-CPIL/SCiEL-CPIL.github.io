import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin before 90s kill

    MAX_MOD = 128
    moduli = list(range(2, MAX_MOD + 1))
    n_moduli = len(moduli)

    # Precompute gcds
    gcd_cache = {}
    for i in range(n_moduli):
        m1 = moduli[i]
        for j in range(i + 1, n_moduli):
            m2 = moduli[j]
            gcd_cache[(m1, m2)] = math.gcd(m1, m2)

    def gcd_of(a, b):
        if a == b:
            return a
        if a < b:
            return gcd_cache.get((a, b), math.gcd(a, b))
        return gcd_cache.get((b, a), math.gcd(b, a))

    def is_compatible(r_new, m_new, selected):
        for r_k, m_k in selected:
            g = gcd_of(m_new, m_k)
            if (r_new - r_k) % g == 0:
                return False
        return True

    def find_residue(m, selected):
        for r in range(m):
            if is_compatible(r, m, selected):
                return r
        return None

    def solve_greedy(order):
        selected = []
        for m in order:
            r = find_residue(m, selected)
            if r is not None:
                selected.append((r, m))
        return selected

    # ---- Heuristic ordering keys ----
    def num_divisors(n):
        count = 0
        for i in range(1, int(math.isqrt(n)) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count

    def count_prime_factors(n):
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count

    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(math.isqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    div_counts = {m: num_divisors(m) for m in moduli}
    pf_counts = {m: count_prime_factors(m) for m in moduli}
    primes = [m for m in moduli if is_prime(m)]
    composites = [m for m in moduli if not is_prime(m)]

    orders = []
    orders.append(sorted(moduli, key=lambda m: div_counts[m]))
    orders.append(sorted(moduli, key=lambda m: -div_counts[m]))
    orders.append(sorted(moduli))
    orders.append(sorted(moduli, reverse=True))
    orders.append(sorted(moduli, key=lambda m: -pf_counts[m]))
    orders.append(sorted(moduli, key=lambda m: pf_counts[m]))
    orders.append(primes + sorted(composites, key=lambda m: div_counts[m]))
    orders.append(sorted(composites, key=lambda m: -div_counts[m]) + primes)

    # ---- Local search: add missing moduli via residue swaps ----
    def local_search(selected, rng, time_budget=3.0):
        selected = list(selected)
        selected_moduli = set(m for _, m in selected)
        missing = [m for m in moduli if m not in selected_moduli]
        start = time.time()

        improved = True
        while improved and time.time() - start < time_budget and time.time() < deadline:
            improved = False
            rng.shuffle(missing)
            for target_m in missing:
                if time.time() - start >= time_budget or time.time() >= deadline:
                    break
                if target_m in selected_moduli:
                    continue
                r_add = find_residue(target_m, selected)
                if r_add is not None:
                    selected.append((r_add, target_m))
                    selected_moduli.add(target_m)
                    improved = True
                    continue
                added = False
                for i in range(len(selected)):
                    r_i, m_i = selected[i]
                    for r_i_new in range(m_i):
                        if r_i_new == r_i:
                            continue
                        ok_i = True
                        for j in range(len(selected)):
                            if j == i:
                                continue
                            r_j, m_j = selected[j]
                            g = gcd_of(m_i, m_j)
                            if (r_i_new - r_j) % g == 0:
                                ok_i = False
                                break
                        if not ok_i:
                            continue
                        ok_t = True
                        for j in range(len(selected)):
                            r_j, m_j = selected[j]
                            g = gcd_of(target_m, m_j)
                            if j == i:
                                r_eff = r_i_new
                            else:
                                r_eff = r_j
                            if (0 - r_eff) % g == 0:  # check if residue 0 works; will verify later
                                pass  # we need to find a valid residue, not just 0
                        # Find valid residue for target_m
                        r_t = None
                        for rt in range(target_m):
                            compatible = True
                            for j in range(len(selected)):
                                r_j, m_j = selected[j]
                                g = gcd_of(target_m, m_j)
                                r_eff = r_i_new if j == i else r_j
                                if (rt - r_eff) % g == 0:
                                    compatible = False
                                    break
                            if compatible:
                                r_t = rt
                                break
                        if r_t is not None:
                            selected[i] = (r_i_new, m_i)
                            selected.append((r_t, target_m))
                            selected_moduli.add(target_m)
                            added = True
                            improved = True
                            break
                    if added:
                        break
                if time.time() - start >= time_budget or time.time() >= deadline:
                    break
        return selected

    # ---- Backtracking search ----
    def backtrack_search(remaining, initial_selected, time_budget=5.0):
        selected = list(initial_selected)
        start = time.time()

        def bt(idx):
            if time.time() - start >= time_budget or time.time() >= deadline:
                return
            if idx >= len(remaining):
                return
            m = remaining[idx]
            for r in range(m):
                if is_compatible(r, m, selected):
                    selected.append((r, m))
                    bt(idx + 1)
                    selected.pop()
                if time.time() - start >= time_budget or time.time() >= deadline:
                    return
        bt(0)
        return selected

    # ---- Main search loop ----
    best_solution = []
    best_len = 0

    rng = random.Random(42)

    all_orders = list(orders)
    for _ in range(3000):
        shuffled = moduli.copy()
        rng.shuffle(shuffled)
        all_orders.append(shuffled)

    for order in all_orders:
        if time.time() >= deadline:
            break
        sol = solve_greedy(order)
        if len(sol) > best_len:
            best_len = len(sol)
            best_solution = sol

    # Polish the best solution with local search
    if best_len > 0 and time.time() < deadline:
        polished = local_search(best_solution, rng, time_budget=4.0)
        if len(polished) > best_len:
            best_len = len(polished)
            best_solution = polished

    # Extra local-search restarts from random greedy solutions
    for _ in range(300):
        if time.time() >= deadline:
            break
        shuffled = moduli.copy()
        rng.shuffle(shuffled)
        sol = solve_greedy(shuffled)
        polished = local_search(sol, rng, time_budget=1.5)
        if len(polished) > best_len:
            best_len = len(polished)
            best_solution = polished

    # Backtracking polish on best solution
    if best_len > 0 and time.time() < deadline:
        remaining = [m for m in moduli if m not in set(mm for _, mm in best_solution)]
        bt_sol = backtrack_search(remaining, best_solution, time_budget=6.0)
        if len(bt_sol) > best_len:
            best_len = len(bt_sol)
            best_solution = bt_sol

    # Final safety: ensure nonempty and canonical
    if not best_solution:
        return [[0, 2]]

    result = []
    seen_moduli = set()
    for r, m in best_solution:
        if m in seen_moduli:
            continue
        seen_moduli.add(m)
        r = r % m
        result.append([r, m])

    return result
