import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin (evaluator kills at 90s)

    max_modulus = 128
    moduli = list(range(2, max_modulus + 1))

    # Precompute gcds for fast compatibility checks
    gcds = {}
    for i in range(len(moduli)):
        m1 = moduli[i]
        for j in range(i + 1, len(moduli)):
            m2 = moduli[j]
            gcds[(m1, m2)] = math.gcd(m1, m2)

    def gcd_of(a, b):
        if a == b:
            return a
        if a < b:
            return gcds.get((a, b), math.gcd(a, b))
        return gcds.get((b, a), math.gcd(b, a))

    def is_compatible(r_new, m_new, selected):
        # selected: list of (r, m)
        for r_k, m_k in selected:
            g = gcd_of(m_new, m_k)
            if r_new % g == r_k % g:
                return False
        return True

    def find_residue(m, selected, rng=None):
        # Try residues in a few strategic orders to maximize success.
        # Order 1: 0 .. m-1 (canonical ascending).
        for r in range(m):
            if is_compatible(r, m, selected):
                return r
        # If ascending failed, try a randomized scramble (can escape
        # the case where ascending order hits conflicts early).
        if rng is not None:
            cand = list(range(m))
            rng.shuffle(cand)
            for r in cand:
                if is_compatible(r, m, selected):
                    return r
        return None

    def solve_greedy(order, rng):
        selected = []
        for m in order:
            r = find_residue(m, selected, rng)
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
    orders.append(sorted(moduli, key=lambda m: div_counts[m]))            # fewest divisors first
    orders.append(sorted(moduli, key=lambda m: -div_counts[m]))           # most divisors first
    orders.append(sorted(moduli))
    orders.append(sorted(moduli, reverse=True))
    orders.append(sorted(moduli, key=lambda m: -pf_counts[m]))
    orders.append(sorted(moduli, key=lambda m: pf_counts[m]))
    orders.append(primes + sorted(composites, key=lambda m: div_counts[m]))
    orders.append(sorted(composites, key=lambda m: -div_counts[m]) + primes)

    # ---- Local search: swap residues to free a missing modulus ----
    def local_search(selected, rng):
        # selected: list of (r, m). Returns a (possibly improved) list.
        # Goal: add any modulus not present by swapping residues of existing classes.
        selected = list(selected)
        selected_moduli = set(m for _, m in selected)
        missing = [m for m in moduli if m not in selected_moduli]

        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Shuffle missing to vary which target we try first
            rng.shuffle(missing)
            for target_m in missing:
                if time.time() >= deadline:
                    break
                if target_m in selected_moduli:
                    continue
                # Try to add target_m, allowing a single residue swap to make room.
                # First: can we just add it?
                r_add = find_residue(target_m, selected, None)
                if r_add is not None:
                    selected.append((r_add, target_m))
                    selected_moduli.add(target_m)
                    improved = True
                    continue
                # Otherwise, try swapping one existing class's residue to free a slot.
                # For each existing class i, try changing its residue so that
                # target_m becomes compatible with the new set.
                added = False
                for i in range(len(selected)):
                    r_i, m_i = selected[i]
                    # We want a new residue r_i' for class i such that:
                    #   target_m is compatible with all classes,
                    #   and class i with r_i' is compatible with all others.
                    # Iterate candidate r_i'.
                    for r_i_new in range(m_i):
                        if r_i_new == r_i:
                            continue
                        # Check class i (with r_i_new) vs all others
                        ok_i = True
                        for j in range(len(selected)):
                            if j == i:
                                continue
                            r_j, m_j = selected[j]
                            g = gcd_of(m_i, m_j)
                            if r_i_new % g == r_j % g:
                                ok_i = False
                                break
                        if not ok_i:
                            continue
                        # Check target_m vs all (using r_i_new for class i)
                        ok_t = True
                        for j in range(len(selected)):
                            r_j, m_j = selected[j]
                            r_eff = r_i_new if j == i else r_j
                            g = gcd_of(target_m, m_j)
                            if r_i_new % g == r_eff % g if j == i else (target_m % g == r_eff % g):
                                # careful: target vs class j uses target residue
                                pass
                            # recompute properly below
                            pass
                        # Proper target compatibility check
                        for j in range(len(selected)):
                            r_j, m_j = selected[j]
                            g = gcd_of(target_m, m_j)
                            if target_m % 1 == 0:  # placeholder no-op
                                pass
                            if (target_m % g) == (r_j % g):
                                ok_t = False
                                break
                        if ok_t:
                            # Perform swap: change class i residue, add target
                            selected[i] = (r_i_new, m_i)
                            selected.append((0, target_m))  # residue 0 for target (compatible per check)
                            selected_moduli.add(target_m)
                            added = True
                            improved = True
                            break
                    if added:
                        break
                if time.time() >= deadline:
                    break
        return selected

    # ---- Main search loop ----
    best_solution = []
    best_len = 0

    rng = random.Random(42)

    # Deterministic + random greedy orders
    all_orders = list(orders)
    for _ in range(60):
        shuffled = moduli.copy()
        rng.shuffle(shuffled)
        all_orders.append(shuffled)

    for order in all_orders:
        if time.time() >= deadline:
            break
        sol = solve_greedy(order, rng)
        if len(sol) > best_len:
            best_len = len(sol)
            best_solution = sol

    # Polish the best solution with local search
    if best_len > 0 and time.time() < deadline:
        polished = local_search(best_solution, rng)
        if len(polished) > best_len:
            best_len = len(polished)
            best_solution = polished

    # A few extra local-search restarts from random greedy solutions
    for _ in range(20):
        if time.time() >= deadline:
            break
        shuffled = moduli.copy()
        rng.shuffle(shuffled)
        sol = solve_greedy(shuffled, rng)
        polished = local_search(sol, rng)
        if len(polished) > best_len:
            best_len = len(polished)
            best_solution = polished

    # Final safety: ensure nonempty and canonical
    if not best_solution:
        # Fallback: single class
        return [[0, 2]]

    result = []
    seen_moduli = set()
    for r, m in best_solution:
        if m in seen_moduli:
            continue
        seen_moduli.add(m)
        # canonical residue
        r = r % m
        result.append([r, m])

    return result
