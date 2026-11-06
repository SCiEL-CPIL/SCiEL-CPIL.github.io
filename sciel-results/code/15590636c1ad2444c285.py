"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random
import sys

_START_TIME = None


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _disjoint(r1, m1, r2, m2):
    g = _gcd(m1, m2)
    if g == 1:
        return False
    return (r2 - r1) % g != 0


def _can_add(r_new, m_new, existing):
    for (r, m) in existing:
        if not _disjoint(r_new, m_new, r, m):
            return False
    return True


def _prime_factors(n):
    factors = set()
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.add(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors


def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

    deadline = _START_TIME + 85.0

    moduli = list(range(2, 129))
    n = len(moduli)
    mod_to_idx = {m: i for i, m in enumerate(moduli)}

    # Precompute gcd table and prime factor sets
    gcd_table = [[1] * n for _ in range(n)]
    for i in range(n):
        gcd_table[i][i] = moduli[i]
        for j in range(i + 1, n):
            g = _gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    prime_sets = [_prime_factors(m) for m in moduli]

    # Constraint: number of other moduli sharing a factor
    constraint = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if gcd_table[i][j] > 1:
                constraint[i] += 1
                constraint[j] += 1

    # Build orderings
    order_most = sorted(range(n), key=lambda i: (-constraint[i], -moduli[i]))
    order_least = sorted(range(n), key=lambda i: (constraint[i], moduli[i]))
    order_desc = list(range(n - 1, -1, -1))
    order_asc = list(range(n))
    order_by_prime_count = sorted(range(n), key=lambda i: (-len(prime_sets[i]), -moduli[i]))

    def greedy_pass(order, residue_mode, rng=None):
        current = []
        used = set()
        for idx in order:
            m = moduli[idx]
            if m in used:
                continue
            if not current:
                r = rng.randrange(m) if (residue_mode == 1 and rng) else 0
                current.append([r, m])
                used.add(m)
                continue

            if residue_mode == 0:
                for r in range(m):
                    if _can_add(r, m, current):
                        current.append([r, m])
                        used.add(m)
                        break
            elif residue_mode == 1 and rng:
                residues = list(range(m))
                rng.shuffle(residues)
                for r in residues:
                    if _can_add(r, m, current):
                        current.append([r, m])
                        used.add(m)
                        break
            else:
                # residue_mode == 2: minimize future blocking
                best_r = None
                best_score = None
                for r in range(m):
                    if not _can_add(r, m, current):
                        continue
                    score = 0
                    for j in order:
                        m2 = moduli[j]
                        if m2 in used or m2 == m:
                            continue
                        g = gcd_table[idx][j]
                        if g > 1:
                            score += m2 // g
                    if best_score is None or score < best_score:
                        best_score = score
                        best_r = r
                if best_r is not None:
                    current.append([best_r, m])
                    used.add(m)
        return current

    def local_search(sol):
        if not sol:
            return sol
        sol = [x[:] for x in sol]
        used = set(m for _, m in sol)
        improved = True
        while improved:
            improved = False
            # Try adding missing moduli
            for m in moduli:
                if m in used:
                    continue
                for r in range(m):
                    if _can_add(r, m, sol):
                        sol.append([r, m])
                        used.add(m)
                        improved = True
                        break
                if improved:
                    break
            # Try swapping residues to free up a modulus
            if not improved:
                for i in range(len(sol)):
                    r1, m1 = sol[i]
                    for r_new in range(m1):
                        if r_new == r1:
                            continue
                        old = sol[i]
                        sol[i] = [r_new, m1]
                        ok = all(_disjoint(r_new, m1, sol[j][0], sol[j][1])
                                 for j in range(len(sol)) if j != i)
                        if ok:
                            for m_add in moduli:
                                if m_add in used:
                                    continue
                                for r_add in range(m_add):
                                    if _can_add(r_add, m_add, sol):
                                        sol.append([r_add, m_add])
                                        used.add(m_add)
                                        improved = True
                                        break
                                if improved:
                                    break
                            if improved:
                                break
                        else:
                            sol[i] = old
                    if improved:
                        break
                    else:
                        sol[i] = old
        return sol

    best_solution = []
    best_size = 0

    def update(sol):
        nonlocal best_solution, best_size
        if len(sol) > best_size:
            best_size = len(sol)
            best_solution = [x[:] for x in sol]

    # Phase 1: deterministic greedy passes
    deterministic_configs = [
        (order_most, 0), (order_most, 2),
        (order_least, 0), (order_least, 2),
        (order_desc, 0), (order_desc, 2),
        (order_asc, 0), (order_asc, 2),
        (order_by_prime_count, 0), (order_by_prime_count, 2),
    ]
    for order, mode in deterministic_configs:
        if time.time() > deadline:
            break
        sol = greedy_pass(order, mode)
        sol = local_search(sol)
        update(sol)

    # Phase 2: randomized restarts
    rng = random.Random(42)
    max_restarts = 200
    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        # Random ordering biased toward constrained
        perm = list(range(n))
        rng.shuffle(perm)
        # Bias: with 50% chance, use most-constrained order
        if rng.random() < 0.3:
            perm = order_most[:]
            rng.shuffle(perm)  # slight randomization
        elif rng.random() < 0.3:
            perm = order_least[:]
            rng.shuffle(perm)

        mode = rng.choice([0, 1, 2])
        sol = greedy_pass(perm, mode, rng)
        sol = local_search(sol)
        update(sol)

    # Phase 3: targeted improvement - try to add moduli that share a prime
    # with the current solution
    if time.time() < deadline:
        cur = [x[:] for x in best_solution]
        used_mods = set(m for _, m in cur)
        # For each missing modulus, try harder
        for m in moduli:
            if m in used_mods:
                continue
            # Try all residues
            for r in range(m):
                if _can_add(r, m, cur):
                    cur.append([r, m])
                    used_mods.add(m)
                    break
        update(cur)

    if not best_solution:
        return [[1, 2]]

    return best_solution
# EVOLVE-BLOCK-END
