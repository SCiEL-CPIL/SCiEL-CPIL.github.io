import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin (limit is 90s)

    MOD_MAX = 128
    moduli = list(range(2, MOD_MAX + 1))  # 2..128 inclusive, 127 values

    # Precompute gcd table for speed
    gcd_table = [[0] * (MOD_MAX + 1) for _ in range(MOD_MAX + 1)]
    for i in range(2, MOD_MAX + 1):
        for j in range(i, MOD_MAX + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    def disjoint(r1, m1, r2, m2):
        g = gcd_table[m1][m2]
        if g <= 1:
            return False  # intersect (CRT guarantees common element)
        return (r1 % g) != (r2 % g)

    def valid_residues(m, selected):
        """Return list of residues r in [0,m) that are disjoint from all selected."""
        out = []
        for r in range(m):
            ok = True
            for r_k, m_k in selected:
                if not disjoint(r, m, r_k, m_k):
                    ok = False
                    break
            if ok:
                out.append(r)
        return out

    def greedy(order, deterministic=True):
        selected = []
        for m in order:
            vr = valid_residues(m, selected)
            if not vr:
                continue
            if deterministic:
                r = vr[0]
            else:
                r = random.choice(vr)
            selected.append((r, m))
        return [[r, m] for r, m in selected]

    def most_constrained_greedy():
        """At each step, among remaining moduli, pick one with fewest valid residues (but >0)."""
        remaining = set(moduli)
        selected = []
        while remaining:
            if time.time() > deadline:
                break
            best_m = None
            best_vr = None
            best_count = None
            for m in remaining:
                vr = valid_residues(m, selected)
                c = len(vr)
                if c == 0:
                    continue
                if best_count is None or c < best_count:
                    best_count = c
                    best_m = m
                    best_vr = vr
                    if c == 1:
                        break
            if best_m is None:
                break
            r = best_vr[0]
            selected.append((r, best_m))
            remaining.discard(best_m)
        return [[r, m] for r, m in selected]

    def local_search(sol, max_rounds=200):
        sol = [list(x) for x in sol]
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            improved = False
            # 1. Try adding any unselected modulus
            selected_set = set(m for _, m in sol)
            for m in moduli:
                if m in selected_set:
                    continue
                vr = valid_residues(m, sol)
                if vr:
                    sol.append([vr[0], m])
                    improved = True
                    break
            if improved:
                continue
            # 2. Try 1-for-1 swap: remove one, see if we can add two
            for i in range(len(sol)):
                if time.time() > deadline:
                    break
                base = sol[:i] + sol[i+1:]
                selected_set = set(m for _, m in base)
                added = 0
                for m in moduli:
                    if m in selected_set:
                        continue
                    vr = valid_residues(m, base)
                    if vr:
                        base.append([vr[0], m])
                        selected_set.add(m)
                        added += 1
                if len(base) > len(sol):
                    sol = base
                    improved = True
                    break
            if not improved:
                break
        return sol

    best = []

    # Initial: single-prime constructions (star solutions)
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]:
        if time.time() > deadline:
            break
        multiples = [m for m in moduli if m % p == 0]
        k = min(len(multiples), p)
        sol = [[i % p, multiples[i]] for i in range(k)]
        if len(sol) > len(best):
            best = [list(x) for x in sol]

    # Greedy with various orders
    def num_divisors(n):
        c = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                c += 1
                if i != n // i:
                    c += 1
        return c

    orders = [
        list(range(2, MOD_MAX + 1)),
        list(range(MOD_MAX, 1, -1)),
        sorted(moduli, key=lambda x: -num_divisors(x)),
        sorted(moduli, key=lambda x: x.bit_count(), reverse=True),
        sorted(moduli, key=lambda x: -x.bit_count()),
    ]
    for order in orders:
        if time.time() > deadline:
            break
        sol = greedy(order, deterministic=True)
        if len(sol) > len(best):
            best = [list(x) for x in sol]

    # Most-constrained greedy (often the strongest single pass)
    sol = most_constrained_greedy()
    if len(sol) > len(best):
        best = [list(x) for x in sol]

    # Local search from best
    best = local_search(best, max_rounds=150)

    # Randomized restarts with local search
    for _ in range(200):
        if time.time() > deadline - 5.0:
            break
        random_order = moduli[:]
        random.shuffle(random_order)
        sol = greedy(random_order, deterministic=False)
        sol = local_search(sol, max_rounds=30)
        if len(sol) > len(best):
            best = sol

    # Final polish with more local search rounds
    if time.time() < deadline - 3.0:
        best = local_search(best, max_rounds=300)

    if not best:
        best = [[0, 2]]

    # Ensure canonical residues and valid output
    result = []
    for r, m in best:
        r = r % m
        if 0 <= r < m:
            result.append([r, m])
    return result
