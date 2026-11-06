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
            return False  # intersect (CRT)
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

    def greedy(order, deterministic=True, rng=None):
        selected = []
        for m in order:
            vr = valid_residues(m, selected)
            if not vr:
                continue
            if deterministic:
                r = vr[0]
            else:
                r = rng.choice(vr)
            selected.append((r, m))
        return [[r, m] for r, m in selected]

    def most_constrained_greedy(rng=None):
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
            if rng:
                r = rng.choice(best_vr)
            else:
                r = best_vr[0]
            selected.append((r, best_m))
            remaining.discard(best_m)
        return [[r, m] for r, m in selected]

    def local_search(sol, max_rounds=200, rng=None):
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
            # 2. Try 1-for-1 swap: remove i, add two
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
            # 3. Try residue re-optimization: for a random selected modulus, try a different residue
            if not improved and len(sol) > 0:
                if rng:
                    idx = rng.randrange(len(sol))
                else:
                    idx = 0
                r_old, m_old = sol[idx]
                vr = valid_residues(m_old, [x for i, x in enumerate(sol) if i != idx])
                if len(vr) > 1:
                    # Try each alternative residue to see if it enables adding more
                    for r_new in vr:
                        if r_new == r_old:
                            continue
                        trial = list(sol)
                        trial[idx] = [r_new, m_old]
                        selected_set = set(m for _, m in trial)
                        added = 0
                        for m in moduli:
                            if m in selected_set:
                                continue
                            vr2 = valid_residues(m, trial)
                            if vr2:
                                trial.append([vr2[0], m])
                                selected_set.add(m)
                                added += 1
                        if len(trial) > len(sol):
                            sol = trial
                            improved = True
                            break
            if not improved:
                break
        return sol

    best = []

    # Initial: single-prime constructions
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]:
        if time.time() > deadline:
            break
        multiples = [m for m in moduli if m % p == 0]
        k = len(multiples)
        if k > p:
            k = p
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

    # Precompute score-based ordering (sum of gcds)
    scores = {}
    for m in moduli:
        s = 0
        for k in moduli:
            if m != k:
                s += gcd_table[m][k]
        scores[m] = s

    orders = [
        list(range(2, MOD_MAX + 1)),
        list(range(MOD_MAX, 1, -1)),
        sorted(moduli, key=lambda x: -num_divisors(x)),
        sorted(moduli, key=lambda x: x.bit_count(), reverse=True),
        sorted(moduli, key=lambda x: -x.bit_count()),
        sorted(moduli, key=lambda x: x),
        sorted(moduli, key=lambda x: scores[x], reverse=True),
        sorted(moduli, key=lambda x: scores[x]),
    ]
    for order in orders:
        if time.time() > deadline:
            break
        sol = greedy(order, deterministic=True)
        if len(sol) > len(best):
            best = [list(x) for x in sol]

    # Most-constrained greedy
    sol = most_constrained_greedy()
    if len(sol) > len(best):
        best = [list(x) for x in sol]

    # Randomized restarts with local search
    for _ in range(500):
        if time.time() > deadline:
            break
        rng = random.Random()
        random_order = moduli[:]
        rng.shuffle(random_order)
        sol = greedy(random_order, deterministic=False, rng=rng)
        if len(sol) > len(best):
            best = [list(x) for x in sol]
        # Apply local search to improve
        sol2 = local_search(sol, max_rounds=30, rng=rng)
        if len(sol2) > len(best):
            best = [list(x) for x in sol2]

    # Local search from best
    best = local_search(best, max_rounds=200)

    # More randomized + local search
    for _ in range(200):
        if time.time() > deadline:
            break
        rng = random.Random()
        random_order = moduli[:]
        rng.shuffle(random_order)
        sol = greedy(random_order, deterministic=False, rng=rng)
        sol = local_search(sol, max_rounds=50, rng=rng)
        if len(sol) > len(best):
            best = [list(x) for x in sol]

    # Final polish with more rounds
    if time.time() < deadline - 3.0:
        best = local_search(best, max_rounds=400)

    if not best:
        best = [[0, 2]]

    # Ensure canonical residues
    result = []
    for r, m in best:
        r = r % m
        if 0 <= r < m:
            result.append([r, m])
    return result
