import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin (limit is 90s)

    def now():
        return time.time()

    def over():
        return now() > deadline

    MOD_MAX = 128
    moduli = list(range(2, MOD_MAX + 1))  # 2..128 inclusive

    # Precompute gcd table
    gcd_table = [[0] * (MOD_MAX + 1) for _ in range(MOD_MAX + 1)]
    for i in range(2, MOD_MAX + 1):
        for j in range(i, MOD_MAX + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    # For each modulus m and residue r, precompute a bitset (over modulus index)
    # of moduli that CONFLICT with (r, m) — i.e., cannot coexist.
    # conflict_mask[m][r] is a Python int bitset over modulus indices 0..126 (m-2).
    # We index moduli by their position in the `moduli` list: idx = m-2.
    def m_to_idx(m):
        return m - 2

    # conflict_mask[m][r]: bit i set means modulus (i+2) conflicts with residue r mod m
    conflict_mask = {}
    for m in moduli:
        cm = [0] * m
        for r in range(m):
            mask = 0
            for k in moduli:
                if k == m:
                    continue
                g = gcd_table[m][k]
                if g > 1:
                    # conflict if r % g == r_k % g for some r_k in [0,k)
                    # For a given (m,r,k), the set of residues r_k in [0,k) that conflict
                    # are those with r_k % g == r % g. Such residues exist iff there's
                    # r_k in [0,k) with r_k % g == r % g. Since g<=k (g divides k, g<=k),
                    # there's always at least one (r % g) in [0,g) <= k-1? g<=k so r%g < g <= k,
                    # yes r%g is in [0,k). So conflict always exists when g>1.
                    # Therefore (r,m) conflicts with ALL residues of modulus k when g>1.
                    mask |= (1 << m_to_idx(k))
            cm[r] = mask
        conflict_mask[m] = cm

    # A modulus k is "always conflicting" with m if gcd(m,k)>1 (since some residue always conflicts).
    # Actually for disjointness we need r1%g != r2%g. For a fixed (r,m), modulus k (g>1)
    # has SOME residues that are disjoint (those r_k with r_k%g != r%g) as long as g < k.
    # If g == k (i.e., k divides m and k<=m, so k|m and k<=m), then all r_k in [0,k) have r_k%k = r_k,
    # and r%k is one specific value; residues r_k != r%k are disjoint. So there are k-1 disjoint residues.
    # So every modulus with gcd>1 has at least one disjoint residue. Good.

    # valid_residues(m, selected): residues r in [0,m) such that for all selected (r_k,m_k),
    # either gcd(m,m_k)==1 is impossible (they'd intersect) ... wait.
    # Disjoint condition: for each selected (r_k,m_k):
    #   g = gcd(m, m_k)
    #   if g == 1: they intersect -> INVALID (r cannot be disjoint)
    #   else: need r % g != r_k % g
    def valid_residues(m, selected):
        # selected is list of [r_k, m_k]
        out = []
        for r in range(m):
            ok = True
            for r_k, m_k in selected:
                g = gcd_table[m][m_k]
                if g <= 1:
                    ok = False
                    break
                if (r % g) == (r_k % g):
                    ok = False
                    break
            if ok:
                out.append(r)
        return out

    def greedy(order, deterministic=True, rng=None):
        selected = []
        for m in order:
            if over():
                break
            vr = valid_residues(m, selected)
            if not vr:
                continue
            if deterministic:
                r = vr[0]
            else:
                r = rng.choice(vr)
            selected.append([r, m])
        return selected

    def most_constrained_greedy(rng=None):
        remaining = set(moduli)
        selected = []
        while remaining:
            if over():
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
            selected.append([r, best_m])
            remaining.discard(best_m)
        return selected

    def local_search(sol, max_rounds=200, rng=None):
        sol = [list(x) for x in sol]
        for _ in range(max_rounds):
            if over():
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
            # 2. Try 1-for-1 swap: remove i, add up to two
            for i in range(len(sol)):
                if over():
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
            # 3. Try 2-for-1 swap: remove two, add up to three
            if not improved and len(sol) >= 2:
                for i in range(len(sol)):
                    if over():
                        break
                    for j in range(i+1, len(sol)):
                        base = sol[:i] + sol[i+1:j] + sol[j+1:]
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
                    if improved:
                        break
            # 4. Try residue re-optimization
            if not improved and len(sol) > 0:
                if rng:
                    idx = rng.randrange(len(sol))
                else:
                    idx = 0
                r_old, m_old = sol[idx]
                vr = valid_residues(m_old, [x for i2, x in enumerate(sol) if i2 != idx])
                if len(vr) > 1:
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
    primes = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113]
    for p in primes:
        if over():
            break
        multiples = [m for m in moduli if m % p == 0]
        k = min(len(multiples), p)
        sol = [[i % p, multiples[i]] for i in range(k)]
        if len(sol) > len(best):
            best = [list(x) for x in sol]

    def num_divisors(n):
        c = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                c += 1
                if i != n // i:
                    c += 1
        return c

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
        sorted(moduli, key=lambda x: scores[x], reverse=True),
        sorted(moduli, key=lambda x: scores[x]),
    ]
    for order in orders:
        if over():
            break
        sol = greedy(order, deterministic=True)
        if len(sol) > len(best):
            best = [list(x) for x in sol]

    sol = most_constrained_greedy()
    if len(sol) > len(best):
        best = [list(x) for x in sol]

    # Randomized restarts with local search
    for _ in range(600):
        if over():
            break
        rng = random.Random()
        random_order = moduli[:]
        rng.shuffle(random_order)
        sol = greedy(random_order, deterministic=False, rng=rng)
        if len(sol) > len(best):
            best = [list(x) for x in sol]
        sol2 = local_search(sol, max_rounds=30, rng=rng)
        if len(sol2) > len(best):
            best = [list(x) for x in sol2]

    # Local search from best
    best = local_search(best, max_rounds=300)

    # More randomized + local search
    for _ in range(300):
        if over():
            break
        rng = random.Random()
        random_order = moduli[:]
        rng.shuffle(random_order)
        sol = greedy(random_order, deterministic=False, rng=rng)
        sol = local_search(sol, max_rounds=50, rng=rng)
        if len(sol) > len(best):
            best = [list(x) for x in sol]

    # Final polish
    if now() < deadline - 3.0:
        best = local_search(best, max_rounds=500)

    if not best:
        best = [[0, 2]]

    # Ensure canonical residues
    result = []
    for r, m in best:
        r = r % m
        if 0 <= r < m:
            result.append([r, m])
    return result
