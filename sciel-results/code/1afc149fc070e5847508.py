"""Disjoint residue construction: maximize sum 1/m with pairwise disjointness.

Combines a multi-strategy search skeleton (structured base packing, greedy,
randomized multi-start + local search) with a fast bitmask validity core:
for each candidate modulus m we intersect the "forbidden residue" masks induced
by already-selected classes (r must not equal r2 mod gcd(m, m2)).  This makes
validity checks O(n) bit-ops instead of O(n*m), so local search can run far
more iterations in the same wall-clock budget.
"""
import time
import math
import random
from fractions import Fraction


def _disjoint(m1, r1, m2, r2):
    g = math.gcd(m1, m2)
    return (r1 % g) != (r2 % g)


def _is_valid(classes):
    if not classes:
        return True
    moduli = [c[0] for c in classes]
    if len(moduli) != len(set(moduli)):
        return False
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            if not _disjoint(classes[i][0], classes[i][1],
                             classes[j][0], classes[j][1]):
                return False
    return True


def _score(classes):
    total = Fraction(0, 1)
    for m, r in classes:
        total += Fraction(1, m)
    return float(total)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before the 150s kill

    MODULI = list(range(20, 201))
    MOD_IDX = {m: i for i, m in enumerate(MODULI)}
    N_MOD = len(MODULI)

    # Precompute gcd table between every pair of moduli.
    gcd_table = [[0] * N_MOD for _ in range(N_MOD)]
    for i in range(N_MOD):
        for j in range(i, N_MOD):
            g = math.gcd(MODULI[i], MODULI[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    # all_mask[m] = (1 << m) - 1 : bits 0..m-1 set.
    all_mask = {m: (1 << m) - 1 for m in MODULI}

    def compute_valid_mask(m, classes):
        """Bitmask of residues r in [0,m) that keep m disjoint from `classes`.

        Returns (valid_mask, count).  If any selected class has gcd 1 with m,
        every residue is forbidden (valid_mask = 0).
        """
        valid = all_mask[m]
        if valid == 0:
            return 0, 0
        i = MOD_IDX[m]
        for (m2, r2) in classes:
            g = gcd_table[i][MOD_IDX[m2]]
            if g == 1:
                return 0, 0
            r0 = r2 % g
            # Forbidden residues: r in [0,m) with r == r0 (mod g)
            forbidden = 0
            for r in range(r0, m, g):
                forbidden |= (1 << r)
            valid &= ~forbidden
            if valid == 0:
                return 0, 0
        return valid, bin(valid).count('1')

    def pick_residue(valid_mask, m, classes, mode):
        """Pick a residue from valid_mask.

        mode 0: first (smallest)
        mode 1: last (largest)
        mode 2: most-constraining-aware -> pick the residue that, when m is
                added, leaves the largest minimum "validity" for the smallest
                not-yet-used moduli (preserves future flexibility).  This is a
                cheap greedy surrogate for a global flexibility objective.
        mode 3: random
        """
        if valid_mask == 0:
            return None
        if mode == 3:
            bits = [r for r in range(m) if (valid_mask >> r) & 1]
            return random.choice(bits)
        if mode == 0:
            r = 0
            while (valid_mask >> r) & 1 == 0:
                r += 1
            return r
        if mode == 1:
            r = m - 1
            while (valid_mask >> r) & 1 == 0:
                r -= 1
            return r

        # mode 2: flexibility-aware.  For a few candidate residues, measure how
        # much they constrain the smallest few unused moduli and keep the best.
        used = set(m2 for m2, _ in classes)
        small_unused = [mm for mm in MODULI if mm not in used][:12]
        candidates = []
        r = 0
        while (valid_mask >> r) & 1:
            candidates.append(r)
            r += 1
            if len(candidates) >= 10:
                break
        best_r = candidates[0]
        best_flex = -1
        for cand in candidates:
            flex = 0
            for mm in small_unused:
                vm, cnt = compute_valid_mask(mm, classes + [[m, cand]])
                flex += cnt
            if flex > best_flex:
                best_flex = flex
                best_r = cand
        return best_r

    def greedy_construct(mode):
        classes = []
        for m in MODULI:
            vm, cnt = compute_valid_mask(m, classes)
            if cnt > 0:
                r = pick_residue(vm, m, classes, mode)
                classes.append([m, r])
        return classes

    def local_search(classes, max_iter=4000):
        best = [list(c) for c in classes]
        best_score = _score(best)
        used = set(m for m, _ in best)

        for _ in range(max_iter):
            if time.time() > deadline:
                break
            improved = False

            # (a) Add any unused modulus, smallest first (highest 1/m first).
            for m in MODULI:
                if m in used:
                    continue
                vm, cnt = compute_valid_mask(m, best)
                if cnt > 0:
                    r = pick_residue(vm, m, best, 2)
                    new_classes = best + [[m, r]]
                    sc = _score(new_classes)
                    if sc > best_score:
                        best = new_classes
                        best_score = sc
                        used.add(m)
                        improved = True
                        break

            # (b) 1-swap: remove one class, greedily re-add up to 2 better ones.
            if not improved:
                for i in range(len(best)):
                    if time.time() > deadline:
                        break
                    m_rem = best[i][0]
                    temp = [best[j] for j in range(len(best)) if j != i]
                    temp_used = set(mm for mm, _ in temp)
                    added = 0
                    for m in MODULI:
                        if m in temp_used or m == m_rem:
                            continue
                        vm, cnt = compute_valid_mask(m, temp)
                        if cnt > 0:
                            r = pick_residue(vm, m, temp, 2)
                            temp = temp + [[m, r]]
                            temp_used.add(m)
                            added += 1
                            if added == 2:
                                break
                    if added > 0:
                        sc = _score(temp)
                        if sc > best_score:
                            best = temp
                            best_score = sc
                            used = set(mm for mm, _ in best)
                            improved = True
                            break

            if not improved:
                break

        return best

    best_classes = None
    best_score = 0.0

    def consider(classes):
        nonlocal best_classes, best_score
        if classes and _is_valid(classes):
            sc = _score(classes)
            if sc > best_score:
                best_score = sc
                best_classes = [list(c) for c in classes]

    # Strategy 1: structured base-modulus packing (multiples of g, distinct
    # residues mod g), then fill in non-multiples.
    for g in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30]:
        if time.time() > deadline - 5:
            break
        multiples = [m for m in MODULI if m % g == 0]
        num_classes = min(g, len(multiples))
        selected = [[m, i] for i, m in enumerate(multiples[:num_classes])]
        if _is_valid(selected):
            for m in MODULI:
                if m % g == 0 or m in set(x[0] for x in selected):
                    continue
                vm, cnt = compute_valid_mask(m, selected)
                if cnt > 0:
                    r = pick_residue(vm, m, selected, 2)
                    selected.append([m, r])
            consider(selected)

    # Strategy 2: greedy with different residue strategies.
    for mode in range(4):
        if time.time() > deadline - 5:
            break
        consider(greedy_construct(mode))

    # Strategy 3: randomized multi-start + local search.
    random.seed(42)
    num_restarts = 40
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        # Weighted random ordering strongly favoring small moduli.
        alpha = 1.0 + 0.5 * random.random()
        weights = [1.0 / (m ** alpha) for m in MODULI]
        order = MODULI[:]
        # Weighted sampling without replacement (Fisher-Yates with weights).
        pool = order[:]
        weighted = []
        for m in pool:
            weighted.append((m, random.random() ** (1.0 / alpha)))
        # Simpler: shuffle then bias by re-sorting partially; use direct weighted
        # selection for correctness.
        chosen = []
        avail = list(MODULI)
        wsum = sum(weights)
        while avail and wsum > 0:
            x = random.random() * wsum
            acc = 0.0
            idx_pick = len(avail) - 1
            for k in range(len(avail)):
                acc += weights[MOD_IDX[avail[k]]]
                if acc >= x:
                    idx_pick = k
                    break
            chosen.append(avail.pop(idx_pick))
            wsum = sum(weights[MOD_IDX[a]] for a in avail)
        selected = []
        for m in chosen:
            if time.time() > deadline - 10:
                break
            vm, cnt = compute_valid_mask(m, selected)
            if cnt > 0:
                r = pick_residue(vm, m, selected, random.choice([0, 1, 2, 3]))
                selected.append([m, r])
        if selected:
            selected = local_search(selected, max_iter=2000)
            consider(selected)

    # Strategy 4: polish the best found so far with an extra local-search pass.
    if best_classes and time.time() < deadline - 2:
        improved = local_search(best_classes, max_iter=6000)
        consider(improved)

    if best_classes is None:
        best_classes = [[20, 0], [40, 1], [60, 2], [80, 3], [100, 4],
                        [120, 5], [140, 6], [160, 7], [180, 8], [200, 9]]

    return best_classes
