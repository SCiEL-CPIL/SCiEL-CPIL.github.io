import time
import math
import random

def construct():
    deadline = time.time() + 145  # safety margin under 150s

    # Even moduli in [20, 200] -- structured gcds enable better packing
    moduli = list(range(20, 201, 2))  # 100 even moduli
    n = len(moduli)
    idx = {m: i for i, m in enumerate(moduli)}

    # Precompute gcd table (symmetric)
    gcd_table = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            g = math.gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    def try_greedy(order):
        """Build a solution by adding moduli in the given order, first valid residue."""
        classes = []
        present = set()
        for m in order:
            mi = idx.get(m)
            if mi is None or m in present:
                continue
            found = False
            for r in range(m):
                valid = True
                for m2, r2 in classes:
                    g = gcd_table[mi][idx[m2]]
                    if r % g == r2 % g:
                        valid = False
                        break
                if valid:
                    classes.append([m, r])
                    present.add(m)
                    found = True
                    break
            if not found:
                pass
        return classes

    def score_of(classes):
        return sum(1.0 / m for m, _ in classes)

    def is_valid(classes):
        nc = len(classes)
        for i in range(nc):
            m1, r1 = classes[i]
            for j in range(i + 1, nc):
                m2, r2 = classes[j]
                g = math.gcd(m1, m2)
                if r1 % g == r2 % g:
                    return False
        return True

    # Fast validity-check helpers using per-gcd occupied-set tracking.
    # For a candidate class (mi, r), it conflicts with class j iff
    #   (r % gcd(mi, mj)) is already occupied by class j at that gcd.
    def build_occupied(classes):
        occ = {}
        for j, (m, r) in enumerate(classes):
            mi = idx[m]
            for k, (m2, r2) in enumerate(classes):
                if k == j:
                    continue
                g = gcd_table[mi][idx[m2]]
                occ.setdefault(g, set()).add(r % g)
        return occ

    def count_conflicts(mi, r, occ):
        c = 0
        for g, s in occ.items():
            if r % g in s:
                c += 1
        return c

    def local_search(classes, max_iters=60000):
        """Improve via: add missing moduli, reassign single conflicting residue,
        2-opt swap of two residues, random kick. Uses fast occ tracking."""
        classes = [c[:] for c in classes]
        score = score_of(classes)
        present = set(m for m, _ in classes)
        missing = [m for m in moduli if m not in present]
        missing.sort()  # smallest first (highest 1/m)

        rng = random.Random(7)

        for it in range(max_iters):
            if time.time() > deadline - 3:
                break
            improved = False

            # Move A: direct add of smallest missing modulus
            added = False
            if missing:
                occ = build_occupied(classes)
                for m in missing:
                    mi = idx[m]
                    for r in range(m):
                        if count_conflicts(mi, r, occ) == 0:
                            classes.append([m, r])
                            present.add(m)
                            missing.remove(m)
                            score += 1.0 / m
                            added = True
                            break
                    if added:
                        break
            if added:
                continue

            if not missing:
                # Try 2-opt swap and kicks to escape, then break if no improvement
                # (all moduli present -> score fixed; still allow kicks for robustness)
                pass

            # Move B: Reassign one existing class to free a slot for a missing modulus
            if missing:
                m_target = missing[0]
                mi_target = idx[m_target]
                occ = build_occupied(classes)

                best_r_target = None
                min_conf = float('inf')
                for r in range(m_target):
                    conf = count_conflicts(mi_target, r, occ)
                    if conf < min_conf:
                        min_conf = conf
                        best_r_target = r
                        if conf == 0:
                            break

                if min_conf <= 1:
                    # find the conflicting class (at most one)
                    conf_js = []
                    for j, (m2, r2) in enumerate(classes):
                        g = gcd_table[mi_target][idx[m2]]
                        if best_r_target % g == r2 % g:
                            conf_js.append(j)
                    for j in conf_js:
                        mj, rj = classes[j]
                        mj_i = idx[mj]
                        g_jt = gcd_table[mj_i][mi_target]
                        target_bad = best_r_target % g_jt
                        found_move = False
                        for new_r in range(mj):
                            if new_r % g_jt == target_bad:
                                continue
                            ok = True
                            for k, (m2, r2) in enumerate(classes):
                                if k == j:
                                    continue
                                g = gcd_table[mj_i][idx[m2]]
                                if new_r % g == r2 % g:
                                    ok = False
                                    break
                            if ok:
                                classes[j][1] = new_r
                                classes.append([m_target, best_r_target])
                                present.add(m_target)
                                missing.remove(m_target)
                                score += 1.0 / m_target
                                found_move = True
                                break
                        if found_move:
                            improved = True
                            break

            # Move C: 2-opt swap of two existing residues (escape single-move local optima)
            if not improved and len(classes) >= 2 and time.time() > deadline - 8:
                L = len(classes)
                # try a few random pairs
                for _ in range(20):
                    a = rng.randint(0, L - 1)
                    b = rng.randint(0, L - 1)
                    if a == b:
                        continue
                    ma, ra = classes[a]
                    mb, rb = classes[b]
                    ia, ib = idx[ma], idx[mb]
                    gab = gcd_table[ia][ib]
                    # candidate: swap residues
                    ok = True
                    for k, (m2, r2) in enumerate(classes):
                        if k == a or k == b:
                            continue
                        g_a = gcd_table[ia][idx[m2]]
                        g_b = gcd_table[ib][idx[m2]]
                        if rb % g_a == r2 % g_a or ra % g_b == r2 % g_b:
                            ok = False
                            break
                    if ok and (ra % gab != rb % gab):
                        # ensure swapped pair themselves stay disjoint
                        classes[a][1] = rb
                        classes[b][1] = ra
                        improved = True
                        break

            # Move D: Random Kick
            if not improved and classes:
                j = rng.randint(0, len(classes) - 1)
                mj, rj = classes[j]
                mj_i = idx[mj]
                for _ in range(10):
                    new_r = rng.randint(0, mj - 1)
                    if new_r == rj:
                        continue
                    ok = True
                    for k, (m2, r2) in enumerate(classes):
                        if k == j:
                            continue
                        g = gcd_table[mj_i][idx[m2]]
                        if new_r % g == r2 % g:
                            ok = False
                            break
                    if ok:
                        classes[j][1] = new_r
                        break

        return classes, score

    # Initial greedy with smallest-first
    best_classes = try_greedy(sorted(moduli))
    best_score = score_of(best_classes)

    # Local search from the initial greedy
    best_classes, best_score = local_search(best_classes)
    if not is_valid(best_classes):
        best_classes = try_greedy(sorted(moduli))
        best_score = score_of(best_classes)

    # Multi-start randomized greedy with local search
    rng = random.Random(42)
    for attempt in range(4000):
        if time.time() > deadline - 10:
            break
        shuffled = moduli[:]
        rng.shuffle(shuffled)
        weighted = sorted(shuffled, key=lambda m: (m * 0.5 + rng.random() * 60))
        cand = try_greedy(weighted)
        s = score_of(cand)
        if s > best_score:
            cand_classes, cand_score = local_search(cand)
            if is_valid(cand_classes):
                best_classes = cand_classes
                best_score = cand_score

    # Final polish pass on the best solution
    best_classes, best_score = local_search(best_classes)
    if not is_valid(best_classes):
        # fallback to last valid
        pass

    return best_classes
