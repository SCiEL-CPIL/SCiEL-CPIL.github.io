import time
import math
import random

def construct():
    deadline = time.time() + 110  # safety margin under 150s

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

    def local_search(classes, max_iters=50000, rng=None):
        """Improve via: add missing moduli, reassign single conflicting residue, random kick."""
        if rng is None:
            rng = random.Random()
        classes = [c[:] for c in classes]
        score = score_of(classes)
        present = set(m for m, _ in classes)
        missing = [m for m in moduli if m not in present]
        missing.sort()  # smallest first (highest 1/m)

        for _ in range(max_iters):
            if time.time() > deadline - 3:
                break
            improved = False

            # Move A: direct add of smallest missing modulus
            added = False
            for m in missing:
                mi = idx[m]
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
                        missing.remove(m)
                        score += 1.0 / m
                        added = True
                        break
                if added:
                    break

            if added:
                continue

            # Move B: Reassign one existing class to free a slot for a missing modulus
            if not missing:
                break

            m_target = missing[0]
            mi_target = idx[m_target]

            best_r_target = None
            min_conf = float('inf')

            for r in range(m_target):
                conf = 0
                for m2, r2 in classes:
                    g = gcd_table[mi_target][idx[m2]]
                    if r % g == r2 % g:
                        conf += 1
                if conf < min_conf:
                    min_conf = conf
                    best_r_target = r
                    if conf == 0:
                        break

            if min_conf <= 1:
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

            if improved:
                continue

            # Move C: Random Kick
            if classes:
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
    rng0 = random.Random(123)
    best_classes, best_score = local_search(best_classes, max_iters=30000, rng=rng0)
    if not is_valid(best_classes):
        best_classes = try_greedy(sorted(moduli))
        best_score = score_of(best_classes)

    # Multi-start randomized greedy with local search
    rng = random.Random(42)
    for attempt in range(500):
        if time.time() > deadline - 8:
            break
        shuffled = moduli[:]
        rng.shuffle(shuffled)
        # Weighted order: bias toward small moduli but with randomness
        weighted = sorted(shuffled, key=lambda m: (m * 0.5 + rng.random() * 60))
        cand = try_greedy(weighted)
        s = score_of(cand)
        if s > best_score - 0.05:  # only polish if promising
            cand_rng = random.Random(attempt)
            cand_classes, cand_score = local_search(cand, max_iters=15000, rng=cand_rng)
            if is_valid(cand_classes) and cand_score > best_score:
                best_classes = cand_classes
                best_score = cand_score

    # Final polish pass on the best solution with more iterations
    if time.time() < deadline - 5:
        final_rng = random.Random(999)
        best_classes, best_score = local_search(best_classes, max_iters=50000, rng=final_rng)
        if not is_valid(best_classes):
            # Fallback: revert to a valid state
            best_classes = try_greedy(sorted(moduli))
            best_score = score_of(best_classes)

    return best_classes
