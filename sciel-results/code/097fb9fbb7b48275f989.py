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
                    break
        return classes

    def score_of(classes):
        return sum(1.0 / m for m, _ in classes)

    def residue_ok(m, r, classes, skip_idx):
        mi = idx[m]
        for j, (m2, r2) in enumerate(classes):
            if j == skip_idx:
                continue
            g = gcd_table[mi][idx[m2]]
            if r % g == r2 % g:
                return False
        return True

    def local_search(classes, max_iters=6000):
        """Improve via: add missing moduli, reassign single conflicting residue."""
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
            for m in missing:
                mi = idx[m]
                for r in range(m):
                    if residue_ok(m, r, classes, -1):
                        classes.append([m, r])
                        present.add(m)
                        missing.remove(m)
                        score += 1.0 / m
                        improved = True
                        break
                if improved:
                    break

            if improved:
                continue

            # Move B: reassign one existing class's residue to free a slot
            for m in missing:
                mi = idx[m]
                # Find residue r_m with minimum conflicts
                best_rm = None
                best_conf = None
                for r_m in range(m):
                    conf = 0
                    for (m2, r2) in classes:
                        g = gcd_table[mi][idx[m2]]
                        if r_m % g == r2 % g:
                            conf += 1
                    if best_conf is None or conf < best_conf:
                        best_conf = conf
                        best_rm = r_m
                        if conf == 0:
                            break
                if best_conf is None or best_conf > 1:
                    continue

                # Identify the single conflicting class
                conf_js = []
                for j, (m2, r2) in enumerate(classes):
                    g = gcd_table[mi][idx[m2]]
                    if best_rm % g == r2 % g:
                        conf_js.append(j)

                for j in conf_js:
                    mj, rj = classes[j]
                    mj_i = idx[mj]
                    g_jm = gcd_table[mj_i][mi]
                    target = best_rm % g_jm
                    for new_r in range(mj):
                        if new_r == rj:
                            continue
                        if new_r % g_jm == target:
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
                            classes[j] = [mj, new_r]
                            classes.append([m, best_rm])
                            present.add(m)
                            missing.remove(m)
                            score += 1.0 / m
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

            if not improved:
                break

        return classes, score

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

    # --- Multi-start greedy ---
    rng = random.Random(42)
    best_classes = try_greedy(sorted(moduli))
    best_score = score_of(best_classes)

    # Randomized greedy restarts
    for attempt in range(500):
        if time.time() > deadline - 15:
            break
        shuffled = moduli[:]
        rng.shuffle(shuffled)
        # Weight towards smaller moduli
        weighted = sorted(shuffled, key=lambda m: (m * 0.3 + rng.random() * 40))
        cand = try_greedy(weighted)
        s = score_of(cand)
        if s > best_score:
            best_score = s
            best_classes = cand

    # Local search on best
    best_classes, best_score = local_search(best_classes)
    if not is_valid(best_classes):
        best_classes = try_greedy(sorted(moduli))
        best_score = score_of(best_classes)

    # More randomized restarts + local search
    for attempt in range(400):
        if time.time() > deadline - 8:
            break
        shuffled = moduli[:]
        rng.shuffle(shuffled)
        weighted = sorted(shuffled, key=lambda m: (m * 0.5 + rng.random() * 50))
        cand = try_greedy(weighted)
        s = score_of(cand)
        if s > best_score * 0.98:  # only local search promising candidates
            cand, s2 = local_search(cand)
            if s2 > best_score:
                best_score = s2
                best_classes = cand

    # Final pass: try to add any remaining moduli
    present = set(m for m, _ in best_classes)
    missing = [m for m in moduli if m not in present]
    missing.sort()
    for m in missing:
        if time.time() > deadline - 2:
            break
        mi = idx[m]
        for r in range(m):
            if residue_ok(m, r, best_classes, -1):
                best_classes.append([m, r])
                best_score += 1.0 / m
                break

    return best_classes
