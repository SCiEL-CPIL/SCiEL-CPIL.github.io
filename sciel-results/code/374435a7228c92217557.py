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
        for m in order:
            mi = idx.get(m)
            if mi is None:
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

    def is_valid(classes):
        n = len(classes)
        for i in range(n):
            m1, r1 = classes[i]
            for j in range(i + 1, n):
                m2, r2 = classes[j]
                g = math.gcd(m1, m2)
                if r1 % g == r2 % g:
                    return False
        return True

    def local_search(classes, max_iters=6000):
        classes = [c[:] for c in classes]
        score = score_of(classes)
        present = set(m for m, _ in classes)
        missing = [m for m in moduli if m not in present]
        missing.sort()  # smallest first (highest 1/m)
        
        # Track consecutive non-improving iterations for kick decision
        stagnant = 0
        max_stagnant = 10  # Kick after 10 stagnant iterations

        for it in range(max_iters):
            if time.time() > deadline - 3:
                break
            improved = False

            # Move A: try to add a missing modulus directly
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
                stagnant = 0
                continue

            # Move B: reassign one existing class's residue to enable adding a missing modulus
            for m in missing:
                mi = idx[m]
                # Find residue with minimum conflicts
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

                # Identify conflicting classes
                conf_js = []
                for j, (m2, r2) in enumerate(classes):
                    g = gcd_table[mi][idx[m2]]
                    if best_rm % g == r2 % g:
                        conf_js.append(j)

                # Try to reassign each conflicting class
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

            if improved:
                stagnant = 0
                continue

            # No improvement: increment stagnant counter
            stagnant += 1
            
            # Move C: Random Kick - perturb a random existing class to escape local minima
            if stagnant >= max_stagnant and classes:
                stagnant = 0
                # Pick a random class to perturb
                kick_idx = random.randint(0, len(classes) - 1)
                mj, rj = classes[kick_idx]
                mj_i = idx[mj]
                
                # Try to find a new residue that doesn't conflict with others
                # We allow temporary conflict with the kicked class's old residue
                # by just picking a random new residue and checking against all others
                for _ in range(50):  # Try up to 50 random residues
                    new_r = random.randint(0, mj - 1)
                    if new_r == rj:
                        continue
                    ok = True
                    for k, (m2, r2) in enumerate(classes):
                        if k == kick_idx:
                            continue
                        g = gcd_table[mj_i][idx[m2]]
                        if new_r % g == r2 % g:
                            ok = False
                            break
                    if ok:
                        classes[kick_idx] = [mj, new_r]
                        break
                # After kick, we don't count this as "improved" score-wise,
                # but it changes the state to allow further moves.
                # We force the loop to continue to try moves A and B again.
                continue

            if not improved:
                # If we've been stagnant for too long and no kick helped, break
                # But the kick logic above handles the 'stagnant' case.
                # If we reach here, it means improved is False and we didn't kick?
                # Actually, if stagnant < max_stagnant, we just continue.
                # If stagnant >= max_stagnant, we kicked and continued.
                # So we only break if we are truly stuck and time is up?
                # The original code broke immediately on non-improvement.
                # We want to allow the kick mechanism to work.
                # So we should NOT break immediately. We just continue the loop.
                # However, if we have no missing moduli, we are done.
                if not missing:
                    break

        return classes, score

    # Start with smallest-first greedy
    best_classes = try_greedy(sorted(moduli))
    best_score = score_of(best_classes)
    best_classes, best_score = local_search(best_classes)

    # Multiple randomized restarts with local search
    rng = random.Random(42)
    attempts = 0
    while time.time() < deadline - 10:
        attempts += 1
        if attempts > 2000:  # Hard cap on attempts to prevent infinite loop if time check fails
            break
        shuffled = moduli[:]
        rng.shuffle(shuffled)
        # Weight towards smaller moduli
        weighted = sorted(shuffled, key=lambda m: (m * 0.3 + rng.random() * 80))
        cand = try_greedy(weighted)
        s = score_of(cand)
        if s > best_score:
            best_classes, best_score = local_search(cand)
        elif s >= best_score - 0.05:
            # Close enough, try local search anyway
            cand2, s2 = local_search(cand)
            if s2 > best_score:
                best_classes, best_score = cand2, s2

    # Final validation
    if not is_valid(best_classes):
        best_classes = try_greedy(sorted(moduli))
        best_score = score_of(best_classes)

    return best_classes
