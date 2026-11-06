from math import gcd
import time
import random

_START = time.time()
_DEADLINE = 140.0  # safety margin under 150s


def _time_left():
    return _DEADLINE - (time.time() - _START)


def _pair_ok(m1, r1, m2, r2):
    g = gcd(m1, m2)
    return (r1 % g) != (r2 % g)


def _is_valid(classes):
    for i in range(len(classes)):
        m1, r1 = classes[i]
        for j in range(i + 1, len(classes)):
            m2, r2 = classes[j]
            if not _pair_ok(m1, r1, m2, r2):
                return False
    return True


def _score(classes):
    return sum(1.0 / m for m, _ in classes)


def _find_residue(classes, m):
    """Return a residue r for modulus m that is disjoint from all existing, else None."""
    for r in range(m):
        ok = True
        for (mi, ri) in classes:
            if (r % gcd(m, mi)) == (ri % gcd(m, mi)):
                ok = False
                break
        if ok:
            return r
    return None


def _greedy_build(order, classes=None):
    """Build a solution by processing moduli in the given order."""
    if classes is None:
        classes = []
    used = set()
    for m in order:
        if m in used:
            continue
        r = _find_residue(classes, m)
        if r is not None:
            classes.append([m, r])
            used.add(m)
    return classes


def _add_pass(classes):
    """Add every missing modulus that has a valid residue. Returns True if anything added."""
    added = False
    used = set(m for m, _ in classes)
    for m in range(20, 201):
        if m in used:
            continue
        r = _find_residue(classes, m)
        if r is not None:
            classes.append([m, r])
            used.add(m)
            added = True
    return added


def _reassign_pass(classes, rng, max_attempts=2000):
    """Try reassigning one existing residue to enable adding a missing modulus."""
    n = len(classes)
    if n == 0:
        return False
    used = set(m for m, _ in classes)
    missing = [m for m in range(20, 201) if m not in used]
    if not missing:
        return False
    attempts = 0
    for m in missing:
        for i in range(n):
            if attempts >= max_attempts or _time_left() < 10:
                return False
            mi = classes[i][0]
            old = classes[i][1]
            for _ in range(4):
                attempts += 1
                if attempts >= max_attempts or _time_left() < 10:
                    break
                rnew = rng.randrange(mi)
                if rnew == old:
                    continue
                classes[i][1] = rnew
                ok = True
                for j in range(n):
                    if j == i:
                        continue
                    mj, rj = classes[j]
                    if not _pair_ok(mi, rnew, mj, rj):
                        ok = False
                        break
                if not ok:
                    classes[i][1] = old
                    continue
                r = _find_residue(classes, m)
                if r is not None:
                    classes.append([m, r])
                    return True
                classes[i][1] = old
    return False


def _local_search(classes, budget, rng):
    """Iteratively add missing moduli and reassign residues until no progress."""
    t0 = time.time()
    while (time.time() - t0) < budget and _time_left() > 10:
        progress = _add_pass(classes)
        if progress:
            continue
        if _reassign_pass(classes, rng):
            continue
        break
    return classes


def _sa_pass(classes, budget, rng):
    """Simulated annealing with proper cooling: try random reassignments."""
    t0 = time.time()
    n = len(classes)
    best = [list(c) for c in classes]
    best_score = _score(best)
    current = [list(c) for c in classes]
    current_score = _score(current)

    # Precompute gcd matrix for speed
    gcd_mat = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            g = gcd(current[i][0], current[j][0])
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g

    T0 = 0.05
    total_iters = 50000
    iter_count = 0

    while (time.time() - t0) < budget and _time_left() > 10:
        # Adaptive temperature
        progress = iter_count / total_iters
        T = T0 * (1.0 - progress) ** 2 + 1e-6

        # Pick a random class to reassign
        i = rng.randrange(n)
        mi = current[i][0]
        old_r = current[i][1]
        new_r = rng.randrange(mi)
        if new_r == old_r:
            continue

        # Check if new assignment is valid with all others
        ok = True
        for j in range(n):
            if j == i:
                continue
            g = gcd_mat[i][j]
            if (new_r % g) == (current[j][1] % g):
                ok = False
                break

        if not ok:
            continue

        # Try the new assignment
        current[i][1] = new_r

        # Try to add missing moduli
        added = []
        used = set(m for m, _ in current)
        for m in range(20, 201):
            if m in used:
                continue
            r = _find_residue(current, m)
            if r is not None:
                current.append([m, r])
                added.append([m, r])

        new_score = _score(current)
        delta = new_score - current_score

        # Accept if better, or with probability exp(delta/T) if worse
        if delta >= 0 or (T > 1e-6 and rng.random() < min(1.0, delta / T)):
            current_score = new_score
            iter_count += 1
            if new_score > best_score:
                best_score = new_score
                best = [list(c) for c in current]
                # Rebuild gcd matrix if we added elements
                if added:
                    n = len(current)
                    gcd_mat = [[0] * n for _ in range(n)]
                    for ii in range(n):
                        for jj in range(ii + 1, n):
                            g = gcd(current[ii][0], current[jj][0])
                            gcd_mat[ii][jj] = g
                            gcd_mat[jj][ii] = g
        else:
            # Revert
            for a in reversed(added):
                current.pop()
            current[i][1] = old_r

    return best


def _branch_and_bound(classes, budget, rng):
    """
    Branch-and-bound MWIS-style optimization.
    Try to improve by swapping out low-value moduli for combinations of higher-value ones.
    """
    t0 = time.time()
    used = set(m for m, _ in classes)
    missing = [m for m in range(20, 201) if m not in used]

    # Sort missing by 1/m descending (most valuable first)
    missing.sort(key=lambda m: -1.0 / m)

    best_score = _score(classes)
    best_classes = [list(c) for c in classes]

    # Try removing one modulus and adding two or more missing ones
    for i in range(len(classes)):
        if _time_left() < 15 or (time.time() - t0) > budget:
            break

        mi, ri = classes[i]
        # Temporarily remove this modulus
        saved = classes.pop(i)

        # Try to add missing moduli in order of value
        added = []
        for m in missing:
            if m == mi:
                continue
            r = _find_residue(classes, m)
            if r is not None:
                classes.append([m, r])
                added.append([m, r])

        # Check if this improvement is better
        if len(added) >= 2:
            s = _score(classes)
            if s > best_score:
                best_score = s
                best_classes = [list(c) for c in classes]

        # Restore
        for a in reversed(added):
            classes.pop()
        classes.insert(i, saved)

    return best_classes


def _swap_search(classes, budget, rng):
    """
    Systematic swap search: try removing one modulus and adding a different one.
    Uses the full moduli space, sorted by value.
    """
    t0 = time.time()
    all_moduli = list(range(20, 201))
    all_moduli.sort()  # ascending = descending 1/m

    best_score = _score(classes)
    best_classes = [list(c) for c in classes]
    used = set(m for m, _ in classes)

    # Try removing each modulus and adding a better one
    for i in range(len(classes)):
        if _time_left() < 15 or (time.time() - t0) > budget:
            break

        mi, ri = classes[i]
        saved = classes.pop(i)
        used.discard(mi)

        # Try adding moduli in order of value (smallest first)
        improved = False
        for m_new in all_moduli:
            if _time_left() < 15 or (time.time() - t0) > budget:
                break
            if m_new == mi or m_new in used:
                continue
            # Only consider if 1/m_new > 1/mi (i.e., m_new < mi)
            if m_new >= mi:
                continue

            r = _find_residue(classes, m_new)
            if r is not None:
                classes.append([m_new, r])
                used.add(m_new)

                s = _score(classes)
                if s > best_score:
                    best_score = s
                    best_classes = [list(c) for c in classes]
                    improved = True
                    # Keep the improvement and continue searching
                    break
                else:
                    # Revert
                    classes.pop()
                    used.discard(m_new)

        if not improved:
            # Restore original
            classes.append(saved)
            used.add(mi)
        else:
            # The improvement was kept; mi is already removed
            pass

    return best_classes


def _polish_pass(classes, budget, rng):
    """Final polish: try all residues for each modulus, plus 2-swap moves."""
    t0 = time.time()
    best = [list(c) for c in classes]
    best_score = _score(best)
    n = len(best)

    # Process moduli in order of 1/m (most valuable first)
    order = sorted(range(n), key=lambda i: -1.0 / best[i][0])

    for idx in order:
        if _time_left() < 10 or (time.time() - t0) > budget:
            break

        mi = best[idx][0]
        improved = False
        # Try all residues for this modulus
        for r in range(mi):
            if _time_left() < 10 or (time.time() - t0) > budget:
                break
            if r == best[idx][1]:
                continue

            # Check validity with all others
            ok = True
            for j in range(len(best)):
                if j == idx:
                    continue
                mj, rj = best[j]
                if not _pair_ok(mi, r, mj, rj):
                    ok = False
                    break

            if not ok:
                continue

            # Try this assignment
            old_r = best[idx][1]
            best[idx][1] = r

            # Try to add missing moduli
            used = set(m for m, _ in best)
            added = []
            for m in range(20, 201):
                if m in used:
                    continue
                r2 = _find_residue(best, m)
                if r2 is not None:
                    best.append([m, r2])
                    added.append([m, r2])

            new_score = _score(best)

            if new_score > best_score:
                best_score = new_score
                improved = True
                n = len(best)
                # Re-sort remaining order indices
                order = sorted(range(n), key=lambda i: -1.0 / best[i][0])
                # Rebuild order to include new elements
                break  # Restart for this modulus with new state
            else:
                # Revert
                for a in reversed(added):
                    best.pop()
                best[idx][1] = old_r

        # 2-swap: try swapping residues of two moduli
        if not improved and _time_left() > 12 and (time.time() - t0) < budget * 0.8:
            for j2 in range(idx + 1, len(best)):
                if _time_left() < 10 or (time.time() - t0) > budget:
                    break
                mi2 = best[j2][0]
                ri = best[idx][1]
                rj = best[j2][1]
                # Try swapping
                ok = True
                for k in range(len(best)):
                    if k == idx or k == j2:
                        continue
                    mk, rk = best[k]
                    if not _pair_ok(mi, rj, mk, rk) or not _pair_ok(mi2, ri, mk, rk):
                        ok = False
                        break
                if not ok:
                    continue
                # Also check the pair itself
                if not _pair_ok(mi, rj, mi2, ri):
                    continue

                best[idx][1] = rj
                best[j2][1] = ri

                used = set(m for m, _ in best)
                added = []
                for m in range(20, 201):
                    if m in used:
                        continue
                    r2 = _find_residue(best, m)
                    if r2 is not None:
                        best.append([m, r2])
                        added.append([m, r2])

                new_score = _score(best)
                if new_score > best_score:
                    best_score = new_score
                    n = len(best)
                    order = sorted(range(n), key=lambda i: -1.0 / best[i][0])
                    break
                else:
                    for a in reversed(added):
                        best.pop()
                    best[idx][1] = ri
                    best[j2][1] = rj

    return best


def construct():
    rng = random.Random(20240517)
    best = None
    best_score = -1.0

    def consider(classes):
        nonlocal best, best_score
        if classes and _is_valid(classes):
            s = _score(classes)
            if s > best_score:
                best_score = s
                best = [list(c) for c in classes]

    # Strategy 1: Even moduli (current program's approach)
    if _time_left() > 20:
        even_moduli = [m for m in range(20, 201) if m % 2 == 0]
        c1 = _greedy_build(even_moduli)
        c1 = _local_search(c1, min(10.0, max(2.0, _time_left() - 15)), rng)
        consider(c1)

    # Strategy 2: Multiples of 3
    if _time_left() > 20:
        mult3 = [m for m in range(20, 201) if m % 3 == 0]
        c2 = _greedy_build(mult3)
        c2 = _local_search(c2, min(10.0, max(2.0, _time_left() - 15)), rng)
        consider(c2)

    # Strategy 3: All moduli in ascending order
    if _time_left() > 20:
        all_moduli = list(range(20, 201))
        c3 = _greedy_build(all_moduli)
        c3 = _local_search(c3, min(10.0, max(2.0, _time_left() - 15)), rng)
        consider(c3)

    # Strategy 4: Random restarts
    restarts = 0
    while _time_left() > 25 and restarts < 50:
        order = list(range(20, 201))
        rng.shuffle(order)
        c = _greedy_build(order)
        c = _local_search(c, min(5.0, max(1.0, _time_left() - 20)), rng)
        consider(c)
        restarts += 1

    # Strategy 5: Swap search on the best solution (from inspiration 2)
    if best is not None and _time_left() > 30:
        c5 = [list(c) for c in best]
        c5 = _swap_search(c5, min(25.0, max(5.0, _time_left() - 25)), rng)
        consider(c5)

    # Strategy 6: Branch-and-bound improvement on the best solution
    if best is not None and _time_left() > 25:
        c6 = [list(c) for c in best]
        c6 = _branch_and_bound(c6, min(20.0, max(5.0, _time_left() - 20)), rng)
        consider(c6)

    # Strategy 7: Simulated annealing on the best solution
    if best is not None and _time_left() > 25:
        c7 = [list(c) for c in best]
        c7 = _sa_pass(c7, min(25.0, max(5.0, _time_left() - 20)), rng)
        consider(c7)

    # Strategy 8: Final polish pass
    if best is not None and _time_left() > 20:
        c8 = [list(c) for c in best]
        c8 = _polish_pass(c8, min(15.0, max(5.0, _time_left() - 15)), rng)
        consider(c8)

    # Strategy 9: Another SA round after polish
    if best is not None and _time_left() > 20:
        c9 = [list(c) for c in best]
        c9 = _sa_pass(c9, min(15.0, max(3.0, _time_left() - 15)), rng)
        consider(c9)

    # Strategy 10: Final polish
    if best is not None and _time_left() > 15:
        c10 = [list(c) for c in best]
        c10 = _polish_pass(c10, min(10.0, max(3.0, _time_left() - 12)), rng)
        consider(c10)

    # Fallback
    if best is None:
        best = [[20 * j, j - 1] for j in range(1, 11)]

    return best
