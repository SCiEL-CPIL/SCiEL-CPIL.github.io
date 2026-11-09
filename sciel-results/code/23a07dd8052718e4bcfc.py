import time
import random
import numpy as np

BASE = 51975
DEADLINE = 115.0  # safety margin under 120s wall-clock

# All divisors of 51975 = 3^3 * 5^2 * 7 * 11
_DIVS = [
    3, 5, 7, 9, 11, 15, 21, 25, 27, 33, 35, 45, 55, 63, 75, 77,
    99, 105, 135, 165, 175, 225, 231, 275, 315, 385, 495, 525, 675,
    7425, 945, 1155, 1375, 1575, 1847, 2205, 2775, 3675, 4455,
    5547, 7735, 8253, 9435, 11095, 14823, 16539, 25385, 51975
]
# (already sorted ascending)

def _score(classes):
    """Number of residues modulo BASE covered by no class (lower is better)."""
    count = np.zeros(BASE, dtype=np.int8)
    for m, r in classes:
        count[r::m] += 1
    return int(np.count_nonzero(count == 0))

def _rarest_greedy():
    """Seed: for each modulus (any order), pick residue covering the rarest uncovered residue."""
    count = np.zeros(BASE, dtype=np.int8)
    classes = []
    for m in _DIVS:
        # find the residue r that maximizes coverage of currently-uncovered residues
        best_r = 0
        best_gain = -1
        for r in range(m):
            seg = count[r::m]
            gain = int(np.count_nonzero(seg == 0))
            if gain > best_gain:
                best_gain = gain
                best_r = r
        classes.append((m, best_r))
        count[best_r::m] += 1
    return classes

def _cd(classes, time_budget=10.0):
    """Coordinate descent: repeatedly try changing one modulus's residue to the best."""
    classes = [list(c) for c in classes]
    mlist = [c[0] for c in classes]
    rlist = np.array([c[1] for c in classes], dtype=np.int64)
    n = len(classes)

    count = np.zeros(BASE, dtype=np.int8)
    for i in range(n):
        count[rlist[i]::mlist[i]] += 1
    uncovered = int(np.count_nonzero(count == 0))
    best_uncovered = uncovered

    t0 = time.time()
    improved = True
    while improved and (time.time() - t0) < time_budget:
        improved = False
        for i in range(n):
            m = mlist[i]
            old_r = int(rlist[i])
            count[old_r::m] -= 1
            base_uncovered = int(np.count_nonzero(count == 0))
            best_r = old_r
            best_u = base_uncovered
            # try every residue
            for r in range(m):
                count[r::m] += 1
                u = int(np.count_nonzero(count == 0))
                count[r::m] -= 1
                if u < best_u:
                    best_u = u
                    best_r = r
            if best_r != old_r:
                count[best_r::m] += 1
                rlist[i] = best_r
                uncovered = best_u
                if uncovered < best_uncovered:
                    best_uncovered = uncovered
                improved = True
            else:
                count[old_r::m] += 1
    return classes, best_uncovered

def construct():
    t_start = time.time()
    rng = random.Random(12345)

    # --- Seed: rarest-first greedy over all 47 moduli ---
    classes = _rarest_greedy()
    classes, best_u = _cd(classes, time_budget=8.0)
    best_classes = [list(c) for c in classes]

    # --- Local search with add/remove moves + coordinate descent ---
    used = set(m for m, _ in best_classes)
    all_divs = set(_DIVS)

    def try_insert(classes, used_set):
        # try adding each unused modulus at its best residue
        best_gain = 0
        best_m = None
        best_r = None
        count = np.zeros(BASE, dtype=np.int8)
        for m, r in classes:
            count[r::m] += 1
        base_u = int(np.count_nonzero(count == 0))
        for m in all_divs - used_set:
            best_r_m = 0
            best_u_m = base_u
            for r in range(m):
                count[r::m] += 1
                u = int(np.count_nonzero(count == 0))
                count[r::m] -= 1
                if u < best_u_m:
                    best_u_m = u
                    best_r_m = r
            if base_u - best_u_m > best_gain:
                best_gain = base_u - best_u_m
                best_m = m
                best_r = best_r_m
        if best_m is not None:
            classes.append([best_m, best_r])
            used_set.add(best_m)
        return classes, used_set

    def try_remove(classes, used_set):
        # try removing each modulus if it reduces or doesn't increase uncovered
        best_gain = 0
        best_idx = None
        count = np.zeros(BASE, dtype=np.int8)
        for m, r in classes:
            count[r::m] += 1
        base_u = int(np.count_nonzero(count == 0))
        for i, (m, r) in enumerate(classes):
            count[r::m] -= 1
            u = int(np.count_nonzero(count == 0))
            count[r::m] += 1
            gain = u - base_u  # removing reduces uncovered by 'gain'
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_idx is not None:
            m, r = classes[best_idx]
            used_set.discard(m)
            classes.pop(best_idx)
        return classes, used_set

    # Iterate add/remove + CD until no improvement or deadline
    while (time.time() - t_start) < DEADLINE - 5.0:
        # Add a modulus if helpful
        classes2, used2 = try_insert(best_classes, set(m for m, _ in best_classes))
        if len(classes2) > len(best_classes):
            classes2, u2 = _cd(classes2, time_budget=4.0)
            if u2 < best_u:
                best_classes = classes2
                best_u = u2
                continue
        # Remove a modulus if helpful
        classes3, used3 = try_remove(best_classes, set(m for m, _ in best_classes))
        if len(classes3) < len(best_classes):
            classes3, u3 = _cd(classes3, time_budget=4.0)
            if u3 < best_u:
                best_classes = classes3
                best_u = u3
                continue
        # Pure CD polish
        classes4, u4 = _cd(best_classes, time_budget=4.0)
        if u4 < best_u:
            best_classes = classes4
            best_u = u4
            continue
        # No improvement: try a random restart of CD from a perturbed state
        if (time.time() - t_start) > DEADLINE - 10.0:
            break
        # Perturb: randomly change a few residues
        perturbed = [list(c) for c in best_classes]
        for _ in range(3):
            i = rng.randrange(len(perturbed))
            m = perturbed[i][0]
            perturbed[i][1] = rng.randrange(m)
        perturbed, u_p = _cd(perturbed, time_budget=4.0)
        if u_p < best_u:
            best_classes = perturbed
            best_u = u_p
            continue
        # If still no progress, stop
        break

    # Final exact score check and return
    final_score = _score(best_classes)
    return [[m, r] for m, r in best_classes]
