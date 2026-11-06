"""Combined program: greedy seed + efficient local search with incremental 3-AP deltas."""
import time
import random

N = 256
DEADLINE = 50.0  # safety margin under 90s kill


def _has_4ap_with(selected, x):
    """Return True if selected ∪ {x} contains a nonconstant 4-term AP involving x."""
    s = selected
    # x as 1st term: x, x+d, x+2d, x+3d
    for d in range(1, (N - x) // 3 + 1):
        if (x + d) in s and (x + 2 * d) in s and (x + 3 * d) in s:
            return True
    # x as 2nd term: x-d, x, x+d, x+2d
    for d in range(1, x):
        if (x - d) in s and (x + d) in s and (x + 2 * d) in s:
            return True
    # x as 3rd term: x-2d, x-d, x, x+d
    for d in range(1, x // 2 + 1):
        if (x - 2 * d) in s and (x - d) in s and (x + d) in s:
            return True
    # x as 4th term: x-3d, x-2d, x-d, x
    for d in range(1, x // 3 + 1):
        if (x - 3 * d) in s and (x - 2 * d) in s and (x - d) in s:
            return True
    return False


def _delta_add(selected, x):
    """Number of new 3-APs created by adding x to selected."""
    s = selected
    cnt = 0
    # x as first term: x, y, z with y = x+d, z = x+2d
    for d in range(1, (N - x) // 2 + 1):
        if (x + d) in s and (x + 2 * d) in s:
            cnt += 1
    # x as middle term: y, x, z with y = x-d, z = x+d
    for d in range(1, min(x, N - x) + 1):
        if (x - d) in s and (x + d) in s:
            cnt += 1
    # x as last term: y, z, x with y = x-2d, z = x-d
    for d in range(1, x // 2 + 1):
        if (x - 2 * d) in s and (x - d) in s:
            cnt += 1
    return cnt


def _delta_remove(selected, x):
    """Number of 3-APs destroyed by removing x from selected."""
    s = selected
    cnt = 0
    for d in range(1, (N - x) // 2 + 1):
        if (x + d) in s and (x + 2 * d) in s:
            cnt += 1
    for d in range(1, min(x, N - x) + 1):
        if (x - d) in s and (x + d) in s:
            cnt += 1
    for d in range(1, x // 2 + 1):
        if (x - 2 * d) in s and (x - d) in s:
            cnt += 1
    return cnt


def _count_3aps(selected):
    """Total number of distinct 3-term APs in selected."""
    s = selected
    cnt = 0
    for x in s:
        cnt += _delta_add(s - {x}, x)
    return cnt


def construct():
    start = time.time()
    random.seed(12345)

    # Greedy seed: add x if it doesn't create a 4-AP
    selected = set()
    for x in range(1, N + 1):
        if not _has_4ap_with(selected, x):
            selected.add(x)

    current_3ap = _count_3aps(selected)
    best_set = set(selected)
    best_3ap = current_3ap

    # Local search: swaps (remove one, add one) to increase 3-AP count
    while time.time() - start < DEADLINE:
        improved = False
        elems = list(selected)
        missing = [x for x in range(1, N + 1) if x not in selected]
        if not elems or not missing:
            break

        # Shuffle to avoid bias
        random.shuffle(elems)
        random.shuffle(missing)

        for rem in elems:
            if time.time() - start > DEADLINE:
                break
            for add in missing:
                if time.time() - start > DEADLINE:
                    break
                # Check 4-AP for the swap: remove rem, add add
                trial = selected - {rem} | {add}
                if _has_4ap_with(trial - {add}, add):
                    continue
                delta = _delta_add(trial - {add}, add) - _delta_remove(selected, rem)
                if delta > 0:
                    selected = trial
                    current_3ap += delta
                    if current_3ap > best_3ap:
                        best_3ap = current_3ap
                        best_set = set(selected)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            # Try single additions
            found = False
            for add in missing:
                if time.time() - start > DEADLINE:
                    break
                if _has_4ap_with(selected, add):
                    continue
                delta = _delta_add(selected, add)
                if delta > 0:
                    selected.add(add)
                    current_3ap += delta
                    if current_3ap > best_3ap:
                        best_3ap = current_3ap
                        best_set = set(selected)
                    found = True
                    break
            if not found:
                # Perturbation: remove a random element to escape local optimum
                if elems:
                    rem = random.choice(elems)
                    selected.discard(rem)
                    current_3ap -= _delta_remove(selected | {rem}, rem)

    return sorted(best_set)
