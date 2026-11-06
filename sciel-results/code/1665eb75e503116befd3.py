"""Construct a large subset of [1,100] with no subset summing to 200."""
# EVOLVE-BLOCK-START
import time
import random

TARGET = 200
MASK = (1 << (TARGET + 1)) - 1  # bits 0..TARGET


def _creates_target(reachable, x):
    """True if adding x makes TARGET reachable (i.e., TARGET-x already reachable)."""
    if x > TARGET:
        return False
    return (reachable >> (TARGET - x)) & 1


def _add(reachable, x):
    return reachable | ((reachable << x) & MASK)


def _is_200_free(selected):
    """Check that no subset of selected sums to TARGET."""
    reachable = 1
    for x in selected:
        if _creates_target(reachable, x):
            return False
        reachable = _add(reachable, x)
    return True


def _reachable_of(selected):
    reachable = 1
    for x in selected:
        reachable = _add(reachable, x)
    return reachable


def _greedy(order):
    """Greedy: take x if it doesn't create TARGET."""
    selected = []
    reachable = 1
    for x in order:
        if not _creates_target(reachable, x):
            selected.append(x)
            reachable = _add(reachable, x)
    return selected, reachable


def _add_local(selected, reachable, all_elements):
    """Repeatedly add any missing element that is safe."""
    selected_set = set(selected)
    changed = True
    while changed:
        changed = False
        for x in all_elements:
            if x not in selected_set and not _creates_target(reachable, x):
                selected.append(x)
                selected_set.add(x)
                reachable = _add(reachable, x)
                changed = True
    return selected, reachable


def _swap_improve(selected, all_elements, deadline):
    """Try removing one element then greedily re-adding; keep if larger."""
    best = list(selected)
    best_size = len(best)
    n = len(selected)
    for i in range(n):
        if time.time() > deadline:
            break
        sel = [x for k, x in enumerate(selected) if k != i]
        reach = _reachable_of(sel)
        sel, reach = _add_local(sel, reach, all_elements)
        if len(sel) > best_size:
            best, best_size = sel, len(sel)
    return best


def _two_swap_improve(selected, all_elements, deadline):
    """Try removing two elements then greedily re-adding; keep if larger."""
    best = list(selected)
    best_size = len(best)
    sel_list = list(selected)
    n = len(sel_list)
    for i in range(n):
        if time.time() > deadline:
            break
        for j in range(i + 1, n):
            if time.time() > deadline:
                break
            sel = [x for k, x in enumerate(sel_list) if k != i and k != j]
            reach = _reachable_of(sel)
            sel, reach = _add_local(sel, reach, all_elements)
            if len(sel) > best_size:
                best, best_size = sel, len(sel)
    return best


def _three_swap_improve(selected, all_elements, deadline):
    """Try removing three elements then greedily re-adding; keep if larger."""
    best = list(selected)
    best_size = len(best)
    sel_list = list(selected)
    n = len(sel_list)
    if n < 3:
        return best
    for i in range(n):
        if time.time() > deadline:
            break
        for j in range(i + 1, n):
            if time.time() > deadline:
                break
            for k in range(j + 1, n):
                if time.time() > deadline:
                    break
                sel = [x for idx, x in enumerate(sel_list) if idx != i and idx != j and idx != k]
                reach = _reachable_of(sel)
                sel, reach = _add_local(sel, reach, all_elements)
                if len(sel) > best_size:
                    best, best_size = sel, len(sel)
    return best


def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin under 90 s wall-clock
    all_elements = list(range(1, 101))

    best = []
    best_size = 0

    def consider(sel):
        nonlocal best, best_size
        if len(sel) > best_size:
            best, best_size = list(sel), len(sel)

    # --- Known good seeds ---
    known = [
        list(range(66, 101)),              # {66..100}
        list(range(66, 98)) + [1, 2, 3],   # {1,2,3} ∪ {66..97}
        [1, 2, 3, 4] + list(range(66, 96)),# {1..4} ∪ {66..95}
    ]
    for seed in known:
        if _is_200_free(seed):
            sel, reach = _add_local(list(seed), _reachable_of(seed), all_elements)
            consider(sel)

    # --- Diverse greedy starts ---
    rng = random.Random(20240517)

    orders = [
        sorted(all_elements, reverse=True),
        sorted(all_elements),
        sorted(all_elements, key=lambda x: x % 7),
        sorted(all_elements, key=lambda x: -x % 13),
        sorted(all_elements, key=lambda x: (x % 5, x)),
    ]
    for _ in range(50):
        o = all_elements[:]
        rng.shuffle(o)
        orders.append(o)

    for o in orders:
        if time.time() > deadline:
            break
        sel, reach = _greedy(o)
        sel, reach = _add_local(sel, reach, all_elements)
        consider(sel)

    # --- Swap-based improvement on the best ---
    if time.time() < deadline - 10.0 and best:
        improved = _swap_improve(best, all_elements, deadline - 10.0)
        consider(improved)

    if time.time() < deadline - 8.0 and len(best) < 100:
        improved2 = _two_swap_improve(best, all_elements, deadline - 8.0)
        consider(improved2)

    if time.time() < deadline - 5.0 and len(best) < 100:
        improved3 = _three_swap_improve(best, all_elements, deadline - 5.0)
        consider(improved3)

    # --- Random restarts: greedy + add-local + occasional swap ---
    iters = 0
    while time.time() < deadline - 5.0:
        o = all_elements[:]
        rng.shuffle(o)
        sel, reach = _greedy(o)
        sel, reach = _add_local(sel, reach, all_elements)
        iters += 1
        if iters % 15 == 0 and time.time() < deadline - 8.0:
            sel = _swap_improve(sel, all_elements, deadline - 8.0)
            if len(sel) < 100 and time.time() < deadline - 5.0:
                sel = _two_swap_improve(sel, all_elements, deadline - 5.0)
            if len(sel) < 100 and time.time() < deadline - 3.0:
                sel = _three_swap_improve(sel, all_elements, deadline - 3.0)
        consider(sel)
        if best_size >= 100:
            break

    # --- Final polish ---
    if time.time() < deadline - 2.0 and best:
        sel, reach = _add_local(list(best), _reachable_of(best), all_elements)
        consider(sel)
        if time.time() < deadline - 5.0:
            sel = _swap_improve(best, all_elements, deadline - 5.0)
            consider(sel)
            if len(sel) < 100 and time.time() < deadline - 3.0:
                sel = _two_swap_improve(best, all_elements, deadline - 3.0)
                consider(sel)
            if len(sel) < 100 and time.time() < deadline - 1.0:
                sel = _three_swap_improve(best, all_elements, deadline - 1.0)
                consider(sel)

    # Safety: ensure nonempty and valid
    if not best or not _is_200_free(best):
        best = list(range(66, 101))

    return best
# EVOLVE-BLOCK-END
