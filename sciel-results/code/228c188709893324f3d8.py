"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _greedy(order):
    """Greedy: process elements in given order, include if 200 not created."""
    reachable = [False] * 201
    reachable[0] = True
    selected = []
    for x in order:
        if 200 - x >= 0 and reachable[200 - x]:
            continue
        selected.append(x)
        for s in range(200 - x, -1, -1):
            if reachable[s]:
                reachable[s + x] = True
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 85

    # Baseline: descending greedy
    best = _greedy(range(100, 0, -1))
    best_len = len(best)

    # Try random orderings
    rng = random.Random(42)
    while time.time() < deadline:
        order = list(range(1, 101))
        rng.shuffle(order)
        result = _greedy(order)
        if len(result) > best_len:
            best = result
            best_len = len(result)

    # Local improvement: try to add back excluded elements
    selected_set = set(best)
    excluded = [x for x in range(1, 101) if x not in selected_set]

    while time.time() < deadline and excluded:
        # Compute reachable sums of current best
        reachable = [False] * 201
        reachable[0] = True
        for y in best:
            for s in range(200 - y, -1, -1):
                if reachable[s]:
                    reachable[s + y] = True

        improved = False
        for x in excluded[:]:
            if x in selected_set:
                continue
            if 200 - x >= 0 and reachable[200 - x]:
                continue
            # Can add x
            best.append(x)
            selected_set.add(x)
            excluded.remove(x)
            improved = True
            break
        if not improved:
            break

    return best
# EVOLVE-BLOCK-END
