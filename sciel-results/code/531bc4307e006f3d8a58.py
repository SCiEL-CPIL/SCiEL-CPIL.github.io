import time
import numpy as np

N = 1000
DEADLINE = 110.0  # safety margin under 120s (and the 150s kill)


def _build_greedy(counts, chosen, start):
    """Greedily add the smallest feasible x from `start` onward."""
    chosen = list(chosen)
    for x in range(start, N + 1):
        new_sums = np.array([x + y for y in chosen] + [2 * x], dtype=np.int64)
        trial = counts.copy()
        trial[new_sums] += 1
        if trial[new_sums].max() <= 2:
            chosen.append(x)
            counts = trial
    return chosen, counts


def _try_add(counts, chosen, x):
    """Return (ok, new_counts) if adding x keeps all sums <= 2."""
    new_sums = np.array([x + y for y in chosen] + [2 * x], dtype=np.int64)
    trial = counts.copy()
    trial[new_sums] += 1
    if trial[new_sums].max() <= 2:
        return True, trial
    return False, None


def _local_search(chosen, counts, time_limit):
    """Swap-based local search: try replacing a chosen element with an unused one."""
    chosen = list(chosen)
    counts = counts.copy()
    chosen_set = set(chosen)
    t0 = time.time()
    best_size = len(chosen)
    improved = True
    while improved and (time.time() - t0) < time_limit:
        improved = False
        # candidate pool: unused numbers
        unused = [x for x in range(1, N + 1) if x not in chosen_set]
        # try each swap (remove i, add x)
        for i, a in enumerate(chosen):
            # remove a
            removed_sums = np.array([a + y for y in chosen if y != a] + [2 * a], dtype=np.int64)
            base = counts.copy()
            base[removed_sums] -= 1
            for x in unused:
                # quick feasibility: check sums involving x with (chosen - a)
                others = [y for y in chosen if y != a]
                new_sums = np.array([x + y for y in others] + [2 * x], dtype=np.int64)
                trial = base.copy()
                trial[new_sums] += 1
                if trial[new_sums].max() <= 2:
                    # accept swap
                    counts = trial
                    chosen[i] = x
                    chosen_set.discard(a)
                    chosen_set.add(x)
                    improved = True
                    # continue searching from this new state
                    break
            if (time.time() - t0) > time_limit:
                break
    return chosen, counts


def construct():
    t0 = time.time()
    # Start with an empty set and greedily build
    counts = np.zeros(2 * N + 1, dtype=np.int64)
    chosen, counts = _build_greedy(counts, [], 1)
    best = list(chosen)

    # Try to improve with local search
    if time.time() - t0 < DEADLINE - 5:
        chosen2, counts2 = _local_search(chosen, counts, DEADLINE - 5)
        if len(chosen2) > len(best):
            best = chosen2

    return best
