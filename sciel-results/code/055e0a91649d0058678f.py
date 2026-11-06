"""Fresh design: greedy superincreasing baseline + deadline-guarded local search."""
import time
import random

_N = 12


def _subset_sums(arr):
    """Return list of all 2**n subset sums for arr (length n)."""
    s = [0]
    for x in arr:
        s += [v + x for v in s]
    return s


def _valid(arr):
    """All subset sums distinct and all elements positive distinct ints."""
    if len(arr) != _N:
        return False
    if any(x <= 0 for x in arr):
        return False
    if len(set(arr)) != _N:
        return False
    return len(set(_subset_sums(arr))) == (1 << _N)


def _greedy_superincreasing():
    """Smallest superincreasing sequence: guarantees distinct subset sums."""
    a = []
    s = 0
    for _ in range(_N):
        x = s + 1  # strictly greater than sum of all previous -> distinct sums
        a.append(x)
        s += x
    return a


def _local_search(a, deadline):
    """Try to reduce the largest element while keeping all subset sums distinct."""
    best = max(a)
    cur = list(a)
    rng = random.Random(12345)

    while time.time() < deadline:
        # Find the largest element
        idx = max(range(_N), key=lambda i: cur[i])
        cur_max = cur[idx]
        others = [cur[i] for i in range(_N) if i != idx]
        others_set = set(others)
        # Lower bound: must stay > 0, distinct, and not equal to any other element
        # Try to lower cur_max to the smallest possible value that keeps validity.
        # Candidate values: scan downward from cur_max-1, skip duplicates.
        # To be efficient, test a handful of promising candidates.
        improved = False
        # Candidate pool: values just below current max and some random lower values
        cands = set()
        for d in range(1, min(60, cur_max)):
            c = cur_max - d
            if c > 0 and c not in others_set:
                cands.add(c)
        # Also try a few random lower values
        for _ in range(20):
            c = rng.randint(1, max(1, cur_max - 1))
            if c not in others_set:
                cands.add(c)
        # Sort ascending to find the smallest valid one (minimizes max)
        for c in sorted(cands):
            trial = list(cur)
            trial[idx] = c
            if _valid(trial):
                cur = trial
                best = max(cur)
                improved = True
                break
        if not improved:
            # Perturb: try swapping in a random element to escape local optimum
            if time.time() < deadline:
                i = rng.randrange(_N)
                j = rng.randrange(_N)
                if i != j:
                    cur[i], cur[j] = cur[j], cur[i]
                    if _valid(cur):
                        best = min(best, max(cur))
                    else:
                        cur[i], cur[j] = cur[j], cur[i]  # revert
    return cur, best


def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin under the 90s kill

    # Baseline: greedy superincreasing (always valid)
    base = _greedy_superincreasing()
    best = max(base)
    best_arr = list(base)

    # Local search to try to beat the baseline
    if time.time() < deadline:
        cur, b = _local_search(list(base), deadline)
        if _valid(cur) and max(cur) < best:
            best = max(cur)
            best_arr = list(cur)

    # Final validity guarantee
    if not _valid(best_arr):
        best_arr = list(base)
        best = max(base)

    return best_arr


if __name__ == "__main__":
    print(construct())
