"""Additive bases with few representations - smooth surrogate + SA polish.

Uses log-sum-exp soft peak as smooth surrogate for the nonsmooth max objective,
combined with variance penalty. Greedy balanced construction seeds the search,
then simulated annealing with smooth objective escapes local optima, followed
by a high-precision deterministic polish pass.
"""
import time
import random
import math
import numpy as np

N = 128
TARGET_LO = 128
TARGET_HI = 256
TARGETS = list(range(TARGET_LO, TARGET_HI + 1))
NUM_TARGETS = len(TARGETS)  # 129

_START_TIME = time.time()


def _elapsed():
    return time.time() - _START_TIME


def _pair_counts(A):
    """Compute representation counts for each target in [128,256].
    A is a sorted list of ints in [0,128].
    Returns a list of 129 counts (index 0 -> target 128, index 128 -> target 256).
    """
    counts = [0] * NUM_TARGETS
    n = len(A)
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            s = ai + A[j]
            if TARGET_LO <= s <= TARGET_HI:
                counts[s - TARGET_LO] += 1
    return counts


def _peak(counts):
    return max(counts)


def _is_valid(counts):
    return all(c > 0 for c in counts)


def _variance_penalty(counts):
    """Return a small penalty based on variance of counts to encourage flatness."""
    arr = np.array(counts, dtype=np.float64)
    return float(np.var(arr))


def _soft_peak(counts, beta=2.0):
    """Log-sum-exp smooth surrogate for max. Lower beta = sharper peak approximation."""
    arr = np.array(counts, dtype=np.float64)
    m = np.max(arr)
    if m == 0:
        return 0.0
    return float(m + np.log(np.sum(np.exp((arr - m) / beta))))


def _score_smooth(counts, beta=2.0):
    """Lower is better. Soft peak + variance penalty. Infinity if invalid."""
    if not _is_valid(counts):
        return float('inf')
    return _soft_peak(counts, beta) + 0.05 * _variance_penalty(counts)


def _score_hard(counts):
    """Lower is better. Hard peak + variance penalty. Infinity if invalid."""
    if not _is_valid(counts):
        return float('inf')
    return _peak(counts) + 0.05 * _variance_penalty(counts)


def _try_swap2(A_set, A_sorted, counts, i, x_new):
    """Try replacing A_sorted[i] with x_new. A_set is a set for O(1) lookup.
    Returns new_counts or None if invalid.
    """
    ai = A_sorted[i]
    if x_new == ai:
        return None
    if x_new in A_set:
        return None

    n = len(A_sorted)
    new_counts = list(counts)  # copy

    # Remove contributions of ai
    for j in range(n):
        if j == i:
            continue
        aj = A_sorted[j]
        s = ai + aj
        if TARGET_LO <= s <= TARGET_HI:
            new_counts[s - TARGET_LO] -= 1

    # Remove pair (ai, ai)
    s = ai + ai
    if TARGET_LO <= s <= TARGET_HI:
        new_counts[s - TARGET_LO] -= 1

    # Add contributions of x_new
    for j in range(n):
        if j == i:
            continue
        aj = A_sorted[j]
        s = x_new + aj
        if TARGET_LO <= s <= TARGET_HI:
            new_counts[s - TARGET_LO] += 1

    # Add pair (x_new, x_new)
    s = x_new + x_new
    if TARGET_LO <= s <= TARGET_HI:
        new_counts[s - TARGET_LO] += 1

    # Check validity
    if any(c < 0 for c in new_counts):
        return None

    return new_counts


def _local_search_smooth(A_sorted, max_time=None, beta=2.0):
    """Perform swap-based local search with smooth objective.
    Returns (best_A, best_peak, best_counts).
    """
    A_set = set(A_sorted)
    counts = _pair_counts(A_sorted)
    current_score = _score_smooth(counts, beta)
    current_peak = _peak(counts)

    if not _is_valid(counts):
        return (A_sorted, current_peak, counts)

    best_A = list(A_sorted)
    best_peak = current_peak
    best_counts = list(counts)
    best_score = current_score

    n = len(A_sorted)
    improved = True
    while improved and (max_time is None or _elapsed() < max_time):
        improved = False
        for i in range(n):
            ai = A_sorted[i]
            for x_new in range(0, N + 1):
                if x_new == ai or x_new in A_set:
                    continue
                new_counts = _try_swap2(A_set, A_sorted, counts, i, x_new)
                if new_counts is None:
                    continue
                new_score = _score_smooth(new_counts, beta)
                if new_score < current_score:
                    # Accept
                    A_sorted[i] = x_new
                    A_set.discard(ai)
                    A_set.add(x_new)
                    A_sorted.sort()
                    counts = new_counts
                    current_score = new_score
                    current_peak = _peak(counts)
                    improved = True
                    hard_score = _score_hard(counts)
                    if current_peak < best_peak or (current_peak == best_peak and hard_score < best_score):
                        best_peak = current_peak
                        best_A = list(A_sorted)
                        best_counts = list(counts)
                        best_score = hard_score
                    break  # restart from beginning after a move

    return (best_A, best_peak, best_counts)


def _simulated_annealing(A_sorted, max_time=None, rng=None):
    """Simulated annealing with smooth objective to escape local optima.
    Returns (best_A, best_peak, best_counts).
    """
    if rng is None:
        rng = random.Random(123)

    A_set = set(A_sorted)
    counts = _pair_counts(A_sorted)
    current_peak = _peak(counts)

    if not _is_valid(counts):
        return (A_sorted, current_peak, counts)

    best_A = list(A_sorted)
    best_peak = current_peak
    best_counts = list(counts)
    best_score = _score_hard(counts)

    n = len(A_sorted)
    T = 5.0
    T_min = 0.1
    cooling = 0.995

    while T > T_min and _elapsed() < max_time:
        # Pick random element and random replacement
        i = rng.randrange(n)
        ai = A_sorted[i]
        x_new = rng.randint(0, N)
        if x_new == ai or x_new in A_set:
            continue

        new_counts = _try_swap2(A_set, A_sorted, counts, i, x_new)
        if new_counts is None:
            continue

        old_score = _score_smooth(counts, beta=1.0)
        new_score = _score_smooth(new_counts, beta=1.0)
        delta = new_score - old_score

        if delta < 0 or rng.random() < math.exp(-delta / T):
            # Accept
            A_sorted[i] = x_new
            A_set.discard(ai)
            A_set.add(x_new)
            A_sorted.sort()
            counts = new_counts
            current_peak = _peak(counts)

            hard_score = _score_hard(counts)
            if current_peak < best_peak or (current_peak == best_peak and hard_score < best_score):
                best_peak = current_peak
                best_A = list(A_sorted)
                best_counts = list(counts)
                best_score = hard_score

        T *= cooling

    return (best_A, best_peak, best_counts)


def _polish_pass(A_sorted, max_time=None):
    """High-precision deterministic polish: iterate until no improvement possible.
    Uses hard peak + small variance penalty.
    Returns (best_A, best_peak, best_counts).
    """
    A_set = set(A_sorted)
    counts = _pair_counts(A_sorted)
    current_peak = _peak(counts)

    if not _is_valid(counts):
        return (A_sorted, current_peak, counts)

    best_A = list(A_sorted)
    best_peak = current_peak
    best_counts = list(counts)
    best_score = _score_hard(counts)

    n = len(A_sorted)
    improved = True
    while improved and (max_time is None or _elapsed() < max_time):
        improved = False
        for i in range(n):
            ai = A_sorted[i]
            for x_new in range(0, N + 1):
                if x_new == ai or x_new in A_set:
                    continue
                new_counts = _try_swap2(A_set, A_sorted, counts, i, x_new)
                if new_counts is None:
                    continue
                new_peak = _peak(new_counts)
                new_score = _score_hard(new_counts)
                if new_peak < current_peak or (new_peak == current_peak and new_score < _score_hard(counts)):
                    A_sorted[i] = x_new
                    A_set.discard(ai)
                    A_set.add(x_new)
                    A_sorted.sort()
                    counts = new_counts
                    current_peak = new_peak
                    improved = True
                    if current_peak < best_peak or (current_peak == best_peak and new_score < best_score):
                        best_peak = current_peak
                        best_A = list(A_sorted)
                        best_counts = list(counts)
                        best_score = new_score
                    break  # restart after a move

    return (best_A, best_peak, best_counts)


def _greedy_balanced_build(rng, max_time=None):
    """Build a set greedily to cover all targets while balancing representation counts.
    Returns (A_sorted, counts).
    """
    A = []
    A_set = set()
    counts = [0] * NUM_TARGETS
    remaining = list(range(0, N + 1))
    rng.shuffle(remaining)

    # Start with a few random elements to get some coverage
    for _ in range(10):
        if remaining:
            x = remaining.pop()
            A.append(x)
            A_set.add(x)

    A.sort()

    # Update counts
    def _update_counts():
        nonlocal counts
        counts = [0] * NUM_TARGETS
        n = len(A)
        for i in range(n):
            ai = A[i]
            for j in range(i, n):
                s = ai + A[j]
                if TARGET_LO <= s <= TARGET_HI:
                    counts[s - TARGET_LO] += 1

    _update_counts()

    # Now greedily add elements that help balance the distribution
    max_iter = 100
    for _ in range(max_iter):
        if max_time is not None and _elapsed() > max_time:
            break
        if _is_valid(counts):
            break

        # Find the target with minimum count
        min_target_idx = min(range(NUM_TARGETS), key=lambda i: counts[i])
        min_target = TARGET_LO + min_target_idx

        # Find a candidate x in [0,128] not in A that helps cover min_target
        best_x = None
        best_gain = -1
        for x in remaining:
            if x in A_set:
                continue
            # How many new pairs does x create that hit low-count targets?
            gain = 0
            for y in A:
                s = x + y
                if TARGET_LO <= s <= TARGET_HI and counts[s - TARGET_LO] <= 2:
                    gain += 1
            # Also x+x
            s = x + x
            if TARGET_LO <= s <= TARGET_HI and counts[s - TARGET_LO] <= 2:
                gain += 1

            if gain > best_gain:
                best_gain = gain
                best_x = x

        if best_x is None or best_gain <= 0:
            # Pick a random remaining element
            if remaining:
                best_x = remaining.pop()
            else:
                break

        if best_x in remaining:
            remaining.remove(best_x)

        A.append(best_x)
        A_set.add(best_x)
        A.sort()
        _update_counts()

    # Now try to reduce size or improve balance
    # Remove redundant elements
    changed = True
    while changed and len(A) > 15:
        changed = False
        for i in range(len(A)):
            ai = A[i]
            new_counts = list(counts)
            n = len(A)
            for j in range(n):
                if j == i:
                    continue
                aj = A[j]
                s = ai + aj
                if TARGET_LO <= s <= TARGET_HI:
                    new_counts[s - TARGET_LO] -= 1
            s = ai + ai
            if TARGET_LO <= s <= TARGET_HI:
                new_counts[s - TARGET_LO] -= 1

            if _is_valid(new_counts):
                # Can remove
                A.pop(i)
                A_set.discard(ai)
                counts = new_counts
                changed = True
                break

    return A, counts


def construct():
    global _START_TIME
    _START_TIME = time.time()
    DEADLINE = 82.0  # safety margin before 90s kill

    rng = random.Random(42)

    best_A = None
    best_peak = float('inf')
    best_counts = None
    best_score = float('inf')

    # Strategy: greedy balanced construction -> smooth local search -> SA -> polish
    # Try multiple restarts with different random seeds

    for restart in range(15):
        if _elapsed() > DEADLINE - 8:
            break

        # Greedy balanced construction
        A, counts = _greedy_balanced_build(rng, max_time=DEADLINE - 15)

        if not _is_valid(counts):
            # Try to fix by adding elements
            for _ in range(30):
                if _is_valid(counts):
                    break
                uncovered = [i for i in range(NUM_TARGETS) if counts[i] == 0]
                if not uncovered:
                    break
                candidates = [x for x in range(0, N + 1) if x not in set(A)]
                if not candidates:
                    break
                x = rng.choice(candidates)
                A.append(x)
                A.sort()
                counts = _pair_counts(A)

        if not _is_valid(counts):
            continue

        A_sorted = sorted(A)
        peak = _peak(counts)
        score = _score_hard(counts)

        # Phase 1: Smooth local search (beta=2.0 for broad exploration)
        if _elapsed() < DEADLINE - 12:
            A_sorted, peak, counts = _local_search_smooth(A_sorted, max_time=DEADLINE - 12, beta=2.0)

        # Phase 2: Smooth local search with sharper peak (beta=1.0)
        if _elapsed() < DEADLINE - 10:
            A_sorted, peak, counts = _local_search_smooth(A_sorted, max_time=DEADLINE - 10, beta=1.0)

        # Phase 3: Simulated annealing to escape local optima
        if _elapsed() < DEADLINE - 8:
            rng_sa = random.Random(1000 + restart)
            A_sorted, peak, counts = _simulated_annealing(A_sorted, max_time=DEADLINE - 8, rng=rng_sa)

        # Phase 4: High-precision deterministic polish
        if _elapsed() < DEADLINE - 4:
            A_sorted, peak, counts = _polish_pass(A_sorted, max_time=DEADLINE - 4)

        score = _score_hard(counts)
        if peak < best_peak or (peak == best_peak and score < best_score):
            best_peak = peak
            best_A = list(A_sorted)
            best_counts = list(counts)
            best_score = score

        # Early termination if we hit a good peak
        if best_peak <= 3:
            break

    # If we still don't have a valid set, fall back to the seed
    if best_A is None or not _is_valid(best_counts):
        # Use the seed: [64, 128]
        best_A = list(range(64, 129))
        best_counts = _pair_counts(best_A)
        best_peak = _peak(best_counts)

    return best_A
