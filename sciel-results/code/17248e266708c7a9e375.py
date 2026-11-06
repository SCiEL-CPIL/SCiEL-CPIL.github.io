"""Additive bases with few representations - Simulated Annealing approach.

Uses Simulated Annealing to minimize the peak representation count, with variance
as a secondary objective. Includes fast incremental swap evaluation and coverage
fixing.
"""
import time
import random
import math
import numpy as np

N = 128
TARGET_LO = 128
TARGET_HI = 256
NUM_TARGETS = 129  # targets 128..256 inclusive

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


def _score_hard(counts):
    """Hard objective: peak + small variance penalty."""
    if not _is_valid(counts):
        return float('inf')
    arr = np.array(counts, dtype=np.float64)
    return _peak(counts) + 0.01 * float(np.var(arr))


def _try_swap(A_sorted, A_set, counts, i, x_new):
    """Try replacing A_sorted[i] with x_new.
    Returns (new_counts, new_peak, new_var) or None if invalid.
    """
    ai = A_sorted[i]
    if x_new == ai:
        return None
    if x_new in A_set:
        return None

    n = len(A_sorted)
    new_counts = list(counts)

    # Remove contributions of ai
    s = ai + ai
    if TARGET_LO <= s <= TARGET_HI:
        new_counts[s - TARGET_LO] -= 1
    for j in range(n):
        if j == i:
            continue
        aj = A_sorted[j]
        s = ai + aj
        if TARGET_LO <= s <= TARGET_HI:
            new_counts[s - TARGET_LO] -= 1

    # Add contributions of x_new
    s = x_new + x_new
    if TARGET_LO <= s <= TARGET_HI:
        new_counts[s - TARGET_LO] += 1
    for j in range(n):
        if j == i:
            continue
        aj = A_sorted[j]
        s = x_new + aj
        if TARGET_LO <= s <= TARGET_HI:
            new_counts[s - TARGET_LO] += 1

    if any(c <= 0 for c in new_counts):
        return None

    new_peak = max(new_counts)
    new_var = float(np.var(np.array(new_counts, dtype=np.float64)))
    return (new_counts, new_peak, new_var)


def _simulated_annealing(A_sorted, max_time):
    """Simulated Annealing to minimize peak and variance."""
    A_sorted = sorted(A_sorted)
    A_set = set(A_sorted)
    counts = _pair_counts(A_sorted)

    if not _is_valid(counts):
        return (A_sorted, float('inf'), counts)

    current_peak = _peak(counts)
    current_var = float(np.var(np.array(counts, dtype=np.float64)))
    current_score = current_peak + 0.01 * current_var
    
    best_A = list(A_sorted)
    best_peak = current_peak
    best_counts = list(counts)
    best_score = current_score

    n = len(A_sorted)
    deadline = _elapsed() + max_time
    
    # SA Parameters
    T_start = 1.0
    T_end = 0.001
    cooling_rate = 0.995
    T = T_start
    
    # Precompute candidates for each index? No, depends on A_set.
    
    while _elapsed() < deadline:
        # Pick a random index
        i = random.randint(0, n - 1)
        ai = A_sorted[i]
        
        # Pick a random candidate not in A_set
        # To be faster, pick from a small random subset of [0, 128]
        # Or just pick a random x and check if in A_set
        x_new = random.randint(0, N)
        if x_new == ai or x_new in A_set:
            continue
            
        result = _try_swap(A_sorted, A_set, counts, i, x_new)
        if result is None:
            continue
            
        new_counts, new_peak, new_var = result
        new_score = new_peak + 0.01 * new_var
        
        delta = new_score - current_score
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            # Accept move
            A_sorted[i] = x_new
            A_set.discard(ai)
            A_set.add(x_new)
            A_sorted.sort()
            counts = new_counts
            current_peak = new_peak
            current_var = new_var
            current_score = new_score
            
            if new_score < best_score:
                best_score = new_score
                best_peak = new_peak
                best_A = list(A_sorted)
                best_counts = list(counts)
                
        T *= cooling_rate
        if T < T_end:
            T = T_end

    return (best_A, best_peak, best_counts)


def _structured_seed(seed_type, rng):
    """Generate a structured initial set based on seed_type."""
    A = set()
    if seed_type == 0:
        # Arithmetic progression: multiples of k
        k = rng.choice([4, 5, 6, 7, 8])
        for x in range(0, N + 1, k):
            A.add(x)
        # Add some random fillers
        for _ in range(rng.randint(5, 15)):
            A.add(rng.randint(0, N))
    elif seed_type == 1:
        # Symmetric around 64
        for _ in range(rng.randint(20, 40)):
            x = rng.randint(0, N)
            A.add(x)
            A.add(N - x)  # symmetry
    elif seed_type == 2:
        # Modular construction: residues mod m
        m = rng.choice([8, 10, 12, 14])
        for r in range(0, m):
            if rng.random() < 0.7:
                # Add all numbers congruent to r mod m in range
                for x in range(r, N + 1, m):
                    if rng.random() < 0.6:
                        A.add(x)
        # Ensure minimum size
        while len(A) < 20:
            A.add(rng.randint(0, N))
    elif seed_type == 3:
        # Dense in middle, sparse at edges
        for x in range(32, 97):
            if rng.random() < 0.5:
                A.add(x)
        for x in range(0, 32):
            if rng.random() < 0.2:
                A.add(x)
        for x in range(97, N + 1):
            if rng.random() < 0.2:
                A.add(x)
        while len(A) < 20:
            A.add(rng.randint(0, N))
    else:
        # Pure random
        size = rng.randint(25, 45)
        A = set(rng.sample(range(0, N + 1), size))

    return sorted(A)


def _fix_coverage(A, rng, max_adds=30):
    """Add elements to cover any uncovered targets."""
    A = sorted(A)
    counts = _pair_counts(A)
    for _ in range(max_adds):
        if _is_valid(counts):
            break
        uncovered = [i for i in range(NUM_TARGETS) if counts[i] == 0]
        if not uncovered:
            break

        A_set = set(A)
        best_x = None
        best_gain = -1
        for x in range(0, N + 1):
            if x in A_set:
                continue
            gain = 0
            for y in A:
                s = x + y
                if TARGET_LO <= s <= TARGET_HI and counts[s - TARGET_LO] == 0:
                    gain += 1
            s = x + x
            if TARGET_LO <= s <= TARGET_HI and counts[s - TARGET_LO] == 0:
                gain += 1
            if gain > best_gain:
                best_gain = gain
                best_x = x

        if best_x is None or best_gain <= 0:
            candidates = [x for x in range(0, N + 1) if x not in A_set]
            if not candidates:
                break
            best_x = rng.choice(candidates)

        A.append(best_x)
        A.sort()
        counts = _pair_counts(A)

    return A, counts


def _reduce_size(A, max_time=10.0):
    """Try to remove redundant elements while maintaining coverage."""
    A = sorted(A)
    A_set = set(A)
    counts = _pair_counts(A)

    if not _is_valid(counts):
        return A, counts

    deadline = _elapsed() + max_time
    changed = True
    while changed and _elapsed() < deadline and len(A) > 18:
        changed = False
        for i in range(len(A)):
            if _elapsed() > deadline:
                break
            ai = A[i]
            n = len(A)
            new_counts = list(counts)

            s = ai + ai
            if TARGET_LO <= s <= TARGET_HI:
                new_counts[s - TARGET_LO] -= 1
            for j in range(n):
                if j == i:
                    continue
                aj = A[j]
                s = ai + aj
                if TARGET_LO <= s <= TARGET_HI:
                    new_counts[s - TARGET_LO] -= 1

            if all(c > 0 for c in new_counts):
                A.pop(i)
                A_set.discard(ai)
                counts = new_counts
                changed = True
                break

    return A, counts


def construct():
    global _START_TIME
    _START_TIME = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    rng = random.Random(42)

    best_A = None
    best_peak = float('inf')
    best_counts = None
    best_score = float('inf')

    # Try multiple structured seeds with different strategies
    seed_types = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]

    for idx, st in enumerate(seed_types):
        if _elapsed() > DEADLINE - 10:
            break

        # Generate structured seed
        A = _structured_seed(st, rng)

        # Fix coverage
        A, counts = _fix_coverage(A, rng)

        if not _is_valid(counts):
            continue

        # Reduce size
        A, counts = _reduce_size(A, max_time=3.0)

        if not _is_valid(counts):
            continue

        peak = _peak(counts)
        score = _score_hard(counts)

        # Simulated Annealing
        remaining_time = DEADLINE - _elapsed() - 5
        if remaining_time < 2:
            break

        A_sorted = sorted(A)
        A_sorted, peak, counts = _simulated_annealing(A_sorted, max_time=remaining_time)

        if peak < best_peak or (peak == best_peak and _score_hard(counts) < best_score):
            best_peak = peak
            best_A = list(A_sorted)
            best_counts = list(counts)
            best_score = _score_hard(counts)

        # Early termination for good solutions
        if best_peak <= 3:
            break

    # Final fallback
    if best_A is None or not _is_valid(best_counts):
        # Use a known valid construction
        best_A = list(range(64, 129))
        best_counts = _pair_counts(best_A)
        best_peak = _peak(best_counts)

    return best_A
