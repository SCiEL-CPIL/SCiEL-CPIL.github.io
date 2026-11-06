"""Additive bases with few representations - simulated annealing rewrite.

We model the problem as a 0/1 indicator vector x of length 129.
The ordered convolution y = x * x gives the ordered pair-sum multiplicities.
The task asks for y[k] > 0 for all k in [128, 256] and minimizes max y[k].
Note: The task description says "Pair representations are unordered and permit
equal summands". The current program uses ordered convolution.
Let's check the contract again: "Minimize the largest representation multiplicity".
"Local pairs are unordered... paper's ordered convention differs by symmetry".
"""
import time
import random
import numpy as np

_N = 128
_LEN = 129
_T0 = 128
_T1 = 256
_WIN = _T1 - _T0 + 1  # 129 targets
_DEADLINE = 85.0      # safety margin before the 90 s kill

_START = time.time()


def _elapsed():
    return time.time() - _START


def _conv(x):
    """Return the integer convolution of x with itself (length 2*_LEN-1)."""
    return np.convolve(x.astype(np.int64), x.astype(np.int64), mode="full")


def _window(y):
    """Slice y to the target window [128, 256]."""
    return y[_T0:_T1 + 1]


def _valid(y):
    """True iff every target in [128,256] has at least one representation."""
    return bool(np.all(_window(y) > 0))


def _peak(y):
    return int(_window(y).max())


def _make_seed(rng, size_hint=64):
    """Build a valid 0/1 indicator using a 2-level (even/odd) construction."""
    x = np.zeros(_LEN, dtype=np.int64)

    evens = np.array([2 * k for k in range(65)])
    odds = np.array([2 * k + 1 for k in range(64)])

    rng.shuffle(evens)
    rng.shuffle(odds)

    n_even = max(24, size_hint // 2)
    n_odd = max(24, size_hint - n_even)
    n_even = min(n_even, len(evens))
    n_odd = min(n_odd, len(odds))

    x[evens[:n_even]] = 1
    x[odds[:n_odd]] = 1

    y = _conv(x)
    while not _valid(y):
        w = _window(y)
        uncovered = np.where(w == 0)[0] + _T0
        if len(uncovered) == 0:
            break
        best_c = None
        best_gain = -1
        for t in uncovered:
            lo = max(0, t - 128)
            hi = min(128, t)
            for c in range(lo, hi + 1):
                if x[c] == 1:
                    continue
                gain = 0
                if _T0 <= c + c <= _T1 and w[c + c - _T0] == 0:
                    gain += 1
                for yy in np.where(x == 1)[0]:
                    if yy == c:
                        continue
                    s = c + yy
                    if _T0 <= s <= _T1 and w[s - _T0] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
        if best_c is None or best_gain <= 0:
            break
        x[best_c] = 1
        y = _conv(x)

    return x


def _simulated_annealing(x, rng, max_time):
    """Simulated annealing with vectorized cost evaluation."""
    y = _conv(x)
    if not _valid(y):
        return x, _peak(y), y

    best_x = x.copy()
    best_peak = _peak(y)
    best_y = y.copy()

    T = 10.0
    T_min = 0.1
    cooling = 0.9995

    # Initial temperature based on current peak
    current_peak = _peak(y)
    T = max(T, current_peak * 0.5)

    while _elapsed() < max_time:
        # Randomly choose a flip operation
        j = rng.randint(0, _LEN - 1)
        old = x[j]

        # Compute delta y
        if old == 0:
            lo = j
            hi = min(j + _LEN, 2 * _LEN - 1)
            dy = np.zeros(2 * _LEN - 1, dtype=np.int64)
            dy[lo:hi] = x[0:hi - lo]
            if 2 * j < len(dy):
                dy[2 * j] += 1
        else:
            lo = j
            hi = min(j + _LEN, 2 * _LEN - 1)
            dy = np.zeros(2 * _LEN - 1, dtype=np.int64)
            dy[lo:hi] = -x[0:hi - lo]
            if 2 * j < len(dy):
                dy[2 * j] -= 1

        new_y = y + dy

        # Check validity
        if not _valid(new_y):
            # Reject invalid moves unless at very high temperature
            if T > 5.0 and rng.random() < 0.05:
                # Try to accept invalid move temporarily (rare)
                pass
            else:
                # Just reject
                if T > T_min:
                    T *= cooling
                continue

        new_peak = _peak(new_y)
        delta = new_peak - current_peak

        # Acceptance criterion
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-6)):
            x[j] = 1 - old
            y = new_y
            current_peak = new_peak

            if current_peak < best_peak:
                best_peak = current_peak
                best_x = x.copy()
                best_y = y.copy()

        if T > T_min:
            T *= cooling

        # Periodic restart if stuck
        if T < T_min * 2:
            # Reheat
            T = T_min * 5
            # Try to escape by random flips
            for _ in range(5):
                jj = rng.randint(0, _LEN - 1)
                x[jj] = 1 - x[jj]
            y = _conv(x)
            if not _valid(y):
                # Revert to best
                x = best_x.copy()
                y = best_y.copy()
                current_peak = best_peak

    return best_x, best_peak, best_y


def _one_run(rng, size_hint, max_time):
    x = _make_seed(rng, size_hint)
    y = _conv(x)
    if not _valid(y):
        return None
    x, peak, y = _simulated_annealing(x, rng, max_time)
    return x, peak, y


def construct():
    global _START
    _START = time.time()

    rng = random.Random(12345)
    np.random.seed(12345)

    best_x = None
    best_peak = float("inf")

    # Phase 1: several structured seeds of varying sizes
    for size in [48, 56, 64, 72, 80]:
        if _elapsed() > _DEADLINE - 5:
            break
        remaining = _DEADLINE - _elapsed() - 3
        if remaining < 2:
            break
        result = _one_run(rng, size, max_time=remaining)
        if result is not None:
            x, peak, y = result
            if peak < best_peak:
                best_peak = peak
                best_x = x.copy()

        if best_peak <= 4:
            break

    # Phase 2: a few random perturbations of the best seed
    if best_x is not None and _elapsed() < _DEADLINE - 5:
        for _ in range(3):
            if _elapsed() > _DEADLINE - 3:
                break
            x = best_x.copy()
            for _ in range(8):
                j = rng.randint(0, _LEN - 1)
                x[j] = 1 - x[j]
            y = _conv(x)
            if _valid(y):
                remaining = _DEADLINE - _elapsed() - 2
                if remaining < 1:
                    break
                x2, peak2, y2 = _simulated_annealing(x, rng, max_time=remaining)
                if peak2 < best_peak:
                    best_peak = peak2
                    best_x = x2.copy()
            if best_peak <= 3:
                break

    # Fallback: if we somehow have no valid answer, use the full set
    if best_x is None:
        best_x = np.ones(_LEN, dtype=np.int64)
        best_peak = _peak(_conv(best_x))

    # Convert to a sorted list of integers
    result = [int(i) for i in np.where(best_x == 1)[0]]
    return result
