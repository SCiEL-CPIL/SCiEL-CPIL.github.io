import time
import random
import numpy as np

N = 17
DEADLINE = 80.0          # safety margin well under the 90s wall-clock kill
_ALL_INPUTS = (1 << N) - 1  # 131071


def _time_left():
    return DEADLINE - (time.perf_counter() - _T0)


def _apply(comps, vec):
    """Apply a list of compare-exchange ops to a vector (in place)."""
    for i, j in comps:
        a = vec[i]
        b = vec[j]
        if a > b:
            vec[i] = b
            vec[j] = a
    return vec


def _verify(comps):
    """Zero-one principle: all-1s must end up as the top bits (131072 inputs)."""
    for k in range(0, N + 1):
        v = np.zeros(N, dtype=np.int8)
        v[:k] = 1
        _apply(comps, v)
        if v[N - k:].sum() != k:
            return False
    return True


def _make_seed():
    """Odd-even transposition network (seed baseline)."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]


def _greedy_reduce(comps):
    """Remove redundant comparators; returns (new_comps, removed_count)."""
    comps = list(comps)
    removed = 0
    for pos in range(len(comps) - 1, -1, -1):
        if _time_left() < 1.0:
            break
        trial = comps[:pos] + comps[pos + 1:]
        if _verify(trial):
            comps = trial
            removed += 1
    return comps, removed


def _random_pair():
    i = random.randrange(N)
    j = random.randrange(N)
    if i == j:
        j = (j + 1) % N
    if i > j:
        i, j = j, i
    return (i, j)


def construct():
    global _T0
    _T0 = time.perf_counter()
    random.seed(12345)

    comps = _make_seed()
    best = list(comps)
    best_len = len(best)

    # First, cheaply shrink the seed via greedy reduction.
    comps, _ = _greedy_reduce(comps)
    if len(comps) < best_len:
        best = list(comps)
        best_len = len(best)

    # Simulated annealing with add / remove / swap kicks.
    T = 1.0
    T_min = 1e-4
    cooling = 0.9995
    last_greedy = 0

    while _time_left() > 2.0:
        r = random.random()
        trial = list(comps)
        if r < 0.40:
            trial.append(_random_pair())
        elif r < 0.75 and len(trial) > 1:
            trial.pop(random.randrange(len(trial)))
        else:
            if len(trial) >= 2:
                a, b = random.sample(range(len(trial)), 2)
                trial[a], trial[b] = trial[b], trial[a]

        ok = _verify(trial)
        d = len(trial) - len(comps)
        if ok:
            if len(trial) < len(comps):
                comps = trial
                if len(comps) < best_len:
                    best = list(comps)
                    best_len = len(comps)
            else:
                # accept neutral / worse with temperature-scaled probability
                if random.random() < np.exp(-max(d, 0) / T):
                    comps = trial
        else:
            # accept a broken move occasionally to escape local minima
            if random.random() < np.exp(-d / T):
                comps = trial

        T = max(T_min, T * cooling)

        # periodic greedy reduction on the current (possibly improved) net
        if time.perf_counter() - last_greedy > 3.0 and _time_left() > 4.0:
            last_greedy = time.perf_counter()
            reduced, _ = _greedy_reduce(comps)
            comps = reduced
            if len(comps) < best_len:
                best = list(comps)
                best_len = len(comps)

    # Final greedy polish on the best network found.
    if _time_left() > 1.0:
        final, _ = _greedy_reduce(best)
        if len(final) < best_len:
            best = final
            best_len = len(best)

    if not _verify(best):
        # defensive fallback to a known-correct seed
        return _make_seed()
    return best


_T0 = 0.0
