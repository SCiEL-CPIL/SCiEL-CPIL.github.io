"""Size-efficient sorting network on 17 wires: fast LUT verify + SA search."""
import time
import random
import numpy as np

N = 17
TOTAL = 1 << N  # 131072

# Precompute, for every Boolean input (as an integer 0..TOTAL-1) and every wire,
# the bit value of that wire. Then a comparator (i,j) transforms an input x to:
#   x ^ swap_mask[i][j][x]
# where swap_mask[i][j][x] is the bitmask that flips wires i and j if they are
# out of order (bit i = 1, bit j = 0).
_WIRE_BITS = np.empty((N, TOTAL), dtype=np.int8)
for _w in range(N):
    _WIRE_BITS[_w] = (np.arange(TOTAL, dtype=np.int32) >> _w) & 1
_WIRE_BITS = _WIRE_BITS.T  # shape (TOTAL, N): _WIRE_BITS[x, w] = bit w of x

_SWAP = np.empty((N, N, TOTAL), dtype=np.int32)
for _i in range(N):
    for _j in range(N):
        if _i < _j:
            _m = (_WIRE_BITS[:, _i] == 1) & (_WIRE_BITS[:, _j] == 0)
            _mask = np.zeros(TOTAL, dtype=np.int32)
            _mask[_m] = (1 << _i) | (1 << _j)
            _SWAP[_i, _j] = _mask
        else:
            _SWAP[_i, _j] = 0

_SORTED = np.arange(TOTAL, dtype=np.int32)


def _build_seed():
    """Odd-even transposition network (136 comparators)."""
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]


def _verify(comp):
    """Zero-one principle over all 2^N Boolean inputs. Returns True iff sorted."""
    if not comp:
        return True
    state = _SORTED.copy()
    for i, j in comp:
        state ^= _SWAP[i, j]
    return bool(np.array_equal(state, _SORTED))


def _prune(comp, deadline):
    """Repeatedly remove redundant comparators (scan from the end)."""
    comp = list(comp)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for idx in range(len(comp) - 1, -1, -1):
            if time.time() > deadline:
                return comp
            cand = comp[:idx] + comp[idx + 1:]
            if _verify(cand):
                comp = cand
                changed = True
                break
    return comp


def _sa_search(comp, deadline, rnd):
    """Simulated-annealing search over the comparator sequence."""
    comp = list(comp)
    n = len(comp)
    all_pairs = [(i, j) for j in range(N) for i in range(j)]
    t = 3.0
    cooling = 0.999
    best = list(comp)
    best_n = n
    start = time.time()

    while time.time() < deadline:
        # pick a move
        r = rnd.random()
        if r < 0.45 and n > 0:
            # remove
            idx = rnd.randint(0, n - 1)
            cand = comp[:idx] + comp[idx + 1:]
            cost = n - len(cand)  # -1 (improvement)
        elif r < 0.70:
            # insert
            c = rnd.choice(all_pairs)
            pos = rnd.randint(0, n)
            cand = comp[:pos] + [c] + comp[pos:]
            cost = 1  # +1 (worse)
        elif r < 0.85 and n > 1:
            # swap adjacent
            idx = rnd.randint(0, n - 2)
            a, b = comp[idx], comp[idx + 1]
            if a[0] == b[0] or a[0] == b[1] or a[1] == b[0] or a[1] == b[1]:
                # shares a wire; swapping changes semantics, still allow
                cand = comp[:idx] + [b, a] + comp[idx + 2:]
            else:
                cand = comp[:idx] + [b, a] + comp[idx + 2:]
            cost = 0
        elif r < 0.95 and n > 1:
            # relocate
            idx = rnd.randint(0, n - 1)
            c = comp[idx]
            pos = rnd.randint(0, n - 1)
            cand = comp[:idx] + comp[idx + 1:]
            if pos > idx:
                pos -= 1
            cand = cand[:pos] + [c] + cand[pos:]
            cost = 0
        else:
            # 2-opt: reverse a short segment
            if n < 3:
                continue
            idx = rnd.randint(0, n - 3)
            ln = rnd.randint(2, min(4, n - idx))
            seg = comp[idx:idx + ln][::-1]
            cand = comp[:idx] + seg + comp[idx + ln:]
            cost = 0

        if not _verify(cand):
            continue

        if cost < 0:
            # improvement
            comp = cand
            n = len(comp)
            if n < best_n:
                best = list(comp)
                best_n = n
                t = 3.0  # reheat a bit
        else:
            # accept with probability
            if rnd.random() < np.exp(-cost / max(t, 1e-6)):
                comp = cand
                n = len(comp)
        t *= cooling

    return best


def construct():
    t_start = time.time()
    deadline = t_start + 80.0  # safety margin before 90s kill
    comp = _build_seed()

    if not _verify(comp):
        return comp

    # Phase 1: aggressive pruning from seed
    comp = _prune(comp, deadline)
    best = list(comp)
    best_len = len(comp)

    # Phase 2: multiple SA restarts from perturbed best
    if time.time() < deadline - 10.0:
        seeds = [1, 7, 13, 23, 42, 99, 123, 456, 789, 1000, 2024, 31415, 27182, 16180, 55555]
        for seed_val in seeds:
            if time.time() > deadline - 10.0:
                break
            rnd = random.Random(seed_val)
            trial = list(best)
            # perturb: random removals
            for _ in range(min(3, len(trial) // 5)):
                idx = rnd.randint(0, len(trial) - 1)
                trial = trial[:idx] + trial[idx + 1:]
                if not _verify(trial):
                    trial = list(best)
                    break
            trial = _sa_search(trial, deadline - 8.0, rnd)
            if len(trial) < best_len and _verify(trial):
                best = trial
                best_len = len(trial)

    # Phase 3: final pruning pass on best
    if time.time() < deadline:
        best = _prune(best, deadline)

    return best
