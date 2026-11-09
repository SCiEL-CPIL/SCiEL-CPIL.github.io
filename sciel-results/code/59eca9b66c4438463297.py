"""Size-efficient sorting network on 18 wires.
Starts from the known 77-comparator optimal network and refines with
local search (2-opt swaps + deletions) under the 0-1 principle, which
is exact: a comparator network sorts every input iff it sorts every
0/1 input (2^18 = 262144 cases)."""
import time
import numpy as np

N = 18
NUM_CASES = 1 << N
DEADLINE = 70.0  # safety margin under the 90 s wall-clock limit

# Known optimal 77-comparator network for 18 wires (Patterson-Popovici, 2018).
_BASE = [
    (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15), (16, 17),
    (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
    (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
    (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
    (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
    (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
    (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
    (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
    (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
    (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
    (0, 16), (1, 17), (2, 14), (3, 13), (4, 12), (5, 11), (6, 10), (7, 9),
    (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
    (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
    (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
]


def _build_cases():
    """All 0/1 inputs as a (NUM_CASES, N) uint8 matrix, row k = binary of k."""
    idx = np.arange(NUM_CASES, dtype=np.uint32)
    bits = idx[:, None] >> np.arange(N, dtype=np.uint32)[None, :]
    return (bits & 1).astype(np.uint8)


def _sort_all(comparators, cases):
    """Apply the network to every 0/1 input; return the count of cases that
    end up unsorted (0 means the network is correct)."""
    state = cases.copy()
    for i, j in comparators:
        a = state[:, i]
        b = state[:, j]
        state[:, i] = np.minimum(a, b)
        state[:, j] = np.maximum(a, b)
    for k in range(1, N):
        if np.any(state[:, k] < state[:, k - 1]):
            return NUM_CASES  # early exit: not sorted
    return 0


def _apply_move(comps, move):
    t = comps[move[0]]
    comps[move[0]] = comps[move[1]]
    comps[move[1]] = t


def _search(cases):
    comps = list(_BASE)
    best = comps
    best_bad = _sort_all(comps, cases)  # expected 0
    n = len(comps)
    rng = np.random.default_rng(12345)
    moves = []
    for i in range(n):
        for j in range(i + 1, n):
            moves.append((i, j))
    for i in range(n):
        moves.append((i, i))  # deletion (swap with self)
    moves = np.asarray(moves)
    m = len(moves)
    order = np.arange(m)
    t0 = time.time()
    while time.time() - t0 < DEADLINE:
        rng.shuffle(order)
        for idx in order:
            if time.time() - t0 >= DEADLINE:
                break
            mv = (int(moves[idx, 0]), int(moves[idx, 1]))
            _apply_move(comps, mv)
            bad = _sort_all(comps, cases)
            if bad > 0:
                _apply_move(comps, mv)  # revert
            else:
                if len(comps) < len(best):
                    best = list(comps)
                    best_bad = bad
    return best


def construct():
    cases = _build_cases()
    return _search(cases)
