# EVOLVE-BLOCK-START
import numpy as np
import time

_P = 7
_Q = 49
_K = 3
_TOTAL = _Q * _Q  # 2401
_PUNCTURED = _Q - 1  # 48


def nikodym_set() -> np.ndarray:
    deadline = time.time() + 540

    # F_49 element representation: integer 0..48, where i = a*7 + b represents a + b*alpha
    # Element 1 = (1,0) -> 7; Element 0 = (0,0) -> 0
    ONE = 7
    ZERO = 0

    # Precompute addition and multiplication tables
    add_tbl = np.zeros((_Q, _Q), dtype=np.int32)
    mul_tbl = np.zeros((_Q, _Q), dtype=np.int32)

    for a1 in range(_P):
        for b1 in range(_P):
            e1 = a1 * _P + b1
            for a2 in range(_P):
                for b2 in range(_P):
                    e2 = a2 * _P + b2
                    add_tbl[e1, e2] = ((a1 + a2) % _P) * _P + ((b1 + b2) % _P)
                    ma = (a1 * a2 + b1 * b2 * _K) % _P
                    mb = (a1 * b2 + a2 * b1) % _P
                    mul_tbl[e1, e2] = ma * _P + mb

    # Directions: 50 total
    # d in 0..48: v = (ONE, d)
    # d = 49: v = (ZERO, ONE)
    n_dirs = _Q + 1  # 50

    # Precompute punctured line indices: punct_lines[pt_idx, d] = 48 point indices
    punct_lines = np.zeros((_TOTAL, n_dirs, _PUNCTURED), dtype=np.int32)

    for x in range(_Q):
        for y in range(_Q):
            pt_idx = x * _Q + y
            for d in range(n_dirs):
                if d < _Q:
                    vx = ONE
                    vy = d
                else:
                    vx = ZERO
                    vy = ONE

                cnt = 0
                for t in range(_Q):
                    if t == ZERO:
                        continue
                    tx = mul_tbl[t, vx]
                    ty = mul_tbl[t, vy]
                    nx = add_tbl[x, tx]
                    ny = add_tbl[y, ty]
                    punct_lines[pt_idx, d, cnt] = nx * _Q + ny
                    cnt += 1

    # Greedy: start with N = all points, try to remove points
    best_removed = 0
    best_in_N = None

    n_restarts = 800
    rng = np.random.default_rng(42)

    for restart in range(n_restarts):
        if time.time() > deadline - 20:
            break

        in_N = np.ones(_TOTAL, dtype=np.bool_)
        removed = 0
        order = rng.permutation(_TOTAL)

        for pt_idx in order:
            if not in_N[pt_idx]:
                continue
            # Try each direction
            for d in range(n_dirs):
                line_pts = punct_lines[pt_idx, d]
                if np.all(in_N[line_pts]):
                    in_N[pt_idx] = False
                    removed += 1
                    break

        if removed > best_removed:
            best_removed = removed
            best_in_N = in_N.copy()

    # Build result
    if best_in_N is None:
        all_pts = []
        for xa in range(_P):
            for xb in range(_P):
                for ya in range(_P):
                    for yb in range(_P):
                        all_pts.append([[xa, xb], [ya, yb]])
        return np.array(all_pts, dtype=np.int64)

    result = []
    for pt_idx in range(_TOTAL):
        if best_in_N[pt_idx]:
            x = pt_idx // _Q
            y = pt_idx % _Q
            xa, xb = x // _P, x % _P
            ya, yb = y // _P, y % _P
            result.append([[xa, xb], [ya, yb]])

    return np.array(result, dtype=np.int64)


# EVOLVE-BLOCK-END
