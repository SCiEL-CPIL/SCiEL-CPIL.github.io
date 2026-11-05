"""Matrix multiplication 2x4 x 4x5 — rank-minimal tensor decomposition search."""
import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 4, 5]

T0 = time.time()
DEADLINE = 140.0  # explicit safety margin below the 150s wall-clock kill


def _build_tensors():
    n, m, p = DIMENSIONS
    # U0 rows indexed by (i, j) flattened i*m + j
    U0 = np.zeros((n * m, n * m * p))
    V0 = np.zeros((m * p, n * m * p))
    W0 = np.zeros((p * n, n * m * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * (m * p) + j * p + k
                U0[i * m + j, c] = 1
                V0[j * p + k, c] = 1
                W0[k * n + i, c] = 1
    return U0, V0, W0


def _round_half(A):
    # Round each entry to the nearest half-integer (multiples of 0.5)
    return np.rint(2.0 * A) / 2.0


def _exact_match(U, V, W, U0, V0, W0):
    # Verify the rounded factors exactly reproduce the target tensor via Fraction
    R = U.shape[1]
    n, m, p = DIMENSIONS
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(R):
                    a = Fraction(int(round(U[i * m + j, r] * 2))) / 2
                    b = Fraction(int(round(V[j * p + k, r] * 2))) / 2
                    c = Fraction(int(round(W[k * n + i, r] * 2))) / 2
                    s += a * b * c
                if s != 1:
                    ok = False
                    return False
    return ok


def _validate(U, V, W):
    n, m, p = DIMENSIONS
    U0, V0, W0 = _build_tensors()
    U, V, W = _round_half(U), _round_half(V), _round_half(W)
    if _exact_match(U, V, W, U0, V0, W0):
        return True, U, V, W
    return False, U, V, W


def _ls_fit_fixed(U, V, W, U0, V0, W0, R):
    # Fix U,V; re-solve W by least squares, then round and check.
    n, m, p = DIMENSIONS
    A = np.zeros((n * m * p, R))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                row = (i * m + j) * (m * p) + (j * p + k)
                for r in range(R):
                    A[row, r] = U[i * m + j, r] * V[j * p + k, r]
    b = W0.reshape(-1)
    try:
        Wsol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        return W
    Wr = _round_half(Wsol)
    ok, _, Wc, _ = _validate(U, V, Wr, U0, V0, W0)
    return Wc if ok else W


def _ls_all(U, V, W, U0, V0, W0, R, n_iters=6):
    # Alternate least-squares over each factor while fixing the others.
    for _ in range(n_iters):
        # fit W
        W = _ls_fit_fixed(U, V, W, U0, V0, W0, R)
        # fit V (fix U, W): linear in V
        n, m, p = DIMENSIONS
        A = np.zeros((n * m * p, R))
        b = np.zeros((m * p * R))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(r for r in range(R)):
                        pass
        # Build A and b properly
        A = np.zeros((n * m * p, R))
        b = np.zeros(R)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row = (i * m + j) * (m * p) + (j * p + k)
                    for r in range(R):
                        A[row, r] = U[i * m + j, r] * W[k * n + i, r]
                    b[row] = V0[j * p + k, r] if False else 1.0
        # Note: target for V is the one-hot V0 pattern, but we want the product to
        # equal 1. We minimize ||A x - V0_flat||. Use V0 as target.
        b = V0.reshape(-1)
        try:
            Vsol, *_ = np.linalg.lstsq(A, b, rcond=None)
        except Exception:
            continue
        Vr = _round_half(Vsol)
        ok, _, _, Wc = _validate(U, Vr, W, U0, V0, W0)
        if ok:
            V = Vr
            W = Wc
        else:
            # try to improve V by LS even if not yet exact
            V = Vr
        # fit U
        A = np.zeros((n * m * p, R))
        b = U0.reshape(-1)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row = (i * m + j) * (m * p) + (j * p + k)
                    for r in range(R):
                        A[row, r] = V[j * p + k, r] * W[k * n + i, r]
        try:
            Usol, *_ = np.linalg.lstsq(A, b, rcond=None)
        except Exception:
            continue
        Ur = _round_half(Usol)
        ok, Uc, _, Wc = _validate(Ur, V, W, U0, V0, W0)
        if ok:
            U = Uc
            W = Wc
        else:
            U = Ur
    return U, V, W


def _try_reduce_rank(U, V, W, U0, V0, W0, R, rng, n_trials):
    # Try to drop components (set to zero) and re-optimize the rest.
    order = list(range(R))
    for _ in range(n_trials):
        cur = R
        U, V, W = _round_half(U), _round_half(V), _round_half(W)
        # pick a component with small magnitude to zero out
        rng.shuffle(order)
        for comp in order:
            if cur <= 1:
                break
            if time.time() > DEADLINE - 5:
                return None
            # measure component magnitude
            mag = (np.abs(U[:, comp]) * np.abs(V[:, comp]) * np.abs(W[:, comp])).sum()
            # zero it
            U[:, comp] = 0
            V[:, comp] = 0
            W[:, comp] = 0
            # re-fit with remaining components
            active = [c for c in range(R) if c != comp]
            Ua = U[:, active]
            Va = V[:, active]
            Wa = W[:, active]
            Ua, Va, Wa = _ls_all(Ua, Va, Wa, U0, V0, W0, len(active), n_iters=4)
            ok, Uc, Vc, Wc = _validate(Ua, Va, Wa, U0, V0, W0)
            if ok:
                # success: keep reduced
                cur -= 1
                U = np.zeros((U.shape[0], R))
                V = np.zeros((V.shape[0], R))
                W = np.zeros((W.shape[0], R))
                for new_idx, c in enumerate(active):
                    U[:, c] = Uc[:, new_idx]
                    V[:, c] = Vc[:, new_idx]
                    W[:, c] = Wc[:, new_idx]
                break
            else:
                # revert
                U[:, comp] = 0
                V[:, comp] = 0
                W[:, comp] = 0
    return (U, V, W, cur) if cur < R else None


def construct():
    start = time.time()
    rng = random.Random(12345)
    n, m, p = DIMENSIONS
    U0, V0, W0 = _build_tensors()
    R0 = n * m * p
    U = U0.copy()
    V = V0.copy()
    W = W0.copy()

    # Baseline: schoolbook decomposition is exact.
    ok, U, V, W = _validate(U, V, W)
    best = (R0, U.copy(), V.copy(), W.copy())

    # Iterative rank reduction.
    R = R0
    for _ in range(4000):
        if time.time() > DEADLINE - 8:
            break
        res = _try_reduce_rank(U, V, W, U0, V0, W0, R, rng, n_trials=2)
        if res is not None:
            U, V, W, R = res
            if R < best[0]:
                best = (R, U.copy(), V.copy(), W.copy())
            # continue trying to reduce further
        else:
            # perturb slightly and try again
            if time.time() > DEADLINE - 20:
                break
            # small random perturbation to escape local minima
            pert = 0.5 * rng.random()
            c = rng.randint(0, R - 1)
            W2 = W.copy()
            W2[rng.randint(0, W2.shape[0] - 1), c] += pert
            U2, V2, W2 = _ls_all(U, V, W2, U0, V0, W0, R, n_iters=2)
            ok, U2, V2, W2 = _validate(U2, V2, W2)
            if ok:
                U, V, W = U2, V2, W2

    # Final: return the best (lowest-rank) valid decomposition.
    Rb, Ub, Vb, Wb = best
    Ub = _round_half(Ub)
    Vb = _round_half(Vb)
    Wb = _round_half(Wb)
    # Convert to lists of Python numbers (int or half-integer float)
    def to_list(A):
        out = []
        for row in A:
            r = []
            for x in row:
                xf = float(x)
                r.append(xf)
            out.append(r)
        return out
    return {"U": to_list(Ub), "V": to_list(Vb), "W": to_list(Wb)}
