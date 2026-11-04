import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 143.0  # 7s safety margin before 150s kill

    nm, mp, pn = n * m, m * p, p * n

    def verify(U, V, W, tol=1e-9):
        r = U.shape[1]
        res = np.zeros((n, m, p))
        for c in range(r):
            Uc = U[:, c].reshape(n, m)
            Vc = V[:, c].reshape(m, p)
            Wc = W[:, c].reshape(p, n)
            res += Uc[:, :, np.newaxis] * Vc[np.newaxis, :, :] * Wc.T[:, np.newaxis, :]
        return np.allclose(res, 1.0, atol=tol)

    def snap_halfint(M):
        return np.round(M * 2) / 2

    def snap_verify(U, V, W, tol=1e-8):
        U2, V2, W2 = snap_halfint(U), snap_halfint(V), snap_halfint(W)
        r = U2.shape[1]
        res = np.zeros((n, m, p))
        for c in range(r):
            Uc = U2[:, c].reshape(n, m)
            Vc = V2[:, c].reshape(m, p)
            Wc = W2[:, c].reshape(p, n)
            res += Uc[:, :, np.newaxis] * Vc[np.newaxis, :, :] * Wc.T[:, np.newaxis, :]
        if np.allclose(res, 1.0, atol=tol):
            return U2, V2, W2
        return None

    def compute_error(U, V, W):
        r = U.shape[1]
        res = np.zeros((n, m, p))
        for c in range(r):
            Uc = U[:, c].reshape(n, m)
            Vc = V[:, c].reshape(m, p)
            Wc = W[:, c].reshape(p, n)
            res += Uc[:, :, np.newaxis] * Vc[np.newaxis, :, :] * Wc.T[:, np.newaxis, :]
        return np.sum((res - 1.0) ** 2)

    def build_structured(r_target):
        """Build a structured decomposition: Strassen-like for j=0,1,k=0,1;
        direct terms for the rest. Returns U,V,W of rank r_target (may have zero cols)."""
        U = np.zeros((nm, r_target))
        V = np.zeros((mp, r_target))
        W = np.zeros((pn, r_target))
        r = 0

        # Direct terms: j=2, all k (n*p = 10 terms)
        for i in range(n):
            for k in range(p):
                if r >= r_target: break
                U[i * m + 2, r] = 1.0
                V[2 * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
                r += 1
        if r >= r_target:
            return U[:, :r_target], V[:, :r_target], W[:, :r_target]

        # Direct terms: j=0,1, k=2,3,4 (2*2*3 = 12 terms)
        for i in range(n):
            for j in range(2):
                for k in range(2, p):
                    if r >= r_target: break
                    U[i * m + j, r] = 1.0
                    V[j * p + k, r] = 1.0
                    W[k * n + i, r] = 1.0
                    r += 1
        if r >= r_target:
            return U[:, :r_target], V[:, :r_target], W[:, :r_target]

        # Strassen for j=0,1, k=0,1 (7 terms)
        # M1 = (A00+A11)(B00+B11)
        if r < r_target:
            U[0 * m + 0, r] = 1.0; U[1 * m + 1, r] = 1.0
            V[0 * p + 0, r] = 1.0; V[1 * p + 1, r] = 1.0
            W[0 * n + 0, r] = 1.0; W[1 * n + 1, r] = 1.0
            r += 1
        # M2 = (A10+A11)B00
        if r < r_target:
            U[1 * m + 0, r] = 1.0; U[1 * m + 1, r] = 1.0
            V[0 * p + 0, r] = 1.0
            W[0 * n + 1, r] = 1.0; W[1 * n + 1, r] = -1.0
            r += 1
        # M3 = A00(B10-B11)
        if r < r_target:
            U[0 * m + 0, r] = 1.0
            V[0 * p + 1, r] = 1.0; V[1 * p + 1, r] = -1.0
            W[1 * n + 0, r] = 1.0; W[1 * n + 1, r] = 1.0
            r += 1
        # M4 = A11(B00-B01)
        if r < r_target:
            U[1 * m + 1, r] = 1.0
            V[0 * p + 0, r] = 1.0; V[0 * p + 1, r] = -1.0
            W[0 * n + 1, r] = 1.0; W[1 * n + 1, r] = -1.0
            r += 1
        # M5 = (A00+A01)B11
        if r < r_target:
            U[0 * m + 0, r] = 1.0; U[0 * m + 1, r] = 1.0
            V[1 * p + 1, r] = 1.0
            W[1 * n + 0, r] = 1.0; W[1 * n + 1, r] = -1.0
            r += 1
        # M6 = (A10-A00)(B00+B01)
        if r < r_target:
            U[1 * m + 0, r] = 1.0; U[0 * m + 0, r] = -1.0
            V[0 * p + 0, r] = 1.0; V[0 * p + 1, r] = 1.0
            W[0 * n + 1, r] = 1.0; W[0 * n + 0, r] = -1.0
            r += 1
        # M7 = (A01-A11)(B10+B11)
        if r < r_target:
            U[0 * m + 1, r] = 1.0; U[1 * m + 1, r] = -1.0
            V[1 * p + 0, r] = 1.0; V[1 * p + 1, r] = 1.0
            W[1 * n + 0, r] = 1.0; W[1 * n + 1, r] = -1.0
            r += 1

        return U[:, :r_target], V[:, :r_target], W[:, :r_target]

    # Verify the structured construction
    def verify_structured(r_target):
        U, V, W = build_structured(r_target)
        if verify(U, V, W):
            snapped = snap_verify(U, V, W)
            if snapped:
                return snapped
            return U, V, W
        return None

    # Get a valid baseline
    baseline = None
    for r_test in [26, 27, 28, 30]:
        baseline = verify_structured(r_test)
        if baseline:
            break

    # Fallback to schoolbook
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    if baseline is None:
        U, V, W = schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_U, best_V, best_W = baseline
    best_rank = best_U.shape[1]

    # ALS with constraint-aware solving
    def solve_U(U, V, W):
        r = U.shape[1]
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, r))
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                b = np.ones(p)
                try:
                    U_new[i * m + j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    U_new[i * m + j, :] = U[i * m + j, :]
        return U_new

    def solve_V(U, V, W):
        r = V.shape[1]
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, r))
                for i in range(n):
                    A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                b = np.ones(n)
                try:
                    V_new[j * p + k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    V_new[j * p + k, :] = V[j * p + k, :]
        return V_new

    def solve_W(U, V, W):
        r = W.shape[1]
        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, r))
                for j in range(m):
                    A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                b = np.ones(m)
                try:
                    W_new[k * n + i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    W_new[k * n + i, :] = W[k * n + i, :]
        return W_new

    def als_polish(U, V, W, max_iters=100):
        prev_err = np.inf
        for it in range(max_iters):
            if time.time() > deadline:
                break
            U = solve_U(U, V, W)
            V = solve_V(U, V, W)
            W = solve_W(U, V, W)
            err = compute_error(U, V, W)
            if err < 1e-12:
                break
            if err > prev_err * 1.05 and it > 5:
                break
            prev_err = err
        return U, V, W

    # Try to reduce rank via ALS from structured init
    for r in range(20, best_rank):
        if time.time() > deadline:
            break
        if deadline - time.time() < 5.0:
            break

        U0, V0, W0 = build_structured(r)
        # Zero out unused columns
        U0 = U0[:, :r]
        V0 = V0[:, :r]
        W0 = W0[:, :r]

        U, V, W = als_polish(U0, V0, W0, max_iters=50)
        if verify(U, V, W):
            snapped = snap_verify(U, V, W)
            if snapped:
                best_U, best_V, best_W = snapped
                best_rank = r
                break
            else:
                best_U, best_V, best_W = U, V, W
                best_rank = r
                break

    # Also try random init ALS for lower ranks
    for r in range(18, best_rank):
        if time.time() > deadline:
            break
        if deadline - time.time() < 3.0:
            break

        for restart in range(3):
            if time.time() > deadline:
                break
            np.random.seed(restart * 1000 + r)
            scale = 1.0 / np.sqrt(r)
            U = np.random.randn(nm, r) * scale
            V = np.random.randn(mp, r) * scale
            W = np.random.randn(pn, r) * scale
            U, V, W = als_polish(U, V, W, max_iters=80)
            if verify(U, V, W):
                snapped = snap_verify(U, V, W)
                if snapped:
                    best_U, best_V, best_W = snapped
                    best_rank = r
                    break

    # Final polish on best solution
    U, V, W = als_polish(best_U, best_V, best_W, max_iters=30)
    snapped = snap_verify(U, V, W)
    if snapped:
        best_U, best_V, best_W = snapped

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
