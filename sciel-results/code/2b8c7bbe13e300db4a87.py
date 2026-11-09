"""Combined program: ALS candidate search + exact linear-system refinement + exact integer verification."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
from fractions import Fraction
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _strassen_lower_bound(n, m, p):
    """Lower bound on the rank of multiplying an n x m by m x p matrix."""
    r = n * m * p
    for n2 in range(1, n):
        for m2 in range(1, m):
            for p2 in range(1, p):
                if n % n2 == 0 and m % m2 == 0 and p % p2 == 0:
                    pairs = (n // n2) * (m // m2) * (p // p2)
                    r = min(r, 7 * pairs)
    return r


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # safety margin under the 150s kill

    nm, mp, pn = n * m, m * p, p * n
    rank_school = n * m * p

    # ---- exact integer verification (optimized with einsum) ----
    def verify_exact(Ui, Vi, Wi):
        Ui = np.asarray(Ui, dtype=np.int64)
        Vi = np.asarray(Vi, dtype=np.int64)
        Wi = np.asarray(Wi, dtype=np.int64)
        r = Ui.shape[1]
        U3 = Ui.reshape(n, m, r)
        V3 = Vi.reshape(m, p, r)
        W3 = Wi.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        return np.all(T == 1)

    def verify_half_int(Uh, Vh, Wh):
        # Multiply by 2 to get integers, then verify
        Ui = np.asarray(Uh, dtype=np.float64) * 2
        Vi = np.asarray(Vh, dtype=np.float64) * 2
        Wi = np.asarray(Wh, dtype=np.float64) * 2
        if not (np.allclose(Ui, np.round(Ui), atol=1e-9) and
                np.allclose(Vi, np.round(Vi), atol=1e-9) and
                np.allclose(Wi, np.round(Wi), atol=1e-9)):
            return False
        Ui = np.round(Ui).astype(np.int64)
        Vi = np.round(Vi).astype(np.int64)
        Wi = np.round(Wi).astype(np.int64)
        r = Ui.shape[1]
        U3 = Ui.reshape(n, m, r)
        V3 = Vi.reshape(m, p, r)
        W3 = Wi.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        # T should be 4 for all entries (since each factor was doubled)
        return np.all(T == 4)

    # ---- polish: try rounding to integers and half-integers ----
    def polish_and_round(U, V, W):
        r = U.shape[1]
        # Try integer rounding
        for denom in (1, 2):
            Ur = (U * denom).round()
            Vr = (V * denom).round()
            Wr = (W * denom).round()
            if np.allclose(U, Ur / denom, atol=1e-8) and \
               np.allclose(V, Vr / denom, atol=1e-8) and \
               np.allclose(W, Wr / denom, atol=1e-8):
                if denom == 1:
                    if verify_exact(Ur, Vr, Wr):
                        return [float(v) for row in Ur for v in row], \
                               [float(v) for row in Vr for v in row], \
                               [float(v) for row in Wr for v in row], r
                else:
                    if verify_half_int(Ur, Vr, Wr):
                        return [float(v) for row in (Ur / 2) for v in row], \
                               [float(v) for row in (Vr / 2) for v in row], \
                               [float(v) for row in (Wr / 2) for v in row], r
        return None

    # ---- ALS with better initialization and line search ----
    def compute_tensor(U, V, W):
        rr = U.shape[1]
        U3 = U.reshape(n, m, rr)
        V3 = V.reshape(m, p, rr)
        W3 = W.reshape(p, n, rr)
        return np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)

    def als_solve(r, seed, max_iters=150):
        rng = np.random.RandomState(seed)
        
        # Initialize with small random values scaled
        scale = 0.5
        U = rng.randn(nm, r) * scale
        V = rng.randn(mp, r) * scale
        W = rng.randn(pn, r) * scale
        
        # Pre-compute views
        U3 = U.reshape(n, m, r)
        V3 = V.reshape(m, p, r)
        W3 = W.reshape(p, n, r)
        
        best_loss = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            # Update U: for each (i,j), solve min ||A^T x - 1|| where A[k,c] = V3[j,k,c]*W3[k,i,c]
            for i in range(n):
                for j in range(m):
                    A = V3[j, :, :] * W3[:, i, :]  # (p, r)
                    A_t = A.T  # (r, p)
                    try:
                        x, _, _, _ = np.linalg.lstsq(A_t, np.ones(p), rcond=1e-12)
                    except Exception:
                        x = U[i * m + j, :].copy()
                    U[i * m + j, :] = x
            
            U3 = U.reshape(n, m, r)
            
            # Update V: for each (j,k), solve min ||A^T x - 1|| where A[i,c] = U3[i,j,c]*W3[k,i,c]
            for j in range(m):
                for k in range(p):
                    A = U3[:, j, :] * W3[k, :, :]  # (n, r)
                    A_t = A.T  # (r, n)
                    try:
                        x, _, _, _ = np.linalg.lstsq(A_t, np.ones(n), rcond=1e-12)
                    except Exception:
                        x = V[j * p + k, :].copy()
                    V[j * p + k, :] = x
            
            V3 = V.reshape(m, p, r)
            
            # Update W: for each (k,i), solve min ||A^T x - 1|| where A[j,c] = U3[i,j,c]*V3[j,k,c]
            for k in range(p):
                for i in range(n):
                    A = U3[i, :, :] * V3[:, k, :]  # (m, r)
                    A_t = A.T  # (r, m)
                    try:
                        x, _, _, _ = np.linalg.lstsq(A_t, np.ones(m), rcond=1e-12)
                    except Exception:
                        x = W[k * n + i, :].copy()
                    W[k * n + i, :] = x
            
            W3 = W.reshape(p, n, r)
            
            # Compute loss
            T = compute_tensor(U, V, W)
            loss = np.sum((T - 1.0) ** 2)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if loss < 1e-12:
                break
        
        # Use best found
        U, V, W = best_U, best_V, best_W
        T = compute_tensor(U, V, W)
        loss = np.sum((T - 1.0) ** 2)
        
        if loss < 1e-6:
            res = polish_and_round(U, V, W)
            if res is not None:
                return res
        
        # Try a few more polish attempts with slightly different rounding
        if loss < 1e-4:
            for denom in (1, 2, 4, 5):
                Ur = (U * denom).round()
                Vr = (V * denom).round()
                Wr = (W * denom).round()
                if denom == 1:
                    if verify_exact(Ur, Vr, Wr):
                        return [float(v) for row in Ur for v in row], \
                               [float(v) for row in Vr for v in row], \
                               [float(v) for row in Wr for v in row], r
                else:
                    # Check if close to half-integers or other rationals
                    if np.allclose(U, Ur / denom, atol=1e-6) and \
                       np.allclose(V, Vr / denom, atol=1e-6) and \
                       np.allclose(W, Wr / denom, atol=1e-6):
                        # Verify with scaled integers
                        Ui = Ur.astype(np.int64)
                        Vi = Vr.astype(np.int64)
                        Wi = Wr.astype(np.int64)
                        r2 = Ui.shape[1]
                        U3v = Ui.reshape(n, m, r2)
                        V3v = Vi.reshape(m, p, r2)
                        W3v = Wi.reshape(p, n, r2)
                        T2 = np.einsum('ijc,jkc,kic->ijk', U3v, V3v, W3v)
                        if np.all(T2 == denom ** 3):
                            return [float(v) for row in (Ur / denom) for v in row], \
                                   [float(v) for row in (Vr / denom) for v in row], \
                                   [float(v) for row in (Wr / denom) for v in row], r
        
        return None

    # ---- schoolbook fallback ----
    U_sb = np.zeros((nm, rank_school))
    V_sb = np.zeros((mp, rank_school))
    W_sb = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    best = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_school

    lower = _strassen_lower_bound(n, m, p)
    
    # Try ranks from just above the lower bound up to schoolbook-1
    for r in range(max(lower, 1), rank_school):
        if time.time() > deadline:
            break
        found = False
        # More seeds for better coverage
        seeds = [r, r + 137, r + 271, r + 409, r + 541, r + 673, 
                 r + 811, r + 947, r + 1081, r + 1217]
        for seed in seeds:
            if time.time() > deadline:
                break
            res = als_solve(r, seed, max_iters=120)
            if res is not None:
                U_l, V_l, W_l, rr = res
                best = {"U": U_l, "V": V_l, "W": W_l}
                best_rank = rr
                found = True
                break
        if found:
            break

    return best
# EVOLVE-BLOCK-END
