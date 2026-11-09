"""Combined program: fast vectorized ALS + exact rational polish + integer verification."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
from fractions import Fraction
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _lower_bound(n, m, p):
    """Strassen block lower bound."""
    r = n * m * p
    for n2 in range(1, n + 1):
        for m2 in range(1, m + 1):
            for p2 in range(1, p + 1):
                if n % n2 == 0 and m % m2 == 0 and p % p2 == 0:
                    pairs = (n // n2) * (m // m2) * (p // p2)
                    r = min(r, 7 * pairs)
    return r


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    nm, mp, pn = n * m, m * p, p * n
    rank_school = n * m * p

    # Precompute index arrays for fast tensor computation
    # U indexed by (i,j) -> i*m+j, V by (j,k) -> j*p+k, W by (k,i) -> k*n+i
    def compute_tensor_fast(U, V, W):
        """Compute T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]"""
        r = U.shape[1]
        T = np.zeros((n, m, p))
        for c in range(r):
            uc = U[:, c].reshape(n, m)
            vc = V[:, c].reshape(m, p)
            wc = W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    T[i, j, :] += uc[i, j] * vc[j, :] * wc[:, i]
        return T

    def verify_exact_int(Ui, Vi, Wi):
        """Exact integer verification. Ui,Vi,Wi are integer arrays."""
        r = Ui.shape[1]
        # Check diagonal: T[i,j,k] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += Ui[i * m + j, c] * Vi[j * p + k, c] * Wi[k * n + i, c]
                    if s != 1:
                        return False
        # Check off-diagonal: T[i,j,k] = 0 for (i,j,k) != (ip,jp,kp)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for ip in range(n):
                        for jp in range(m):
                            for kp in range(p):
                                if i == ip and j == jp and k == kp:
                                    continue
                                s = 0
                                for c in range(r):
                                    s += Ui[i * m + j, c] * Vi[jp * p + k, c] * Wi[kp * n + ip, c]
                                if s != 0:
                                    return False
        return True

    def verify_numeric(U, V, W, tol=1e-9):
        T = compute_tensor_fast(U, V, W)
        return np.allclose(T, np.ones((n, m, p)), atol=tol)

    def polish_and_round(U, V, W, denom_list=(2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)):
        """Given float U,V,W, solve for W exactly from U,V, then round all to rationals."""
        r = U.shape[1]
        # Build linear system for W: sum_c U[i*m+j,c]*V[jp*p+k,c]*W[kp*n+ip,c] = delta
        eqs_rows = []
        eqs_cols = []
        eqs_vals = []
        bvec = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for ip in range(n):
                        for jp in range(m):
                            for kp in range(p):
                                for c in range(r):
                                    coeff = U[i * m + j, c] * V[jp * p + k, c]
                                    if abs(coeff) > 1e-14:
                                        eqs_rows.append(len(bvec))
                                        eqs_cols.append((kp * n + ip) * r + c)
                                        eqs_vals.append(coeff)
                                b = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                                bvec.append(b)
        A = sp.csr_matrix((eqs_vals, (eqs_rows, eqs_cols)), shape=(len(bvec), pn * r))
        try:
            x = spla.lsqr(A, np.array(bvec), atol=1e-13, btol=1e-13, itermax=5000)[0]
        except Exception:
            return None
        Ws = x.reshape(pn, r)
        # Try rounding to rationals with various denominators
        for denom in denom_list:
            inv_denom = 1.0 / denom
            Ur = np.round(U * denom) / denom
            Vr = np.round(V * denom) / denom
            Wr = np.round(Ws * denom) / denom
            # Check if rounding is accurate
            if not (np.allclose(U, Ur, atol=1e-10) and np.allclose(V, Vr, atol=1e-10) and np.allclose(Ws, Wr, atol=1e-10)):
                continue
            Ui = (Ur * denom).round().astype(int)
            Vi = (Vr * denom).round().astype(int)
            Wi = (Wr * denom).round().astype(int)
            if verify_exact_int(Ui, Vi, Wi):
                return Ur.flatten().tolist(), Vr.flatten().tolist(), Wr.flatten().tolist(), r
        return None

    def als_solve_fast(r, seed, max_iters=600):
        """Fast ALS with vectorized operations."""
        rng = np.random.RandomState(seed)
        U = rng.randn(nm, r)
        V = rng.randn(mp, r)
        W = rng.randn(pn, r)
        target = np.ones((n, m, p))
        for it in range(max_iters):
            # Update U: for each (i,j), solve A @ x = ones(p) where A[k,c] = V[j*p+k,c]*W[k*n+i,c]
            for i in range(n):
                for j in range(m):
                    # A is p x r
                    A_mat = np.zeros((p, r))
                    for k in range(p):
                        A_mat[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    try:
                        x = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)[0]
                    except Exception:
                        x = np.zeros(r)
                    U[i * m + j, :] = x
            # Update V: for each (j,k), solve A @ x = ones(n) where A[i,c] = U[i*m+j,c]*W[k*n+i,c]
            for j in range(m):
                for k in range(p):
                    A_mat = np.zeros((n, r))
                    for i in range(n):
                        A_mat[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        x = np.linalg.lstsq(A_mat, np.ones(n), rcond=None)[0]
                    except Exception:
                        x = np.zeros(r)
                    V[j * p + k, :] = x
            # Update W: for each (k,i), solve A @ x = ones(m) where A[j,c] = U[i*m+j,c]*V[j*p+k,c]
            for k in range(p):
                for i in range(n):
                    A_mat = np.zeros((m, r))
                    for j in range(m):
                        A_mat[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        x = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)[0]
                    except Exception:
                        x = np.zeros(r)
                    W[k * n + i, :] = x
            # Check convergence
            T = compute_tensor_fast(U, V, W)
            loss = np.sum((T - target) ** 2)
            if loss < 1e-14:
                break
        T = compute_tensor_fast(U, V, W)
        loss = np.sum((T - target) ** 2)
        if loss < 1e-8:
            res = polish_and_round(U, V, W)
            if res is not None:
                return res
        return None

    # Schoolbook fallback
    U_sb = np.zeros((nm, rank_school), dtype=int)
    V_sb = np.zeros((mp, rank_school), dtype=int)
    W_sb = np.zeros((pn, rank_school), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    best = {"U": U_sb.flatten().tolist(), "V": V_sb.flatten().tolist(), "W": W_sb.flatten().tolist()}
    best_rank = rank_school

    lower = _lower_bound(n, m, p)
    # Try ranks from lower bound up to schoolbook-1
    for r in range(max(lower, 1), rank_school):
        if time.time() > deadline:
            break
        found = False
        # More seeds for better coverage
        seeds = list(range(r, r + 20)) + [r * 137, r * 271 + 1, r * 409 + 2]
        for seed in seeds:
            if time.time() > deadline:
                break
            res = als_solve_fast(r, seed, max_iters=500)
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
