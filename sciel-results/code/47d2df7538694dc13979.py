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
    deadline = start_time + 110  # safety margin under the 120s kill

    nm, mp, pn = n * m, m * p, p * n
    target = np.ones((n, m, p))
    rank_school = n * m * p  # 84

    # ---- exact integer verification ----
    def verify_exact(Ui, Vi, Wi):
        Ui = np.asarray(Ui, dtype=np.int64)
        Vi = np.asarray(Vi, dtype=np.int64)
        Wi = np.asarray(Wi, dtype=np.int64)
        r = Ui.shape[1]
        # Check diagonal = 1 and off-diagonal = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += Ui[i * m + j, c] * Vi[j * p + k, c] * Wi[k * n + i, c]
                    if s != 1:
                        return False
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
        T = np.zeros((n, m, p))
        for c in range(U.shape[1]):
            uc = U[:, c].reshape(n, m)
            vc = V[:, c].reshape(m, p)
            wc = W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    T[i, j, :] += uc[i, j] * vc[j, :] * wc[:, i]
        return np.allclose(T, 1.0, atol=tol)

    # ---- exact rational polish: given U,V (floats), solve for W exactly and round ----
    def polish_and_round(U, V, W, denom_list=(2, 4, 10, 20, 50, 100, 200, 500)):
        r = U.shape[1]
        # Build the linear system for W from U,V (floats)
        # Equations: for all (i,j,k) and (ip,jp,kp):
        #   sum_c U[i*m+j,c] V[jp*p+k,c] W[kp*n+ip,c] = delta
        # Unknowns: W flat (pn * r)
        eqs = []
        rhs = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for ip in range(n):
                        for jp in range(m):
                            for kp in range(p):
                                row = {}
                                for c in range(r):
                                    coeff = U[i * m + j, c] * V[jp * p + k, c]
                                    if abs(coeff) > 1e-12:
                                        row[(kp * n + ip) * r + c] = coeff
                                b = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                                eqs.append((row, b))
        rows = []
        cols = []
        vals = []
        bvec = []
        for row, b in eqs:
            for idx, val in row.items():
                rows.append(len(bvec))
                cols.append(idx)
                vals.append(val)
            bvec.append(b)
        A = sp.csr_matrix((vals, (rows, cols)), shape=(len(bvec), pn * r))
        try:
            x = spla.lsqr(A, np.array(bvec), atol=1e-14, btol=1e-14, itermax=5000)[0]
        except Exception:
            x = None
        if x is None:
            return None
        Ws = x.reshape(pn, r)
        # round U, V, W with increasing denominators and verify exactly
        for denom in denom_list:
            Ur = np.array([[float(Fraction(float(v)).limit_denominator(denom)) for v in row] for row in U])
            Vr = np.array([[float(Fraction(float(v)).limit_denominator(denom)) for v in row] for row in V])
            Wr = np.array([[float(Fraction(float(v)).limit_denominator(denom)) for v in row] for row in Ws])
            # convert to integer multiples of 1/denom
            Ui = (Ur * denom).round()
            Vi = (Vr * denom).round()
            Wi = (Wr * denom).round()
            if np.allclose(Ur, Ui / denom, atol=1e-9) and np.allclose(Vr, Vi / denom, atol=1e-9) and np.allclose(Wr, Wi / denom, atol=1e-9):
                # exact check with integers (denom cancels as denom^3 * (1/denom^3) = 1)
                if verify_exact(Ui, Vi, Wi):
                    # scale back: U=Ui/denom etc. are half-integers if denom even
                    return [float(v) for row in (Ui / denom) for v in row], \
                           [float(v) for row in (Vi / denom) for v in row], \
                           [float(v) for row in (Wi / denom) for v in row], r
        return None

    # ---- ALS candidate search ----
    def compute_tensor(U, V, W):
        T = np.zeros((n, m, p))
        for c in range(U.shape[1]):
            uc = U[:, c].reshape(n, m)
            vc = V[:, c].reshape(m, p)
            wc = W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    T[i, j, :] += uc[i, j] * vc[j, :] * wc[:, i]
        return T

    def als_solve(r, seed, max_iters=500):
        rng = np.random.RandomState(seed)
        U = rng.randn(nm, r)
        V = rng.randn(mp, r)
        W = rng.randn(pn, r)
        for it in range(max_iters):
            if time.time() > deadline:
                break
            # Update U
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, r))
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    try:
                        x = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                    except Exception:
                        x = np.zeros(r)
                    U[i * m + j, :] = x
            # Update V
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, r))
                    for i in range(n):
                        A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        x = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                    except Exception:
                        x = np.zeros(r)
                    V[j * p + k, :] = x
            # Update W
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, r))
                    for j in range(m):
                        A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        x = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                    except Exception:
                        x = np.zeros(r)
                    W[k * n + i, :] = x
            T = compute_tensor(U, V, W)
            loss = np.sum((T - target) ** 2)
            if loss < 1e-14:
                break
        T = compute_tensor(U, V, W)
        loss = np.sum((T - target) ** 2)
        if loss < 1e-8:
            res = polish_and_round(U, V, W)
            if res is not None:
                return res
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
        seeds = [r, r + 137, r + 271, r + 409, r + 541, r + 673, r + 811, r + 947]
        for seed in seeds:
            if time.time() > deadline:
                break
            res = als_solve(r, seed, max_iters=400)
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
