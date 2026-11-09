import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [4, 4, 6]

def _build_base():
    """Build the base 77-rank Strassen decomposition."""
    n, m, p = 4, 4, 6

    alpha = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, 1)],
        [(0, 1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (0, -1)],
        [(1, 1), (3, -1)],
    ]
    beta = [
        [(0, 1), (3, 1)],
        [(0, 1)],
        [(1, 1), (3, -1)],
        [(2, 1), (0, -1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (3, 1)],
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)],
    ]

    inner_alpha = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, 1)],
        [(0, 1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (0, -1)],
        [(1, 1), (3, -1)],
    ]
    inner_beta = [
        [(0, 1), (3, 1)],
        [(0, 1)],
        [(1, 1), (3, -1)],
        [(2, 1), (0, -1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (3, 1)],
    ]
    inner_recon = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)],
    ]

    U = {}
    V = {}
    W = {}

    col = 0
    for s in range(7):
        for si in range(7):
            for (a, coef_i) in inner_alpha[si]:
                xr_i, xc_i = a // 2, a % 2
                for (r, coef_s) in alpha[s]:
                    ar, ac = r // 2, r % 2
                    gi = ar * 2 + xr_i
                    gk = ac * 2 + xc_i
                    U.setdefault((gi, gk), {})[col] = U.get((gi, gk), {}).get(col, 0) + coef_i * coef_s
            for (b, coef_i) in inner_beta[si]:
                y_row, y_inner_col = b // 2, b % 2
                for (c, coef_s) in beta[s]:
                    ac, bc = c // 2, c % 2
                    gk = ac * 2 + y_row
                    gj = bc * 3 + y_inner_col
                    V.setdefault((gk, gj), {})[col] = V.get((gk, gj), {}).get(col, 0) + coef_i * coef_s
            for (a, b, gamma_s) in recon[s]:
                for (za, zb, gamma_i) in inner_recon[si]:
                    gi = a * 2 + za
                    gj = b * 3 + zb
                    W.setdefault((gj, gi), {})[col] = W.get((gj, gi), {}).get(col, 0) + gamma_s * gamma_i
            col += 1

        for u in range(2):
            for v in range(2):
                for (r, coef_s) in alpha[s]:
                    ar, ac = r // 2, r % 2
                    gi = ar * 2 + u
                    gk = ac * 2 + v
                    U.setdefault((gi, gk), {})[col] = U.get((gi, gk), {}).get(col, 0) + coef_s
                for (c, coef_s) in beta[s]:
                    ac, bc = c // 2, c % 2
                    gk = ac * 2 + v
                    gj = bc * 3 + 2
                    V.setdefault((gk, gj), {})[col] = V.get((gk, gj), {}).get(col, 0) + coef_s
                for (a, b, gamma_s) in recon[s]:
                    gi = a * 2 + u
                    gj = b * 3 + 2
                    W.setdefault((gj, gi), {})[col] = W.get((gj, gi), {}).get(col, 0) + gamma_s
                col += 1

    R = col
    Umat = np.zeros((16, R))
    Vmat = np.zeros((24, R))
    Wmat = np.zeros((24, R))
    for (gi, gk), d in U.items():
        Umat[gi * 4 + gk] = [d.get(c, 0) for c in range(R)]
    for (gk, gj), d in V.items():
        Vmat[gk * 6 + gj] = [d.get(c, 0) for c in range(R)]
    for (gj, gi), d in W.items():
        Wmat[gj * 4 + gi] = [d.get(c, 0) for c in range(R)]
    return Umat, Vmat, Wmat

def _try_reduce_rank(U, V, W, target_rank, deadline):
    """Try to reduce rank by finding linear dependencies."""
    R = U.shape[1]
    if R <= target_rank:
        return U, V, W, R
    
    # Build the tensor T[i,k,j] = sum_r U[i,k,r]*V[k,j,r]*W[j,i,r]
    n, m, p = 4, 4, 6
    T = np.zeros((n, m, p))
    for r in range(R):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    T[i, k, j] += U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    
    # Try to find if we can drop one column by solving for coefficients
    # This is a rank-reduction test: can we express column R-1 as combo of others?
    # Simple approach: try each column, see if removing it still allows exact decomposition
    # with adjusted coefficients (this is hard in general)
    
    # Alternative: use numerical rank of the "flattened" structure
    # For now, just return as-is if we can't easily reduce
    return U, V, W, R

def _verify(U, V, W):
    n, m, p = 4, 4, 6
    R = U.shape[1]
    ok = True
    for i in range(n):
        for k in range(m):
            for j in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i*m + k, r] * V[k*p + j, r] * W[j*n + i, r]
                if abs(s - 1.0) > 1e-6:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
    return ok

def construct():
    deadline = time.time() + 110.0
    U, V, W = _build_base()
    
    if not _verify(U, V, W):
        R = 4 * 4 * 6
        U = np.zeros((16, R))
        V = np.zeros((24, R))
        W = np.zeros((24, R))
        idx = 0
        for i in range(4):
            for k in range(4):
                for j in range(6):
                    U[i*4 + k, idx] = 1
                    V[k*6 + j, idx] = 1
                    W[j*4 + i, idx] = 1
                    idx += 1
    
    # Round to integers where possible
    def to_list(M):
        out = []
        for row in M:
            r = []
            for x in row:
                rx = round(x)
                r.append(int(rx) if abs(x - rx) < 1e-9 else float(x))
            out.append(r)
        return out
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
