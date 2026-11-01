"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 138  # safety margin before the 150 s wall-clock limit
    rng = random.Random(20240517)

    n, m, p = DIMENSIONS  # n=2, m=2, p=8
    nm, mp, pn = n * m, m * p, p * n

    # Build the target tensor T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i, else 0.
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Precompute the three unfoldings (matricizations) of T.
    # T_U[b, a, c] = T[a, b, c]  -> shape (mp, nm, pn)
    # T_V[a, b, c] = T[a, b, c]  -> shape (nm, mp, pn)
    # T_W[a, b, c] = T[a, b, c]  -> shape (nm, mp, pn)
    T_U = np.transpose(T, (1, 0, 2))  # (mp, nm, pn)
    T_V = np.transpose(T, (0, 1, 2))  # (nm, mp, pn)
    T_W = np.transpose(T, (0, 1, 2))  # (nm, mp, pn)

    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank), dtype=np.float64)
        V = np.zeros((mp, rank), dtype=np.float64)
        W = np.zeros((pn, rank), dtype=np.float64)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1.0
                    V[j * p + k, c] = 1.0
                    W[k * n + i, c] = 1.0
                    c += 1
        return U, V, W

    def verify_exact(U, V, W, tol=1e-7):
        rank = U.shape[1]
        T_decomp = np.zeros((nm, mp, pn), dtype=np.float64)
        for r in range(rank):
            T_decomp += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.max(np.abs(T_decomp - T)) < tol

    def objective(x, rank):
        U = x[0: nm * rank].reshape(nm, rank)
        V = x[nm * rank: nm * rank + mp * rank].reshape(mp, rank)
        W = x[nm * rank + mp * rank: nm * rank + mp * rank + pn * rank].reshape(pn, rank)
        # Residual via unfoldings: for each r, subtract U[:,r] V[:,r]^T from T_U
        # R_U[b,a,c] = T_U[b,a,c] - sum_r U[a,r] V[b,r] W[c,r]
        # Use matricization: R_U as (mp, nm*pn) = T_U.reshape(mp, nm*pn) - V * (U^T * W)
        # (U^T * W) is (rank, nm*pn) with (r, a*pn+c) = U[a,r]*W[c,r]
        UW = (U.T * W).reshape(rank, nm * pn)  # (rank, nm*pn)
        R_U = T_U.reshape(mp, nm * pn) - V @ UW
        return float(np.sum(R_U * R_U))

    def solve_for_W(U, V, rank):
        # Solve for W minimizing ||T_W - (U^T * W) V^T||
        # T_W[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # For each c: T_W[:,:,c] (nm*mp) = sum_r W[c,r] * (U[:,r] V[:,r]^T)
        # So T_W.reshape(nm*mp, pn) = (U^T * V) @ W^T  where (U^T*V) is (nm*mp, rank)
        UV = (U.T * V).reshape(nm * mp, rank)  # (nm*mp, rank)
        TW = T_W.reshape(nm * mp, pn)  # (nm*mp, pn)
        # TW = UV @ W^T  ->  W^T = UV^+ TW  ->  W = (UV^+ TW)^T
        W = np.linalg.lstsq(UV, TW, rcond=None)[0].T
        return W

    def solve_for_V(U, W, rank):
        # T_V[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # T_V.reshape(mp, nm*pn) = V @ (U^T * W)
        UW = (U.T * W).reshape(rank, nm * pn)  # (rank, nm*pn)
        TV = T_V.reshape(mp, nm * pn)  # (mp, nm*pn)
        V = np.linalg.lstsq(UW.T, TV.T, rcond=None)[0].T
        return V

    def solve_for_U(V, W, rank):
        # T_U[b,a,c] = sum_r U[a,r] V[b,r] W[c,r]
        # T_U.reshape(nm, mp*pn) = U @ (V^T * W)
        VW = (V.T * W).reshape(rank, mp * pn)  # (rank, mp*pn)
        TU = T_U.reshape(nm, mp * pn)  # (nm, mp*pn)
        U = np.linalg.lstsq(VW.T, TU.T, rcond=None)[0].T
        return U

    def refine(U, V, W, rank, max_iters=200):
        # Alternating least squares (ALS) to drive residual to zero.
        for it in range(max_iters):
            if time.time() > deadline:
                break
            W_new = solve_for_W(U, V, rank)
            V_new = solve_for_V(U, W_new, rank)
            U_new = solve_for_V if False else solve_for_U(V_new, W_new, rank)
            U, V, W = U_new, V_new, W_new
            # Check convergence
            T_decomp = np.zeros((nm, mp, pn), dtype=np.float64)
            for r in range(rank):
                T_decomp += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            if np.max(np.abs(T_decomp - T)) < 1e-9:
                break
        return U, V, W

    # Schoolbook as guaranteed baseline.
    best_U, best_V, best_W = schoolbook()
    best_rank = n * m * p  # 32

    # Try to find lower-rank decompositions via ALS + random restarts.
    for target_rank in range(20, best_rank):
        if time.time() > deadline:
            break
        found = False
        max_restarts = 8
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            # Random initialization
            scale = 1.0
            U = rng.uniform(-scale, scale, (nm, target_rank))
            V = rng.uniform(-scale, scale, (mp, target_rank))
            W = rng.uniform(-scale, scale, (pn, target_rank))
            # Refine with ALS
            U, V, W = refine(U, V, W, target_rank, max_iters=300)
            if verify_exact(U, V, W, tol=1e-7):
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                best_rank = target_rank
                found = True
                break
        if found:
            # Try to go even lower
            continue

    # Convert to lists (integer/half-integer if close).
    def to_list(A):
        out = []
        for row in A:
            r = []
            for v in row:
                # Round to nearest half-integer if close
                half = round(2 * v) / 2.0
                if abs(v - half) < 1e-6:
                    r.append(half)
                else:
                    r.append(v)
            out.append(r)
        return out

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
