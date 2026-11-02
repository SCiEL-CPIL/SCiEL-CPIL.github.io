"""Search for rank-minimal exact tensor decomposition of 2x3 x 3x9 multiplication."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 9]

DEADLINE = 120.0

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + DEADLINE - 10.0  # safety margin

    # Build the multiplication tensor T[i,j,k] -> W index (k*n+i), U index (i*m+j), V index (j*p+k)
    # T[U_idx, V_idx, W_idx] = 1 if U=(i,j), V=(j,k), W=(k,i) consistent
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    def schoolbook():
        rank = n*m*p
        U = np.zeros((n*m, rank)); V = np.zeros((m*p, rank)); W = np.zeros((p*n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1; V[j*p+k, c] = 1; W[k*n+i, c] = 1
                    c += 1
        return U, V, W

    def verify(U, V, W, tol=1e-8):
        # reconstruct tensor
        r = U.shape[1]
        recon = np.einsum('ac,bc,dc->abd', U, V, W)
        return np.max(np.abs(recon - T)) < tol

    # Start from schoolbook and try to reduce rank via random low-rank factorization search.
    best_U, best_V, best_W = schoolbook()
    best_rank = best_U.shape[1]

    # Try progressively lower ranks with random init + ALS-style refinement
    for rank in range(best_rank - 1, 30, -1):
        if time.time() > deadline:
            break
        found = False
        for trial in range(40):
            if time.time() > deadline:
                break
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            # alternating least squares to fit T
            for it in range(300):
                if time.time() > deadline:
                    break
                # update U: for each a, U[a,:] solves sum_b T[a,b,:] * W^T[b,:] = U[a,:] V^T
                # V^T is (rank, m*p); form B[a, c] = sum_b T[a,b,:] W[b,c]^T  -> (m*p, rank)
                # solve U[a] V^T = B[a]  =>  U[a] = B[a] (V^T)^-1  => use lstsq
                B = np.einsum('abc,dc->abd', T, W)  # (n*m, m*p, rank)
                VT = V.T  # (rank, m*p)
                for a in range(n*m):
                    U[a] = np.linalg.lstsq(VT, B[a], rcond=None)[0]
                # update V
                B = np.einsum('abc,ac->bdc', T, U)  # (m*p, p*n, rank)
                UT = U.T
                for b in range(m*p):
                    V[b] = np.linalg.lstsq(UT, B[b], rcond=None)[0]
                # update W
                B = np.einsum('abc,bc->adc', T, V)  # (n*m, p*n, rank)
                VT2 = V.T
                for d in range(p*n):
                    W[d] = np.linalg.lstsq(U.T, B[:, d, :], rcond=None)[0]
                if verify(U, V, W, tol=1e-7):
                    found = True
                    break
            if found:
                break
        if found:
            best_U, best_V, best_W = U, V, W
            best_rank = rank

    # Convert to fractions (round to nearest half-integer / rational)
    def to_frac_matrix(M):
        F = np.zeros(M.shape, dtype=object)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                F[i, j] = Fraction(M[i, j]).limit_denominator(1000)
        return F

    Uf = to_frac_matrix(best_U)
    Vf = to_frac_matrix(best_V)
    Wf = to_frac_matrix(best_W)

    # exact verify with fractions
    def exact_verify(Uf, Vf, Wf):
        r = Uf.shape[1]
        for i in range(n*m):
            for j in range(m*p):
                for k in range(p*n):
                    s = Fraction(0)
                    for c in range(r):
                        s += Uf[i, c] * Vf[j, c] * Wf[k, c]
                    if abs(s - T[i, j, k]) > 1e-6:
                        return False
        return True

    if not exact_verify(Uf, Vf, Wf):
        # fallback to schoolbook
        Uf, Vf, Wf = to_frac_matrix(best_U), to_frac_matrix(best_V), to_frac_matrix(best_W)
        if not exact_verify(Uf, Vf, Wf):
            sU, sV, sW = schoolbook()
            Uf, Vf, Wf = to_frac_matrix(sU), to_frac_matrix(sV), to_frac_matrix(sW)

    return {"U": Uf.tolist(), "V": Vf.tolist(), "W": Wf.tolist()}
# EVOLVE-BLOCK-END
