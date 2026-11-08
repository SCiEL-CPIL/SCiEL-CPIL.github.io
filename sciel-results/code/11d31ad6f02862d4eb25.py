"""Search for rank-minimal exact tensor decomposition of 2x3 @ 3x8 matrix multiply."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 110  # safety margin under 120s

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Build target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Schoolbook decomposition as fallback
    def get_schoolbook():
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
        return U, V, W, rank

    U_school, V_school, W_school, rank_school = get_schoolbook()
    best_U, best_V, best_W, best_rank = U_school, V_school, W_school, rank_school

    def verify_exact(U, V, W, R):
        """Check if U, V, W (as half-integers) exactly decompose T using Fraction."""
        # U, V, W are numpy arrays with values that should be multiples of 0.5
        # Convert to fractions for exact check
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(R):
                        ui = Fraction(int(round(U[i * m + j, c] * 2)), 2)
                        vj = Fraction(int(round(V[j * p + k, c] * 2)), 2)
                        wk = Fraction(int(round(W[k * n + i, c] * 2)), 2)
                        s += ui * vj * wk
                    if s != 1:
                        return False
        return True

    def snap_to_half_integers(U, V, W):
        """Snap to nearest half-integers"""
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        return U_snap, V_snap, W_snap

    def als_solve(R, seed, max_iters=200):
        """Run ALS for rank R decomposition."""
        np.random.seed(seed)
        U = np.random.randn(nm, R) * 0.5
        V = np.random.randn(mp, R) * 0.5
        W = np.random.randn(pn, R) * 0.5

        S = np.zeros_like(T)
        for r in range(R):
            S += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])

        prev_error = float('inf')
        for it in range(max_iters):
            if time.time() > deadline:
                break

            # Update U
            for r in range(R):
                S -= np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                vw_mat = np.outer(V[:, r], W[:, r])
                numerator = np.einsum('ijk,jk->i', T - S, vw_mat)
                denominator = np.sum(vw_mat ** 2)
                if denominator > 1e-10:
                    U[:, r] = numerator / denominator
                S += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])

            # Update V
            for r in range(R):
                S -= np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                uw_mat = np.outer(U[:, r], W[:, r])
                numerator = np.einsum('ijk,ik->j', T - S, uw_mat)
                denominator = np.sum(uw_mat ** 2)
                if denominator > 1e-10:
                    V[:, r] = numerator / denominator
                S += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])

            # Update W
            for r in range(R):
                S -= np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                uv_mat = np.outer(U[:, r], V[:, r])
                numerator = np.einsum('ijk,ij->k', T - S, uv_mat)
                denominator = np.sum(uv_mat ** 2)
                if denominator > 1e-10:
                    W[:, r] = numerator / denominator
                S += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])

            error = np.linalg.norm(T - S)
            if error < 1e-10:
                break
            if abs(prev_error - error) < 1e-14:
                break
            prev_error = error

        return U, V, W

    # Try ranks from a lower bound up to schoolbook
    # For matrix multiplication, known results suggest rank is often close to n*m*p but can be lower
    # Start from a reasonable lower bound
    min_rank = 20  # conservative lower bound to try

    for R in range(min_rank, rank_school):
        if time.time() > deadline:
            break

        # Try multiple random initializations
        found = False
        for init_idx in range(15):
            if time.time() > deadline:
                break

            U, V, W = als_solve(R, seed=init_idx * 1000 + R * 7)

            # Snap to half-integers
            U_snap, V_snap, W_snap = snap_to_half_integers(U, V, W)

            # Quick numerical check first
            S_snap = np.zeros_like(T)
            for r in range(R):
                S_snap += np.einsum('a,b,c->abc', U_snap[:, r], V_snap[:, r], W_snap[:, r])
            snap_error = np.linalg.norm(T - S_snap)

            if snap_error < 1e-8:
                # Exact verification with Fractions
                if verify_exact(U_snap, V_snap, W_snap, R):
                    best_U, best_V, best_W, best_rank = U_snap, V_snap, W_snap, R
                    found = True
                    break

        if found:
            break

    # Convert to list of lists for interface
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
