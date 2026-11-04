"""
Rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication.

Strategy:
  - Build a rank-84 baseline using three independent 4x4x2 subproblems,
    each decomposed via 4 rank-7 Strassen 2x2*2x2 products.
  - Then attempt rank reduction via Gauss-Newton / Levenberg-Marquardt
    to find exact lower-rank decompositions.
  - Return the lowest-rank valid decomposition found.
"""
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [4, 4, 6]

# Standard 2x2*2x2 Strassen decomposition (rank 7)
_STRassen_U_222 = np.array([
    [0, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 0],
], dtype=np.float64)

_STRassen_V_222 = np.array([
    [1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0],
], dtype=np.float64)

_STRassen_W_222 = np.array([
    [1, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0],
], dtype=np.float64)


def _build_rank84(n, m, p):
    """
    Build rank-84 decomposition for 4x4 * 4x6.
    
    Split the 6 columns of B into three groups of 2.
    For each 4x4 * 4x2 subproblem, use 4 independent 2x2*2x2 Strassen
    decompositions (one per 2x2 block of the 4x4 A and 4x2 C).
    Each 2x2*2x2 uses rank 7, so each 4x4x2 uses 4*7=28, total 3*28=84.
    """
    total_rank = 84
    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)
    
    # For each of the 3 column groups (k=0,1,2 corresponding to cols 0-1, 2-3, 4-5)
    for k in range(3):
        # For each of the 4 2x2 blocks in the 4x4 A matrix
        # and the corresponding 2x2 block in the 4x2 C output
        for a_row_block in range(2):  # rows 0-1 or 2-3 of A
            for a_col_block in range(2):  # cols 0-1 or 2-3 of A
                # This 2x2 block of A multiplies a 2x2 block of B
                # to produce a 2x2 block of C
                a_row_start = a_row_block * 2
                a_col_start = a_col_block * 2
                b_col_start = k * 2  # columns k*2, k*2+1 of B
                c_row_start = a_row_block * 2  # rows in C
                c_col_start = k * 2  # cols in C (within the 4x6, but C is 4x6)
                
                # For Strassen 2x2*2x2, we need:
                # A_block (2x2), B_block (2x2), C_block (2x2)
                # A_block is at rows [a_row_start, a_row_start+1], cols [a_col_start, a_col_start+1] of A
                # B_block is at rows [a_col_start, a_col_start+1], cols [b_col_start, b_col_start+1] of B
                # C_block is at rows [c_row_start, c_row_start+1], cols [c_col_start, c_col_start+1] of C
                
                for t in range(7):
                    col = k * 28 + (a_row_block * 2 + a_col_block) * 7 + t
                    
                    # U: A_block entries
                    for i in range(2):
                        for j in range(2):
                            idx = _STRassen_U_222[i, j]
                            if idx != 0:
                                # A element at (a_row_start+i, a_col_start+j)
                                # In U, row index is (i_A * m + j_A) where i_A, j_A are 0-3
                                u_row = (a_row_start + i) * m + (a_col_start + j)
                                U[u_row, col] += idx
                    
                    # V: B_block entries
                    for i in range(2):
                        for j in range(2):
                            idx = _STRassen_V_222[i, j]
                            if idx != 0:
                                # B element at (a_col_start+i, b_col_start+j)
                                # In V, row index is (i_B * p + j_B) where i_B, j_B are 0-3, 0-5
                                v_row = (a_col_start + i) * p + (b_col_start + j)
                                V[v_row, col] += idx
                    
                    # W: C_block entries
                    for i in range(2):
                        for j in range(2):
                            idx = _STRassen_W_222[i, j]
                            if idx != 0:
                                # C element at (c_row_start+i, c_col_start+j)
                                # In W, row index is (k_C * n + i_C) where k_C, i_C are 0-5, 0-3
                                w_row = (c_col_start + j) * n + (c_row_start + i)
                                W[w_row, col] += idx
    
    return U, V, W


def _build_target_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _tensor_residual(params, T, n, m, p, rank):
    """Compute residual of tensor decomposition from flattened params."""
    nm, mp, pn = n * m, m * p, p * n
    U = params[:nm * rank].reshape(nm, rank)
    V = params[nm * rank: nm * rank + mp * rank].reshape(mp, rank)
    W = params[nm * rank + mp * rank:].reshape(pn, rank)
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    return (T - recon).ravel()


def _try_reduce_rank(U0, V0, W0, T, n, m, p, target_rank, max_iter=200, deadline=None):
    """Try to find a rank-target_rank decomposition using least_squares."""
    nm, mp, pn = n * m, m * p, p * n
    # Initialize: take first target_rank columns
    U_init = U0[:, :target_rank].copy()
    V_init = V0[:, :target_rank].copy()
    W_init = W0[:, :target_rank].copy()
    x0 = np.concatenate([U_init.ravel(), V_init.ravel(), W_init.ravel()])

    def residual(x):
        return _tensor_residual(x, T, n, m, p, target_rank)

    try:
        result = least_squares(
            residual, x0, method='lm', max_nfev=max_iter,
            xtol=1e-14, ftol=1e-14, gtol=1e-14
        )
    except Exception:
        return None

    if deadline is not None and time.time() > deadline:
        return None

    final_res = result.fun
    err = np.sum(final_res ** 2)
    if err < 1e-10:
        x = result.x
        U = x[:nm * target_rank].reshape(nm, target_rank)
        V = x[nm * target_rank: nm * target_rank + mp * target_rank].reshape(mp, target_rank)
        W = x[nm * target_rank + mp * target_rank:].reshape(pn, target_rank)
        # Verify
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        if np.allclose(recon, T, atol=1e-9):
            return U, V, W
    return None


def construct():
    t0 = time.time()
    deadline = 140.0  # safety margin before 150s

    n, m, p = DIMENSIONS  # 4, 4, 6
    T = _build_target_tensor(n, m, p)

    # Build rank-84 baseline using three 4x4x2 subproblems
    U84, V84, W84 = _build_rank84(n, m, p)

    # Verify baseline
    recon = np.einsum('ar,br,cr->abc', U84, V84, W84)
    if not np.allclose(recon, T, atol=1e-9):
        # Fallback to schoolbook
        rank = n * m * p
        U_s = np.zeros((n * m, rank), dtype=np.float64)
        V_s = np.zeros((m * p, rank), dtype=np.float64)
        W_s = np.zeros((p * n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j][col] = 1
                    V_s[j * p + k][col] = 1
                    W_s[k * n + i][col] = 1
                    col += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

    best_U, best_V, best_W = U84, V84, W84
    best_rank = 84

    # Try to reduce rank from 83 down to 70
    for target_rank in range(83, 69, -1):
        if time.time() > deadline - 15:
            break

        # Try multiple random initializations
        found = False
        for attempt in range(3):
            if time.time() > deadline - 15:
                break

            if attempt == 0:
                # Use first target_rank columns
                U_init = U84[:, :target_rank].copy()
                V_init = V84[:, :target_rank].copy()
                W_init = W84[:, :target_rank].copy()
            else:
                # Random perturbation of the full 84-col solution, then truncate
                rng = np.random.RandomState(42 + attempt)
                U_init = U84[:, :target_rank] + rng.randn(n * m, target_rank) * 0.01
                V_init = V84[:, :target_rank] + rng.randn(m * p, target_rank) * 0.01
                W_init = W84[:, :target_rank] + rng.randn(p * n, target_rank) * 0.01

            result = _try_reduce_rank(U_init, V_init, W_init, T, n, m, p, target_rank,
                                       max_iter=100, deadline=deadline - 15)
            if result is not None:
                U_r, V_r, W_r = result
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = target_rank
                found = True
                break

    # Final verification
    recon = np.einsum('ar,br,cr->abc', best_U, best_V, best_W)
    if np.allclose(recon, T, atol=1e-9):
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        # Fallback to rank-84
        return {
            "U": U84.tolist(),
            "V": V84.tolist(),
            "W": W84.tolist()
        }
