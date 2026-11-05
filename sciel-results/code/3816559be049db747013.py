import numpy as np
import time
from itertools import product, combinations

DIMENSIONS = [2, 4, 7]

# Precompute the target tensor
_T = np.zeros((8, 28, 14))
for _i in range(2):
    for _j in range(4):
        for _k in range(7):
            _T[_i * 4 + _j, _j * 7 + _k, _k * 2 + _i] = 1.0
_T1 = _T.reshape(8, -1)
_T2 = _T.reshape(28, -1)
_T3 = _T.reshape(14, -1)

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130
    T = _T
    T1, T2, T3 = _T1, _T2, _T3

    def verify_decomp(U, V, W, atol=1e-8):
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(recon, T, atol=atol)

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def get_rank50():
        rank = 50
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0

        def add_strassen(i0, j0, k0):
            nonlocal col
            U[i0 * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            U[(i0 + 1) * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] -= 1
            col += 1

            U[i0 * m + j0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + k0, col] += 1
            V[j0 * p + k0, col] -= 1
            W[k0 * n + i0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            col += 1

            U[i0 * m + j0, col] += 1
            U[i0 * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            col += 1

            U[(i0 + 1) * m + j0, col] += 1
            U[i0 * m + j0, col] -= 1
            V[j0 * p + k0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            U[i0 * m + (j0 + 1), col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] -= 1
            V[(j0 + 1) * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            col += 1

        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                add_strassen(0, j0, k0)

        for j in range(m):
            U[0 * m + j, col] = 1
            V[j * p + 6, col] = 1
            W[6 * n + 0, col] = 1
            col += 1
            U[1 * m + j, col] = 1
            V[j * p + 6, col] = 1
            W[6 * n + 1, col] = 1
            col += 1

        assert col == rank, f"Expected rank 50, got {col}"
        return U, V, W

    # Get the rank-50 decomposition as a starting point
    U50, V50, W50 = get_rank50()
    if not verify_decomp(U50, V50, W50):
        U, V, W = get_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_rank = 50
    best_U, best_V, best_W = U50.copy(), V50.copy(), W50.copy()

    def khatri_rao(A, B):
        M, R = A.shape
        N, _ = B.shape
        KR = np.zeros((M * N, R))
        for r in range(R):
            KR[:, r] = np.kron(A[:, r], B[:, r])
        return KR

    def polish(U, V, W, max_iters=200, tol=1e-13):
        """High-precision polish via ALS with tight tolerance."""
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            try:
                KR_VW = khatri_rao(V, W)
                U_new, *_ = np.linalg.lstsq(KR_VW, T1.T, rcond=None)
                U_new = U_new.T
            except:
                break
            try:
                KR_WU = khatri_rao(W, U_new)
                V_new, *_ = np.linalg.lstsq(KR_WU, T2.T, rcond=None)
                V_new = V_new.T
            except:
                break
            try:
                KR_UV = khatri_rao(U_new, V_new)
                W_new, *_ = np.linalg.lstsq(KR_UV, T3.T, rcond=None)
                W_new = W_new.T
            except:
                break
            
            change = np.linalg.norm(U_new - U) + np.linalg.norm(V_new - V) + np.linalg.norm(W_new - W)
            U, V, W = U_new, V_new, W_new
            if change < tol:
                break
        return U, V, W

    def try_reduce(U, V, W, cols_to_keep):
        """
        Try to find a decomposition using only the specified columns.
        Fix U and V to the given columns, solve for W, then iterate.
        """
        U_sub = U[:, cols_to_keep].copy()
        V_sub = V[:, cols_to_keep].copy()
        
        # Initialize W by solving least squares
        try:
            KR_UV = khatri_rao(U_sub, V_sub)
            W_sub, *_ = np.linalg.lstsq(KR_UV, T3.T, rcond=None)
            W_sub = W_sub.T
        except:
            return None, None, None, None
        
        # Iterate ALS
        for it in range(100):
            if time.time() > deadline - 5:
                return None, None, None, None
            try:
                KR_VW = khatri_rao(V_sub, W_sub)
                U_sub_new, *_ = np.linalg.lstsq(KR_VW, T1.T, rcond=None)
                U_sub_new = U_sub_new.T
            except:
                return None, None, None, None
            try:
                KR_WU = khatri_rao(W_sub, U_sub_new)
                V_sub_new, *_ = np.linalg.lstsq(KR_WU, T2.T, rcond=None)
                V_sub_new = V_sub_new.T
            except:
                return None, None, None, None
            try:
                KR_UV = khatri_rao(U_sub_new, V_sub_new)
                W_sub_new, *_ = np.linalg.lstsq(KR_UV, T3.T, rcond=None)
                W_sub_new = W_sub_new.T
            except:
                return None, None, None, None
            
            change = (np.linalg.norm(U_sub_new - U_sub) + 
                      np.linalg.norm(V_sub_new - V_sub) + 
                      np.linalg.norm(W_sub_new - W_sub))
            U_sub, V_sub, W_sub = U_sub_new, V_sub_new, W_sub_new
            if change < 1e-12:
                break
        
        recon = np.einsum('ar,br,cr->abc', U_sub, V_sub, W_sub)
        residual = np.linalg.norm(recon - T)
        return U_sub, V_sub, W_sub, residual

    # Strategy: try removing 1, 2, 3 columns from rank-50
    # For each target rank, try all subsets of columns to keep
    
    # First, try rank 49: remove 1 column
    for target_rank in range(49, 44, -1):
        if time.time() > deadline - 10:
            break
        
        cols_to_remove = 50 - target_rank
        
        # Try all combinations of columns to remove
        found = False
        for remove_combo in combinations(range(50), cols_to_remove):
            if time.time() > deadline - 5:
                break
            keep_cols = [i for i in range(50) if i not in remove_combo]
            
            U_sub, V_sub, W_sub, residual = try_reduce(U50, V50, W50, keep_cols)
            
            if U_sub is not None and residual < 1e-8:
                # Polish for higher precision
                U_sub, V_sub, W_sub = polish(U_sub, V_sub, W_sub, max_iters=300, tol=1e-14)
                if verify_decomp(U_sub, V_sub, W_sub, atol=1e-8):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_sub, V_sub, W_sub
                    found = True
                    break
        
        if found:
            continue  # Try next lower rank
        else:
            break  # Can't achieve this rank, stop

    # If we found a better rank, try to push further with more iterations
    if best_rank < 50 and time.time() < deadline - 15:
        for target_rank in range(best_rank - 1, max(40, best_rank - 5), -1):
            if time.time() > deadline - 5:
                break
            cols_to_remove = 50 - target_rank
            found = False
            for remove_combo in combinations(range(50), cols_to_remove):
                if time.time() > deadline - 3:
                    break
                keep_cols = [i for i in range(50) if i not in remove_combo]
                U_sub, V_sub, W_sub, residual = try_reduce(U50, V50, W50, keep_cols)
                if U_sub is not None and residual < 1e-8:
                    U_sub, V_sub, W_sub = polish(U_sub, V_sub, W_sub, max_iters=200, tol=1e-14)
                    if verify_decomp(U_sub, V_sub, W_sub, atol=1e-8):
                        best_rank = target_rank
                        best_U, best_V, best_W = U_sub, V_sub, W_sub
                        found = True
                        break
            if not found:
                break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
