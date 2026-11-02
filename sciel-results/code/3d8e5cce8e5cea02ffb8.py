import numpy as np
import itertools
import random

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target Tensor Shape: (n*m, m*p, p*n)
    # Indexing:
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    
    # Build the target tensor T
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    def is_valid_decomposition(U, V, W, T):
        try:
            decomposed = np.einsum('ar,br,cr->abc', U, V, W)
            error = np.linalg.norm(decomposed - T)
            return error, decomposed
        except Exception:
            return float('inf'), None

    def snap_to_half(x):
        return np.round(x * 2.0) / 2.0

    def discrete_refine(U, V, W, T, max_iters=20):
        # Local search on half-integer grid
        # Try to reduce error by tweaking entries
        best_error, _ = is_valid_decomposition(U, V, W, T)
        
        for _ in range(max_iters):
            improved = False
            # Try perturbing U
            for r in range(U.shape[1]):
                for a in range(U.shape[0]):
                    orig = U[a, r]
                    # Try neighbors: orig +/- 0.5
                    for delta in [-0.5, 0.5]:
                        new_val = orig + delta
                        # Keep it simple: just try the two nearest half-integers not equal to current
                        U[a, r] = snap_to_half(new_val)
                        err, _ = is_valid_decomposition(U, V, W, T)
                        if err < best_error - 1e-10:
                            best_error = err
                            improved = True
                        else:
                            U[a, r] = orig
            if not improved:
                # Try perturbing V
                for r in range(V.shape[1]):
                    for b in range(V.shape[0]):
                        orig = V[b, r]
                        for delta in [-0.5, 0.5]:
                            new_val = orig + delta
                            V[b, r] = snap_to_half(new_val)
                            err, _ = is_valid_decomposition(U, V, W, T)
                            if err < best_error - 1e-10:
                                best_error = err
                                improved = True
                            else:
                                V[b, r] = orig
            if not improved:
                # Try perturbing W
                for r in range(W.shape[1]):
                    for c in range(W.shape[0]):
                        orig = W[c, r]
                        for delta in [-0.5, 0.5]:
                            new_val = orig + delta
                            W[c, r] = snap_to_half(new_val)
                            err, _ = is_valid_decomposition(U, V, W, T)
                            if err < best_error - 1e-10:
                                best_error = err
                                improved = True
                            else:
                                W[c, r] = orig
            if not improved:
                break
        return U, V, W, best_error

    # Strategy: Try ranks from lower bound up to 50.
    # Use SVD initialization for each rank.
    
    # Matricizations for SVD initialization
    # T is (nm, mp, pn)
    # Mat1: (nm, mp*pn)
    T_mat1 = T.reshape(n * m, m * p * p * n)
    
    # We will try ranks from 15 to 45
    for R in range(15, 46):
        # Multiple attempts per rank
        for attempt in range(10):
            # 1. SVD Initialization
            # Compute SVD of T_mat1
            U_svd, s, Vt_svd = np.linalg.svd(T_mat1, full_matrices=False)
            
            # Take top R components
            U_svd_r = U_svd[:, :R]
            s_r = s[:R]
            Vt_svd_r = Vt_svd[:R, :]
            
            # U_svd_r is (nm, R)
            # Vt_svd_r is (R, mp*pn)
            
            # We need to split Vt_svd_r into V (mp, R) and W (pn, R)
            # Vt_svd_r[r, :] is a vector of length mp*pn.
            # Reshape to (mp, pn)
            VW_mat = Vt_svd_r.T.reshape(R, m * p, p * n) # (R, mp, pn)
            
            # For each r, SVD the (mp, pn) matrix
            U_init = np.zeros((n * m, R))
            V_init = np.zeros((m * p, R))
            W_init = np.zeros((p * n, R))
            
            for r in range(R):
                # Scale U
                U_init[:, r] = U_svd_r[:, r] * np.sqrt(s_r[r])
                
                # SVD of Vw_mat[r]
                Uvw, sw, Vvw = np.linalg.svd(VW_mat[r], full_matrices=False)
                # Uvw is (mp, min(mp,pn)), Vvw is (min(mp,pn), pn)
                # We want V[:, r] and W[:, r] such that V[:, r] W[:, r].T approx VW_mat[r]
                # Actually, we want outer product.
                # Let V[:, r] = Uvw[:, 0] * sw[0] and W[:, r] = Vvw[0, :] ?
                # This gives rank 1 approximation.
                # To get better, we can just use the first component if we are constrained to rank 1 per slice?
                # No, the tensor decomposition allows full rank in the factor matrices.
                # The SVD of the matricization gives a rank R approximation.
                # The split into V and W is not unique.
                # A simple split:
                V_init[:, r] = Uvw[:, 0] * np.sqrt(sw[0])
                W_init[:, r] = Vvw[0, :] * np.sqrt(sw[0])
            
            # Now we have U_init, V_init, W_init (real values)
            # Scale them to have unit norm columns to help snapping?
            # Or just snap directly.
            
            U_snap = snap_to_half(U_init)
            V_snap = snap_to_half(V_init)
            W_snap = snap_to_half(W_init)
            
            error, _ = is_valid_decomposition(U_snap, V_snap, W_snap, T)
            
            if error < 1e-8:
                return {
                    "U": U_snap.tolist(),
                    "V": V_snap.tolist(),
                    "W": W_snap.tolist()
                }
            
            # If direct snap fails, try discrete refinement
            U_ref, V_ref, W_ref, err_ref = discrete_refine(U_snap, V_snap, W_snap, T, max_iters=5)
            
            if err_ref < 1e-8:
                return {
                    "U": U_ref.tolist(),
                    "V": V_ref.tolist(),
                    "W": W_ref.tolist()
                }

    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
