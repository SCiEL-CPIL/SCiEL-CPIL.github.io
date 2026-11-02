"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying
    a 3x4 matrix by a 4x7 matrix.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the target tensor
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook decomposition
    schoolbook_rank = n * m * p
    U_sb = np.zeros((nm, schoolbook_rank))
    V_sb = np.zeros((mp, schoolbook_rank))
    W_sb = np.zeros((pn, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_sb[a, col] = 1.0
                V_sb[b, col] = 1.0
                W_sb[c, col] = 1.0
                col += 1
    
    # Verify schoolbook
    def verify(U, V, W, T, tol=1e-10):
        R = U.shape[1]
        for r in range(R):
            T_check = np.einsum('ar,br->ab', U[:, r], V[:, r])
            # Actually, we need: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
            # So: for each c, T[:,:,c] = sum_r W[c,r] * (U[:,r] * V[:,r])
            pass
        # Better verification
        reconstructed = np.einsum('ar,br,cr->abc', U, V, W)
        error = np.max(np.abs(reconstructed - T))
        return error < tol
    
    # Try to find a lower rank decomposition
    # Strategy: try ranks from a lower bound up to schoolbook_rank
    # Lower bound: max of matrix ranks of unfoldings
    
    # Compute rank lower bounds from unfoldings
    T_mat1 = T.reshape(nm * mp, pn)
    T_mat2 = T.reshape(nm, mp * pn)
    T_mat3 = T.reshape(mp * pn, nm)
    
    rank_lb1 = np.linalg.matrix_rank(T_mat1, tol=1e-10)
    rank_lb2 = np.linalg.matrix_rank(T_mat2, tol=1e-10)
    rank_lb3 = np.linalg.matrix_rank(T_mat3, tol=1e-10)
    
    rank_lower_bound = max(rank_lb1, rank_lb2, rank_lb3)
    
    # Try to find a valid decomposition at each rank from lower bound to schoolbook
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = schoolbook_rank
    
    # Try ranks from lower bound up to schoolbook_rank
    for rank in range(rank_lower_bound, schoolbook_rank + 1):
        if time.time() > deadline:
            break
        
        if rank >= best_rank:
            break
        
        # Try to find a decomposition at this rank
        # Use SVD on the first unfolding
        try:
            U_svd, s_vals, Vt_svd = np.linalg.svd(T_mat1, full_matrices=False)
            
            # Truncate to desired rank
            U_trunc = U_svd[:, :rank] * s_vals[:rank]
            Vt_trunc = Vt_svd[:rank, :]
            
            U_3d = U_trunc.reshape(nm, rank)
            V_3d = Vt_trunc.reshape(mp, rank)
            
            # Build M: M[r] = (U[:,r] * V[:,r]).flatten()
            M = np.zeros((nm * mp, rank))
            for r in range(rank):
                M[:, r] = (U_3d[:, r] * V_3d[:, r]).flatten()
            
            # Solve for W
            W_3d = np.zeros((pn, rank))
            for c in range(pn):
                target = T[:, :, c].flatten()
                w, residuals, rank_ls, sv = np.linalg.lstsq(M, target, rcond=None)
                W_3d[c, :] = w
            
            # Verify
            if verify(U_3d, V_3d, W_3d, T):
                # Try to round to half-integers
                U_rounded = np.round(U_3d * 2) / 2
                V_rounded = np.round(V_3d * 2) / 2
                W_rounded = np.round(W_3d * 2) / 2
                
                if verify(U_rounded, V_rounded, W_rounded, T):
                    best_result = {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }
                    best_rank = rank
                    break
                else:
                    # Use unrounded values
                    best_result = {
                        "U": U_3d.tolist(),
                        "V": V_3d.tolist(),
                        "W": W_3d.tolist()
                    }
                    best_rank = rank
                    break
        except Exception:
            continue
    
    return best_result
# EVOLVE-BLOCK-END
