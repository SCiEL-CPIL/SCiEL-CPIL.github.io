"""Matrix multiplication 2,4,8 - Strassen-based rank-56 decomposition with rank reduction."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Return a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 2x4 matrix by a 4x8 matrix.
    
    Strategy: Build the rank-56 Strassen decomposition, then check for
    linear dependencies among the 56 rank-1 tensors to reduce rank.
    """
    n, m, p = DIMENSIONS  # 2, 4, 8
    nrm = n * m  # 8
    mp = m * p    # 32
    pn = p * n    # 16
    rank = 56
    
    # Build the full rank-56 tensor as a 3D array T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # and also store individual rank-1 tensors for dependency analysis.
    
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    # Store each rank-1 tensor as a flattened vector of length nrm*mp*pn
    # T_r[i,j,k] = U_r[i] * V_r[j] * W_r[k]
    tensors = []  # list of 56 vectors, each of length 8*32*16 = 4096
    
    for ba in range(2):
        for bb in range(4):
            for t in range(7):
                U_loc, V_loc, W_loc = strassen_terms[t]
                # Build U_r, V_r, W_r vectors
                U_r = [0] * nrm
                V_r = [0] * mp
                W_r = [0] * pn
                
                for i_l in range(2):
                    for j_l in range(2):
                        u_idx = i_l * m + (2 * ba + j_l)
                        U_r[u_idx] = U_loc[i_l * 2 + j_l]
                for j_l in range(2):
                    for k_l in range(2):
                        v_idx = (2 * ba + j_l) * p + (2 * bb + k_l)
                        V_r[v_idx] = V_loc[j_l * 2 + k_l]
                for k_l in range(2):
                    for i_l in range(2):
                        w_idx = (2 * bb + k_l) * n + i_l
                        W_r[w_idx] = W_loc[k_l * 2 + i_l]
                
                # Flatten the rank-1 tensor
                vec = np.zeros(nrm * mp * pn)
                for i in range(nrm):
                    if U_r[i] != 0:
                        for j in range(mp):
                            if V_r[j] != 0:
                                for k in range(pn):
                                    if W_r[k] != 0:
                                        vec[i * mp * pn + j * pn + k] = U_r[i] * V_r[j] * W_r[k]
                tensors.append(vec)
    
    # Check for linear dependencies using exact rational arithmetic
    # Build matrix M where column r is tensor r
    # We want to find the rank of the 4096 x 56 matrix
    # Use numpy for speed, then verify with exact arithmetic if needed
    
    M = np.array(tensors, dtype=np.float64).T  # 4096 x 56
    
    # Compute rank via SVD
    try:
        U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
        # Determine rank with tolerance
        tol = max(M.shape) * np.finfo(np.float64).eps * s[0]
        rank_found = np.sum(s > tol)
    except:
        rank_found = 56
    
    # If we found dependencies, try to find a minimal subset
    # For now, just use the full 56 (safe fallback)
    # But let's check if rank < 56
    
    if rank_found < 56:
        # Find which columns are linearly dependent
        # Use the right singular vectors
        # The last (56 - rank_found) columns of Vt span the null space
        # We need to find a basis for the column space
        # Simple approach: greedy selection of linearly independent columns
        selected = []
        selected_vecs = []
        for r in range(56):
            v = M[:, r]
            # Check if v is in span of selected_vecs
            if not selected_vecs:
                selected.append(r)
                selected_vecs.append(v)
            else:
                # Try to express v as linear combination
                A_sel = np.array(selected_vecs, dtype=np.float64).T  # 4096 x len(selected)
                # Solve least squares
                try:
                    coeffs, residuals, rank_sel, sv = np.linalg.lstsq(A_sel, v, rcond=None)
                    # Check residual
                    if len(residuals) > 0 and residuals[0] < 1e-9 * np.linalg.norm(v):
                        # Linearly dependent, skip
                        continue
                    else:
                        selected.append(r)
                        selected_vecs.append(v)
                except:
                    selected.append(r)
                    selected_vecs.append(v)
        
        if len(selected) < 56:
            # Reconstruct using only selected columns
            rank_new = len(selected)
            U = [[0] * rank_new for _ in range(nrm)]
            V = [[0] * rank_new for _ in range(mp)]
            W = [[0] * rank_new for _ in range(pn)]
            
            col = 0
            for ba in range(2):
                for bb in range(4):
                    for t in range(7):
                        orig_col = ba * 28 + bb * 7 + t
                        if orig_col in selected:
                            U_loc, V_loc, W_loc = strassen_terms[t]
                            for i_l in range(2):
                                for j_l in range(2):
                                    u_idx = i_l * m + (2 * ba + j_l)
                                    U[u_idx][col] = U_loc[i_l * 2 + j_l]
                            for j_l in range(2):
                                for k_l in range(2):
                                    v_idx = (2 * ba + j_l) * p + (2 * bb + k_l)
                                    V[v_idx][col] = V_loc[j_l * 2 + k_l]
                            for k_l in range(2):
                                for i_l in range(2):
                                    w_idx = (2 * bb + k_l) * n + i_l
                                    W[w_idx][col] = W_loc[k_l * 2 + i_l]
                            col += 1
            
            # Verify the decomposition
            U_arr = np.array(U, dtype=np.float64)
            V_arr = np.array(V, dtype=np.float64)
            W_arr = np.array(W, dtype=np.float64)
            
            # Compute the tensor from the decomposition
            T_comp = np.zeros(nrm * mp * pn)
            for r in range(rank_new):
                for i in range(nrm):
                    if U_arr[i, r] != 0:
                        for j in range(mp):
                            if V_arr[j, r] != 0:
                                for k in range(pn):
                                    if W_arr[k, r] != 0:
                                        T_comp[i * mp * pn + j * pn + k] += U_arr[i, r] * V_arr[j, r] * W_arr[k, r]
            
            # Compute the true multiplication tensor
            T_true = np.zeros(nrm * mp * pn)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_true[i * mp * pn + j * pn + k] = 1.0
            
            if np.allclose(T_comp, T_true, atol=1e-9):
                return {"U": U, "V": V, "W": W}
    
    # Fallback: return the full rank-56 decomposition
    U = [[0] * rank for _ in range(nrm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    col = 0
    for ba in range(2):
        for bb in range(4):
            for t in range(7):
                U_loc, V_loc, W_loc = strassen_terms[t]
                for i_l in range(2):
                    for j_l in range(2):
                        u_idx = i_l * m + (2 * ba + j_l)
                        U[u_idx][col] = U_loc[i_l * 2 + j_l]
                for j_l in range(2):
                    for k_l in range(2):
                        v_idx = (2 * ba + j_l) * p + (2 * bb + k_l)
                        V[v_idx][col] = V_loc[j_l * 2 + k_l]
                for k_l in range(2):
                    for i_l in range(2):
                        w_idx = (2 * bb + k_l) * n + i_l
                        W[w_idx][col] = W_loc[k_l * 2 + i_l]
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
