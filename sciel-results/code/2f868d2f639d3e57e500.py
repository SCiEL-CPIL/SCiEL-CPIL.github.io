import numpy as np
import time
from itertools import product

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 * 2x6 matrix multiplication.
    Attempts rank 20 via slice merging; falls back to rank 21 Strassen.
    """
    n, m, p = 2, 2, 6
    nm, mp, pn = n * m, m * p, p * n  # 4, 12, 12
    
    deadline = time.time() + 140.0
    
    # Build the target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def verify(U, V, W):
        """Check if U, V, W decompose T exactly."""
        T_check = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_check, T, atol=1e-10)
    
    def strassen_21():
        """Build rank-21 Strassen decomposition."""
        rank = 21
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        strassen_steps = [
            {'u': [1, 0, 0, 1], 'v': [1, 0, 0, 1], 'w': [1, 0, 0, 1]},
            {'u': [0, 0, 1, 1], 'v': [1, 0, 0, 0], 'w': [0, 0, 1, -1]},
            {'u': [1, 0, 0, 0], 'v': [0, 1, 0, -1], 'w': [0, 1, 0, 1]},
            {'u': [0, 0, 0, 1], 'v': [-1, 0, 1, 0], 'w': [1, 0, 1, 0]},
            {'u': [1, 1, 0, 0], 'v': [0, 0, 0, 1], 'w': [-1, 1, 0, 0]},
            {'u': [-1, 0, 1, 0], 'v': [1, 1, 0, 0], 'w': [0, 0, 0, 1]},
            {'u': [0, 1, 0, -1], 'v': [0, 0, 1, 1], 'w': [1, 0, 0, 0]},
        ]
        
        def map_a(local_idx):
            return local_idx
            
        def map_b(local_idx, g):
            k0 = 2 * g
            k1 = 2 * g + 1
            if local_idx == 0: return 0 * 6 + k0
            elif local_idx == 1: return 0 * 6 + k1
            elif local_idx == 2: return 1 * 6 + k0
            else: return 1 * 6 + k1

        def map_c(local_idx, g):
            k0 = 2 * g
            k1 = 2 * g + 1
            if local_idx == 0: return k0 * 2 + 0
            elif local_idx == 1: return k1 * 2 + 0
            elif local_idx == 2: return k0 * 2 + 1
            else: return k1 * 2 + 1

        for g in range(3):
            for s in range(7):
                r = 7 * g + s
                step = strassen_steps[s]
                for local_idx in range(4):
                    u_val = step['u'][local_idx]
                    if u_val != 0:
                        U[map_a(local_idx), r] = u_val
                    v_val = step['v'][local_idx]
                    if v_val != 0:
                        V[map_b(local_idx, g), r] = v_val
                    w_val = step['w'][local_idx]
                    if w_val != 0:
                        W[map_c(local_idx, g), r] = w_val
        return U, V, W
    
    def schoolbook():
        """Build rank-24 schoolbook decomposition."""
        rank = 24
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        return U, V, W
    
    # First, get the rank-21 baseline
    U21, V21, W21 = strassen_21()
    if verify(U21, V21, W21):
        best_U, best_V, best_W = U21.copy(), V21.copy(), W21.copy()
        best_rank = 21
    else:
        best_U, best_V, best_W = schoolbook()
        best_rank = 24
    
    # Try to find rank 20 by removing one rank-1 term from the 21-term decomposition
    # and adjusting the remaining terms
    # Strategy: For each pair of terms to "merge" (or one to remove), check if the residual
    # can be absorbed
    
    # A more direct approach: try to see if rank 20 is achievable by checking
    # if the tensor has border rank 20 or if there's a valid rank-20 decomposition.
    # Since we can't easily do a full search, let's try a heuristic:
    # Remove one term at a time and see if the residual is rank-1 (can be absorbed)
    
    # For each term r to potentially remove:
    for r_remove in range(21):
        if time.time() > deadline:
            break
        
        # Create the decomposition without term r_remove
        U_test = np.zeros((nm, 20))
        V_test = np.zeros((mp, 20))
        W_test = np.zeros((pn, 20))
        
        col = 0
        for r in range(21):
            if r == r_remove:
                continue
            U_test[:, col] = U21[:, r]
            V_test[:, col] = V21[:, r]
            W_test[:, col] = W21[:, r]
            col += 1
        
        # Compute residual: T - sum of remaining terms
        T_residual = T - np.einsum('ar,br,cr->abc', U_test, V_test, W_test)
        
        # Check if T_residual is rank-1 (or zero)
        # A tensor is rank-1 if it can be written as u ⊗ v ⊗ w
        # This means all its "slices" are proportional
        
        # Flatten and check rank of certain unfoldings
        # For a rank-1 tensor T[i,j,k] = a[i]*b[j]*c[k]
        # The matrix unfolding T_(i|jk) should have rank 1
        
        # Check unfolding along first mode
        T_0 = T_residual.reshape(nm, mp * pn)
        if T_0.size > 0:
            # Check if all rows are proportional
            nonzero_rows = np.any(T_0 != 0, axis=1)
            if np.sum(nonzero_rows) <= 1:
                # At most one nonzero row, so rank is at most 1
                if np.allclose(T_0, 0, atol=1e-10):
                    # Residual is zero, we found rank 20!
                    best_U, best_V, best_W = U_test, V_test, W_test
                    best_rank = 20
                    break
                # Check if it's rank 1 by seeing if all nonzero rows are multiples of each other
                nz_idx = np.where(nonzero_rows)[0]
                if len(nz_idx) == 1:
                    # Only one nonzero row, rank is 1
                    # We need to express this as a rank-1 term
                    # T_residual = u ⊗ v ⊗ w where u has one nonzero entry
                    i0 = nz_idx[0]
                    # v and w are the remaining 2D slice
                    slice_2d = T_residual[i0, :, :]
                    # Check if slice_2d is rank 1
                    # For a 12x12 matrix, check if all rows are proportional
                    nz_rows_2d = np.any(slice_2d != 0, axis=1)
                    if np.sum(nz_rows_2d) <= 1:
                        # Rank 1 or 0
                        if np.allclose(slice_2d, 0, atol=1e-10):
                            best_U, best_V, best_W = U_test, V_test, W_test
                            best_rank = 20
                            break
                        # It's a rank-1 term: u[i0]=1, and slice_2d is outer product
                        # Find v and w such that slice_2d = v * w^T
                        nz_cols_2d = np.any(slice_2d != 0, axis=0)
                        if np.sum(nz_cols_2d) <= 1:
                            # Rank 1
                            nz_j = np.where(nz_rows_2d)[0]
                            nz_k = np.where(nz_cols_2d)[0]
                            if len(nz_j) == 1 and len(nz_k) == 1:
                                val = slice_2d[nz_j[0], nz_k[0]]
                                # Add this rank-1 term
                                U_test[i0, 0] = 1.0
                                V_test[nz_j[0], 0] = 1.0
                                W_test[nz_k[0], 0] = 1.0
                                # But we need to adjust the scaling
                                # Actually, this is getting complex. Let's just verify.
                                U_final = U_test.copy()
                                V_final = V_test.copy()
                                W_final = W_test.copy()
                                U_final[i0, 0] = 1.0
                                V_final[nz_j[0], 0] = 1.0
                                W_final[nz_k[0], 0] = val
                                # Shift the other terms
                                U_final[:, 1:] = U_test[:, :19]
                                V_final[:, 1:] = V_test[:, :19]
                                W_final[:, 1:] = W_test[:, :19]
                                if verify(U_final, V_final, W_final):
                                    best_U, best_V, best_W = U_final, V_final, W_final
                                    best_rank = 20
                                    break
                        elif np.sum(nz_rows_2d) == 1:
                            # One row, check if it's rank 1
                            row = slice_2d[nz_rows_2d, :]
                            # Check if row is rank 1 (it's a vector, so always rank 1 if nonzero)
                            # But we need the whole tensor to be rank 1
                            # This means slice_2d has only one nonzero row
                            # So the tensor is rank 1
                            U_final = U_test.copy()
                            V_final = V_test.copy()
                            W_final = W_test.copy()
                            U_final[i0, 0] = 1.0
                            # V and W need to encode the row
                            # row = v * w^T where v is in R^12, w is in R^12
                            # Since only one row is nonzero, v has one nonzero entry
                            j0 = nz_rows_2d
                            # row = v[j0] * w^T
                            # So w is proportional to row, and v[j0] is the scaling
                            if np.any(row != 0):
                                w = row.copy()
                                norm = np.linalg.norm(w)
                                if norm > 1e-10:
                                    w = w / norm
                                    v_val = norm
                                    V_final[j0, 0] = v_val
                                    W_final[:, 0] = w
                                    if verify(U_final, V_final, W_final):
                                        best_U, best_V, best_W = U_final, V_final, W_final
                                        best_rank = 20
                                        break
    
    # If we didn't find rank 20, return rank 21
    if best_rank == 21:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    elif best_rank == 20:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
