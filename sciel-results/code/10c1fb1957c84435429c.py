import time
import numpy as np
from scipy.linalg import svd

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 3x3 matrix by a 3x8 matrix.
    
    Strategy:
    - Use Strassen's 7-multiplication algorithm for 2x2 * 2x2 blocks
    - Apply this to overlapping 2x2 blocks to share terms
    - Use linear algebra to find and remove linearly dependent terms
    """
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill
    
    def strassen_decomp(row_off_a, col_off_a, row_off_b, col_off_b, U_t, V_t, W_t):
        """
        Add Strassen decomposition for 2x2 * 2x2 block at given offsets.
        """
        a = (row_off_a, col_off_a)
        b = (row_off_a, col_off_a + 1)
        c = (row_off_a + 1, col_off_a)
        d = (row_off_a + 1, col_off_a + 1)
        e = (row_off_b, col_off_b)
        f = (row_off_b, col_off_b + 1)
        g = (row_off_b + 1, col_off_b)
        h = (row_off_b + 1, col_off_b + 1)
        
        C00 = (row_off_a, col_off_b)
        C01 = (row_off_a, col_off_b + 1)
        C10 = (row_off_a + 1, col_off_b)
        C11 = (row_off_a + 1, col_off_b + 1)
        
        # m1 = (a+d)(e+h)
        U_t.append([(a, 1), (d, 1)])
        V_t.append([(e, 1), (h, 1)])
        W_t.append([(C00, 1), (C11, 1)])
        
        # m2 = (c+d)*e
        U_t.append([(c, 1), (d, 1)])
        V_t.append([(e, 1)])
        W_t.append([(C10, 1), (C11, -1)])
        
        # m3 = a*(f-h)
        U_t.append([(a, 1)])
        V_t.append([(f, 1), (h, -1)])
        W_t.append([(C01, 1), (C11, 1)])
        
        # m4 = d*(g-e)
        U_t.append([(d, 1)])
        V_t.append([(g, 1), (e, -1)])
        W_t.append([(C00, 1), (C10, 1)])
        
        # m5 = (a+b)*h
        U_t.append([(a, 1), (b, 1)])
        V_t.append([(h, 1)])
        W_t.append([(C00, -1), (C01, 1)])
        
        # m6 = (c-a)*(e+f)
        U_t.append([(c, 1), (a, -1)])
        V_t.append([(e, 1), (f, 1)])
        W_t.append([(C11, 1)])
        
        # m7 = (b-d)*(g+h)
        U_t.append([(b, 1), (d, -1)])
        V_t.append([(g, 1), (h, 1)])
        W_t.append([(C00, 1)])
        
    def direct_term(i, j, k, U_t, V_t, W_t):
        """Add a direct term A[i,j]*B[j,k] to C[i,k]."""
        U_t.append([((i, j), 1)])
        V_t.append([((j, k), 1)])
        W_t.append([((i, k), 1)])
        
    U_t = []
    V_t = []
    W_t = []
    
    # Use overlapping Strassen blocks to cover A[0:2, 0:2] with B columns
    # Block 1: A[0:2,0:2] * B[0:2, 0:2]
    strassen_decomp(0, 0, 0, 0, U_t, V_t, W_t)
    # Block 2: A[0:2,0:2] * B[0:2, 2:4]
    strassen_decomp(0, 0, 0, 2, U_t, V_t, W_t)
    # Block 3: A[0:2,0:2] * B[0:2, 4:6]
    strassen_decomp(0, 0, 0, 4, U_t, V_t, W_t)
    # Block 4: A[0:2,0:2] * B[0:2, 6:8]
    strassen_decomp(0, 0, 0, 6, U_t, V_t, W_t)
    
    # Remaining: A[0:2,2] * B[2, 0:8]
    for i in range(2):
        for k in range(8):
            direct_term(i, 2, k, U_t, V_t, W_t)
    
    # A[2, 0:2] * B[0:2, 0:8]
    for j in range(2):
        for k in range(8):
            direct_term(2, j, k, U_t, V_t, W_t)
    
    # A[2,2] * B[2, 0:8]
    for k in range(8):
        direct_term(2, 2, k, U_t, V_t, W_t)
    
    rank = len(W_t)
    
    # Build factor matrices as numpy arrays
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for idx in range(rank):
        for (i, j), coeff in U_t[idx]:
            U[i * m + j, idx] = coeff
        for (j, k), coeff in V_t[idx]:
            V[j * p + k, idx] = coeff
        for (i, k), coeff in W_t[idx]:
            W[k * n + i, idx] = coeff
    
    # Try to find linear dependencies and reduce rank
    # The tensor T = sum_k U_k ⊗ V_k ⊗ W_k
    # If some columns are linearly dependent, we can reduce rank
    # Check if we can remove terms that are linear combinations of others
    
    def try_reduce_rank(U, V, W, max_iter=5):
        """Try to reduce rank by finding linear dependencies."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        for _ in range(max_iter):
            if time.time() > DEADLINE:
                break
            r = U.shape[1]
            if r == 0:
                break
            # Try removing each column and check if tensor is preserved
            for col in range(r):
                if time.time() > DEADLINE:
                    return U, V, W
                # Check if column col is a linear combination of others
                # Build matrix M where each column is the "flattened" contribution
                # T_col = U[:,col] ⊗ V[:,col] ⊗ W[:,col]
                # We need to check if this can be expressed as sum of other terms
                # This is equivalent to checking if the 3D tensor T has rank < r
                # Use a simpler heuristic: check if the slice matrices have rank deficiency
                
                # For each fixed index in one dimension, check if the resulting 
                # 2D matrices are linearly dependent
                # Try: for each i, look at the matrix (V * W^T) scaled by U[i]
                # Actually, let's use a direct approach:
                # The tensor T can be seen as a linear map from A to C given B
                # T[a,b,c] = sum_k U[a,k]*V[b,k]*W[c,k]
                
                # Check if removing column k still gives the same tensor
                U_test = np.delete(U, col, axis=1)
                V_test = np.delete(V, col, axis=1)
                W_test = np.delete(W, col, axis=1)
                
                # Reconstruct tensor from remaining terms
                # T_test[i,j,k] = sum_l U_test[i,l]*V_test[j,l]*W_test[k,l]
                # Compare with original T[i,j,k] - U[i,col]*V[j,col]*W[k,col]
                
                # Original contribution of column col:
                T_col = np.einsum('ik,jk,lk->ijkl', U[:, col:col+1], V[:, col:col+1], W[:, col:col+1])
                T_col = T_col.squeeze()
                
                # Reconstruct from remaining:
                T_rem = np.einsum('ik,jk,lk->ijkl', U_test, V_test, W_test)
                
                # The full tensor should be T_rem + T_col
                # If T_col is already in T_rem, we can remove it
                # Check if T_col is in the span of the remaining tensors
                # Flatten T_col and check against flattened remaining tensors
                T_col_flat = T_col.flatten()
                
                # Build matrix of remaining tensor slices
                # Each remaining term l gives a tensor U_test[:,l] ⊗ V_test[:,l] ⊗ W_test[:,l]
                # Flatten each and stack
                r_rem = U_test.shape[1]
                if r_rem == 0:
                    # Only one term left, can't reduce
                    continue
                T_rem_flat = np.zeros((r_rem, T_col_flat.size))
                for l in range(r_rem):
                    T_l = np.einsum('i,j,k->ijk', U_test[:, l], V_test[:, l], W_test[:, l])
                    T_rem_flat[l, :] = T_l.flatten()
                
                # Check if T_col_flat is in the column space of T_rem_flat
                # Use SVD to check
                try:
                    U_svd, s, Vt_svd = np.linalg.svd(T_rem_flat, full_matrices=False)
                    # Project T_col_flat onto the column space
                    proj = U_svd @ (U_svd.T @ T_col_flat)
                    residual = T_col_flat - proj
                    if np.linalg.norm(residual) < 1e-9 * np.linalg.norm(T_col_flat) + 1e-12:
                        # Column col is redundant, remove it
                        U = U_test
                        V = V_test
                        W = W_test
                        break
                except:
                    continue
            else:
                # No column could be removed
                break
        return U, V, W
    
    U, V, W = try_reduce_rank(U, V, W)
    
    def to_nice_list(mat):
        """Convert matrix to list with integer/half-integer entries."""
        res = []
        for row in mat:
            nrow = []
            for v in row:
                if abs(v - round(v*2)/2) < 1e-9:
                    nv = round(v*2)/2
                    if abs(nv - round(nv)) < 1e-9:
                        nrow.append(int(round(nv)))
                    else:
                        nrow.append(nv)
                else:
                    nrow.append(v)
            res.append(nrow)
        return res
        
    return {"U": to_nice_list(U), "V": to_nice_list(V), "W": to_nice_list(W)}
# EVOLVE-BLOCK-END
