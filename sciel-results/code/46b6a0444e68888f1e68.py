"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def _schoolbook_factors(n, m, p):
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

def _verify(U, V, W, n, m, p, tol=1e-9):
    rank = U.shape[1]
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if U.shape[1] != rank or V.shape[1] != rank or W.shape[1] != rank:
        return False
        
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                x = U_3d[i, j, :] * V_3d[j, k, :]
                val = np.dot(x, W_3d[k, i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _apply_strassen_to_block(U_base, V_base, W_base, n, m, p, i0, j0, k0):
    """
    Replace the 8 schoolbook terms in a 2x2x2 sub-block starting at (i0, j0, k0)
    with 7 Strassen terms.
    
    The sub-block covers:
    - A rows i0, i0+1; columns j0, j0+1
    - B rows j0, j0+1; columns k0, k0+1
    - C rows i0, i0+1; columns k0, k0+1
    
    We need to find the indices in the base factor matrices for these 8 terms
    and replace them with 7 new terms.
    """
    # Find the indices of the 8 schoolbook terms in the base matrices
    # The schoolbook term for (i, j, k) has index: i*m + j in U, j*p + k in V, k*n + i in W
    # But we need to find which column in the base matrices corresponds to each (i,j,k)
    
    # In _schoolbook_factors, the column index for (i,j,k) is:
    # col = i*m*p + j*p + k
    # Wait, let's re-check the schoolbook construction:
    # for i in range(n):
    #     for j in range(m):
    #         for k in range(p):
    #             U[i*m+j, col] = 1
    #             V[j*p+k, col] = 1
    #             W[k*n+i, col] = 1
    #             col += 1
    # So col = i*m*p + j*p + k
    
    # The 8 terms to replace:
    indices = []
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                i = i0 + di
                j = j0 + dj
                k = k0 + dk
                col = i * m * p + j * p + k
                indices.append((i, j, k, col))
    
    # Create new factor matrices by copying the base
    U_new = U_base.copy()
    V_new = V_base.copy()
    W_new = W_base.copy()
    
    # Zero out the 8 columns
    for (i, j, k, col) in indices:
        U_new[i * m + j, col] = 0
        V_new[j * p + k, col] = 0
        W_new[k * n + i, col] = 0
    
    # Now add 7 Strassen terms
    # Strassen for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M6
    # C11 = M1 - M2 + M3 + M6
    #
    # Where A00 is the element at (i0,j0), A01 at (i0,j0+1), etc.
    # B00 at (j0,k0), B01 at (j0,k0+1), etc.
    # C00 at (i0,k0), C01 at (i0,k0+1), etc.
    
    # We'll use the 8 original column indices, replacing 7 of them with Strassen terms
    # and leaving one column zeroed out (or we can reuse the columns)
    
    # Actually, let's use 7 of the 8 columns for the Strassen terms
    cols = [col for (i, j, k, col) in indices]
    
    # M1 = (A00+A11)(B00+B11)
    r = cols[0]
    U_new[i0 * m + j0, r] = 1
    U_new[(i0+1) * m + (j0+1), r] = 1
    V_new[j0 * p + k0, r] = 1
    V_new[(j0+1) * p + (k0+1), r] = 1
    W_new[k0 * n + i0, r] = 1
    W_new[(k0+1) * n + (i0+1), r] = 1
    
    # M2 = (A10+A11)B00
    r = cols[1]
    U_new[(i0+1) * m + j0, r] = 1
    U_new[(i0+1) * m + (j0+1), r] = 1
    V_new[j0 * p + k0, r] = 1
    W_new[k0 * n + (i0+1), r] = 1
    
    # M3 = A00(B01-B11)
    r = cols[2]
    U_new[i0 * m + j0, r] = 1
    V_new[j0 * p + (k0+1), r] = 1
    V_new[(j0+1) * p + (k0+1), r] = -1
    W_new[(k0+1) * n + i0, r] = 1
    
    # M4 = A11(B10-B00)
    r = cols[3]
    U_new[(i0+1) * m + (j0+1), r] = 1
    V_new[(j0+1) * p + k0, r] = 1
    V_new[j0 * p + k0, r] = -1
    W_new[k0 * n + (i0+1), r] = 1
    
    # M5 = (A00+A01)B11
    r = cols[4]
    U_new[i0 * m + j0, r] = 1
    U_new[i0 * m + (j0+1), r] = 1
    V_new[(j0+1) * p + (k0+1), r] = 1
    W_new[(k0+1) * n + i0, r] = 1
    
    # M6 = (A10-A00)(B00+B01)
    r = cols[5]
    U_new[(i0+1) * m + j0, r] = 1
    U_new[i0 * m + j0, r] = -1
    V_new[j0 * p + k0, r] = 1
    V_new[j0 * p + (k0+1), r] = 1
    W_new[k0 * n + (i0+1), r] = 1
    W_new[(k0+1) * n + (i0+1), r] = 1
    
    # M7 = (A01-A11)(B10+B11)
    r = cols[6]
    U_new[i0 * m + (j0+1), r] = 1
    U_new[(i0+1) * m + (j0+1), r] = -1
    V_new[(j0+1) * p + k0, r] = 1
    V_new[(j0+1) * p + (k0+1), r] = 1
    W_new[k0 * n + i0, r] = 1
    
    # The 8th column (cols[7]) remains zeroed out
    # This effectively reduces the rank by 1
    
    return U_new, V_new, W_new

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    U_base, V_base, W_base = _schoolbook_factors(n, m, p)
    
    # Apply Strassen to block (0,0,0)
    U_opt, V_opt, W_opt = _apply_strassen_to_block(U_base, V_base, W_base, n, m, p, 0, 0, 0)
    
    # Verify
    if not _verify(U_opt, V_opt, W_opt, n, m, p):
        # Fallback to schoolbook if verification fails
        U_opt, V_opt, W_opt = U_base, V_base, W_base
    
    return {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
# EVOLVE-BLOCK-END
