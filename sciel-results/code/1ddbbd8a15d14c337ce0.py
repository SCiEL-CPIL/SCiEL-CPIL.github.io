# EVOLVE-BLOCK-START
import numpy as np
from numpy.linalg import svd, norm
import time
import itertools
from fractions import Fraction

def _build_rank14():
    """Return the verified rank-14 Strassen-based decomposition."""
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # A indices: A[0]=A00, A[1]=A01, A[2]=A10, A[3]=A11
    # B indices: B[0]=B00, B[1]=B01, B[2]=B02, B[3]=B03, B[4]=B10, B[5]=B11, B[6]=B12, B[7]=B13
    # C indices (W): C[0]=C00, C[1]=C10, C[2]=C01, C[3]=C11, C[4]=C02, C[5]=C12, C[6]=C03, C[7]=C13
    
    # Left block (cols 0,1) - Strassen
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[5][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4 = A11(B10-B00)
    U[3][3] = 1
    V[4][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[5][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[4][6] = 1; V[5][6] = 1
    W[0][6] = 1

    # Right block (cols 2,3) - Strassen
    # N1 = (A00+A11)(B02+B13)
    U[0][7] = 1; U[3][7] = 1
    V[2][7] = 1; V[7][7] = 1
    W[4][7] = 1; W[7][7] = 1
    
    # N2 = (A10+A11)B02
    U[2][8] = 1; U[3][8] = 1
    V[2][8] = 1
    W[5][8] = 1; W[7][8] = -1
    
    # N3 = A00(B03-B13)
    U[0][9] = 1
    V[3][9] = 1; V[7][9] = -1
    W[6][9] = 1; W[7][9] = 1
    
    # N4 = A11(B12-B02)
    U[3][10] = 1
    V[6][10] = 1; V[2][10] = -1
    W[4][10] = 1; W[5][10] = 1
    
    # N5 = (A00+A01)B13
    U[0][11] = 1; U[1][11] = 1
    V[7][11] = 1
    W[4][11] = -1; W[6][11] = 1
    
    # N6 = (A10-A00)(B02+B03)
    U[2][12] = 1; U[0][12] = -1
    V[2][12] = 1; V[3][12] = 1
    W[7][12] = 1
    
    # N7 = (A01-A11)(B12+B13)
    U[1][13] = 1; U[3][13] = -1
    V[6][13] = 1; V[7][13] = 1
    W[4][13] = 1

    return {"U": U, "V": V, "W": W}

def _verify_decomposition(result, tol=1e-10):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(result['U'], dtype=float)
    V = np.array(result['V'], dtype=float)
    W = np.array(result['W'], dtype=float)
    
    n, m, p = 2, 2, 4
    rank = U.shape[1]
    
    # Build the tensor T[i,j,k] where C[k*n+i] = sum_j A[i,j] * B[j,k]
    # T[a_idx, b_idx, c_idx] = 1 if c_idx = k*n+i where A[i,j]*B[j,k] contributes
    # A indices: A[0]=A00, A[1]=A01, A[2]=A10, A[3]=A11 (row-major)
    # B indices: B[0]=B00, B[1]=B01, B[2]=B02, B[3]=B03, B[4]=B10, B[5]=B11, B[6]=B12, B[7]=B13
    # C indices (W): C[0]=C00, C[1]=C10, C[2]=C01, C[3]=C11, C[4]=C02, C[5]=C12, C[6]=C03, C[7]=C13
    
    # The tensor element T[a,b,c] = sum over r of U[a,r] * V[b,r] * W[c,r]
    # This should equal 1 if there exists j such that a = i*m+j, b = j*p+k, c = k*n+i
    # For our case: n=2, m=2, p=4
    # a = i*2+j, b = j*4+k, c = k*2+i
    
    # Build expected tensor
    T_expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T_expected[a_idx, b_idx, c_idx] = 1.0
    
    # Compute actual tensor from decomposition
    # T_actual[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    T_actual = np.einsum('ar,br,cr->abc', U, V, W)
    
    diff = np.abs(T_actual - T_expected)
    max_diff = np.max(diff)
    return max_diff < tol, max_diff

def _search_rank13(deadline):
    """
    Search for a rank-13 decomposition using a modified Strassen structure.
    The idea: try to merge or eliminate one term by finding a linear dependency.
    """
    n, m, p = 2, 2, 4
    rank = 13
    
    # Start from rank-14 and try to find a linear combination that eliminates one term
    base = _build_rank14()
    U14 = np.array(base['U'], dtype=float)
    V14 = np.array(base['V'], dtype=float)
    W14 = np.array(base['W'], dtype=float)
    
    # Strategy: try to express one column as a linear combination of others
    # For each column to eliminate, check if it's in the span of the others
    for elim_col in range(14):
        if time.time() > deadline:
            break
        
        # Try to find coefficients alpha such that col elim = sum_{r != elim} alpha_r * col_r
        # For U, V, W simultaneously
        
        # Build the system: for each row in U, V, W, the eliminated column should
        # equal the linear combination of other columns
        
        # This is a linear system. Let's set it up.
        # For U: U[:, elim] = U[:, [r for r in range(14) if r != elim]] @ alpha
        # For V: V[:, elim] = V[:, [r for r in range(14) if r != elim]] @ alpha  
        # For W: W[:, elim] = W[:, [r for r in range(14) if r != elim]] @ alpha
        
        other_cols = [r for r in range(14) if r != elim_col]
        
        # Stack all three matrices vertically
        U_other = U14[:, other_cols]  # (4, 13)
        V_other = V14[:, other_cols]  # (8, 13)
        W_other = W14[:, other_cols]  # (8, 13)
        
        U_elim = U14[:, elim_col]  # (4,)
        V_elim = V14[:, elim_col]  # (8,)
        W_elim = W14[:, elim_col]  # (8,)
        
        # Stack: [U_other; V_other; W_other] @ alpha = [U_elim; V_elim; W_elim]
        A_stack = np.vstack([U_other, V_other, W_other])  # (20, 13)
        b_stack = np.concatenate([U_elim, V_elim, W_elim])  # (20,)
        
        # Solve least squares
        try:
            alpha, residuals, rank_sys, sv = np.linalg.lstsq(A_stack, b_stack, rcond=None)
        except:
            continue
        
        # Check if the solution is exact
        recon = A_stack @ alpha
        if np.allclose(recon, b_stack, atol=1e-10):
            # We found an exact linear dependency!
            # Build the rank-13 matrices
            U13 = np.zeros((4, 13))
            V13 = np.zeros((8, 13))
            W13 = np.zeros((8, 13))
            
            for new_col, old_col in enumerate(other_cols):
                U13[:, new_col] = U14[:, old_col]
                V13[:, new_col] = V14[:, old_col]
                W13[:, new_col] = W14[:, old_col]
            
            # Verify
            result = {"U": U13.tolist(), "V": V13.tolist(), "W": W13.tolist()}
            valid, max_diff = _verify_decomposition(result)
            if valid:
                return result
    
    return None

def construct():
    deadline = time.time() + 120  # Safety margin
    
    # First try to find rank-13
    result13 = _search_rank13(deadline)
    if result13 is not None:
        return result13
    
    # Fall back to rank-14
    return _build_rank14()
# EVOLVE-BLOCK-END
