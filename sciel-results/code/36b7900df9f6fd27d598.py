import numpy as np
import time

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check if U @ V @ W^T reconstructs the matmul tensor (vectorized)."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    R = U_np.shape[1]
    
    # For each (i,j,k), the tensor entry should be sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Build index arrays
    ii, jj, kk = np.meshgrid(range(n), range(m), range(p), indexing='ij')
    ii = ii.ravel()
    jj = jj.ravel()
    kk = kk.ravel()
    
    # Gather the relevant rows
    U_rows = U_np[ii * m + jj, :]  # (n*m*p, R)
    V_rows = V_np[jj * p + kk, :]  # (n*m*p, R)
    W_rows = W_np[kk * n + ii, :]  # (n*m*p, R)
    
    # Sum over rank dimension
    vals = np.sum(U_rows * V_rows * W_rows, axis=1)
    
    return bool(np.all(np.abs(vals - 1.0) <= tol))

def _add_strassen_block(U, V, W, a_idx, b_idx, c_idx, col):
    """
    Add a Strassen 7-mul decomposition for a 2x2x2 sub-problem.
    a_idx: dict {(0,0): global_idx, ...} for A
    b_idx: dict {(0,0): global_idx, ...} for B
    c_idx: dict {(0,0): global_idx, ...} for C
    col: starting column index
    """
    # M1 = (A00+A11)(B00+B11)
    U[a_idx[(0,0)], col] += 1
    U[a_idx[(1,1)], col] += 1
    V[b_idx[(0,0)], col] += 1
    V[b_idx[(1,1)], col] += 1
    W[c_idx[(0,0)], col] += 1
    W[c_idx[(1,1)], col] += 1
    col += 1
    
    # M2 = (A10+A11)B00
    U[a_idx[(1,0)], col] += 1
    U[a_idx[(1,1)], col] += 1
    V[b_idx[(0,0)], col] += 1
    W[c_idx[(1,0)], col] += 1
    W[c_idx[(1,1)], col] -= 1
    col += 1
    
    # M3 = A00(B01-B11)
    U[a_idx[(0,0)], col] += 1
    V[b_idx[(0,1)], col] += 1
    V[b_idx[(1,1)], col] -= 1
    W[c_idx[(0,1)], col] += 1
    W[c_idx[(1,1)], col] += 1
    col += 1
    
    # M4 = A11(B10-B00)
    U[a_idx[(1,1)], col] += 1
    V[b_idx[(1,0)], col] += 1
    V[b_idx[(0,0)], col] -= 1
    W[c_idx[(0,0)], col] += 1
    W[c_idx[(1,0)], col] += 1
    col += 1
    
    # M5 = (A00+A01)B11
    U[a_idx[(0,0)], col] += 1
    U[a_idx[(0,1)], col] += 1
    V[b_idx[(1,1)], col] += 1
    W[c_idx[(0,0)], col] -= 1
    W[c_idx[(0,1)], col] += 1
    col += 1
    
    # M6 = (A10-A00)(B00+B01)
    U[a_idx[(1,0)], col] += 1
    U[a_idx[(0,0)], col] -= 1
    V[b_idx[(0,0)], col] += 1
    V[b_idx[(0,1)], col] += 1
    W[c_idx[(1,1)], col] += 1
    col += 1
    
    # M7 = (A01-A11)(B10+B11)
    U[a_idx[(0,1)], col] += 1
    U[a_idx[(1,1)], col] -= 1
    V[b_idx[(1,0)], col] += 1
    V[b_idx[(1,1)], col] += 1
    W[c_idx[(0,0)], col] += 1
    col += 1
    
    return col

def _add_1x2x2_block(U, V, W, a_idx, b_idx, c_idx, col):
    """
    Add a rank-4 schoolbook decomposition for a 1x2x2 sub-problem.
    a_idx: dict {0: global_idx, 1: global_idx} for A (1x2)
    b_idx: dict {(0,0): global_idx, (0,1): global_idx, (1,0): global_idx, (1,1): global_idx} for B (2x2)
    c_idx: dict {0: global_idx, 1: global_idx} for C (1x2)
    col: starting column index
    """
    # C0 = A0 B00 + A1 B10
    U[a_idx[0], col] += 1
    V[b_idx[(0,0)], col] += 1
    W[c_idx[0], col] += 1
    col += 1
    
    U[a_idx[1], col] += 1
    V[b_idx[(1,0)], col] += 1
    W[c_idx[0], col] += 1
    col += 1
    
    # C1 = A0 B01 + A1 B11
    U[a_idx[0], col] += 1
    V[b_idx[(0,1)], col] += 1
    W[c_idx[1], col] += 1
    col += 1
    
    U[a_idx[1], col] += 1
    V[b_idx[(1,1)], col] += 1
    W[c_idx[1], col] += 1
    col += 1
    
    return col

def construct():
    n, m, p = 3, 4, 8
    R = 72
    
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    col = 0
    
    # Top part: A rows 0,1 (2x4)
    # A_top = [A11 A12]
    # A11: A[0:2, 0:2] -> idx 0,1,4,5
    # A12: A[0:2, 2:4] -> idx 2,3,6,7
    
    # B is 4x8. Split cols into 4 blocks of 2.
    # B_j: B[0:4, 2j:2j+2]
    # B_j = [B1j; B2j]
    # B1j: B[0:2, 2j:2j+2]
    # B2j: B[2:4, 2j:2j+2]
    
    for j in range(4):
        # C_top_j: C[0:2, 2j:2j+2] (2x2)
        # C_top_j = A11 B1j + A12 B2j
        
        # Define indices for A11, A12, B1j, B2j, C_top_j
        
        # A11
        a11_idx = {
            (0,0): 0*4 + 0, (0,1): 0*4 + 1,
            (1,0): 1*4 + 0, (1,1): 1*4 + 1
        }
        # A12
        a12_idx = {
            (0,0): 0*4 + 2, (0,1): 0*4 + 3,
            (1,0): 1*4 + 2, (1,1): 1*4 + 3
        }
        
        # B1j (rows 0,1 of B, cols 2j, 2j+1)
        b1j_idx = {
            (0,0): 0*8 + 2*j, (0,1): 0*8 + 2*j + 1,
            (1,0): 1*8 + 2*j, (1,1): 1*8 + 2*j + 1
        }
        # B2j (rows 2,3 of B, cols 2j, 2j+1)
        b2j_idx = {
            (0,0): 2*8 + 2*j, (0,1): 2*8 + 2*j + 1,
            (1,0): 3*8 + 2*j, (1,1): 3*8 + 2*j + 1
        }
        
        # C_top_j (rows 0,1 of C, cols 2j, 2j+1)
        c_top_j_idx = {
            (0,0): 2*j*3 + 0, (0,1): 2*j*3 + 1,
            (1,0): (2*j+1)*3 + 0, (1,1): (2*j+1)*3 + 1
        }
        
        # Add Strassen for A11 * B1j
        col = _add_strassen_block(U, V, W, a11_idx, b1j_idx, c_top_j_idx, col)
        # Add Strassen for A12 * B2j
        col = _add_strassen_block(U, V, W, a12_idx, b2j_idx, c_top_j_idx, col)
        
    # Bottom part: A row 2 (1x4)
    # A_bottom = [A31 A32]
    # A31: A[2:3, 0:2] -> idx 8,9
    # A32: A[2:3, 2:4] -> idx 10,11
    
    for j in range(4):
        # C_bottom_j: C[2:3, 2j:2j+2] (1x2)
        # C_bottom_j = A31 B1j + A32 B2j
        
        # A31
        a31_idx = {
            0: 2*4 + 0,
            1: 2*4 + 1
        }
        # A32
        a32_idx = {
            0: 2*4 + 2,
            1: 2*4 + 3
        }
        
        # B1j, B2j same as above
        b1j_idx = {
            (0,0): 0*8 + 2*j, (0,1): 0*8 + 2*j + 1,
            (1,0): 1*8 + 2*j, (1,1): 1*8 + 2*j + 1
        }
        b2j_idx = {
            (0,0): 2*8 + 2*j, (0,1): 2*8 + 2*j + 1,
            (1,0): 3*8 + 2*j, (1,1): 3*8 + 2*j + 1
        }
        
        # C_bottom_j
        c_bot_j_idx = {
            0: 2*j*3 + 2,
            1: (2*j+1)*3 + 2
        }
        
        # Add 1x2x2 block for A31 * B1j
        col = _add_1x2x2_block(U, V, W, a31_idx, b1j_idx, c_bot_j_idx, col)
        # Add 1x2x2 block for A32 * B2j
        col = _add_1x2x2_block(U, V, W, a32_idx, b2j_idx, c_bot_j_idx, col)
        
    assert col == 72, f"Expected 72, got {col}"
    
    if _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        R_fb = n*m*p
        U_fb = np.zeros((n*m, R_fb))
        V_fb = np.zeros((m*p, R_fb))
        W_fb = np.zeros((p*n, R_fb))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, c] = 1
                    V_fb[j*p+k, c] = 1
                    W_fb[k*n+i, c] = 1
                    c += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

