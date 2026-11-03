"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Find a low-rank tensor decomposition for 4x4 * 4x6 matrix multiplication.
    Uses block-Strassen: decompose into 2x2 blocks along n and m, and 2-wide
    column blocks along p. For each (a, c, jb) block triple, apply 7 Strassen
    terms. Total rank = 2 * 3 * 2 * 7 = 84.
    
    The tensor decomposition convention:
    - U has shape (n*m, rank), indexed by (i, j) -> U[i*m+j, r]
    - V has shape (m*p, rank), indexed by (j, k) -> V[j*p+k, r]  
    - W has shape (p*n, rank), indexed by (k, i) -> W[k*n+i, r]
    
    The tensor is T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    For matrix multiplication, T[i,j,k] = delta_{i,k} (since C[i,k] = sum_j A[i,j]*B[j,k])
    """
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Block sizes
    bn = 2  # block size for n dimension
    bm = 2  # block size for m dimension  
    bp = 2  # block size for p dimension
    
    n_blocks = n // bn  # 2
    m_blocks = m // bm  # 2
    p_blocks = p // bp  # 3
    
    # Strassen terms for 2x2 * 2x2:
    # Each term is (A_combo, B_combo, C_combo) where:
    #   A_combo: dict mapping (ar, ac) -> coeff for A[ar, ac]
    #   B_combo: dict mapping (br, bc) -> coeff for B[br, bc]
    #   C_combo: dict mapping (cr, cc) -> coeff for C[cr, cc]
    #
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00       -> C10 += M2
    # M3 = A00(B01-B11)       -> C01 += M3
    # M4 = A11(B10-B00)       -> C00 += M4, C10 += M4
    # M5 = (A00+A01)B11       -> C01 += M5, C00 -= M5
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    # M7 = (A01-A11)(B10+B11) -> C00 += M7, C11 -= M7
    
    strassen_terms = [
        # M1: (A00+A11)(B00+B11)
        ({(0,0):1, (1,1):1},
         {(0,0):1, (1,1):1},
         {(0,0):1, (1,1):1}),
        # M2: (A10+A11)B00
        ({(1,0):1, (1,1):1},
         {(0,0):1},
         {(1,0):1}),
        # M3: A00(B01-B11)
        ({(0,0):1},
         {(0,1):1, (1,1):-1},
         {(0,1):1}),
        # M4: A11(B10-B00)
        ({(1,1):1},
         {(1,0):1, (0,0):-1},
         {(0,0):1, (1,0):1}),
        # M5: (A00+A01)B11
        ({(0,0):1, (0,1):1},
         {(1,1):1},
         {(0,1):1, (0,0):-1}),
        # M6: (A10-A00)(B00+B01)
        ({(1,0):1, (0,0):-1},
         {(0,0):1, (0,1):1},
         {(1,1):1}),
        # M7: (A01-A11)(B10+B11)
        ({(0,1):1, (1,1):-1},
         {(1,0):1, (1,1):1},
         {(0,0):1, (1,1):-1}),
    ]
    
    max_rank = n_blocks * p_blocks * m_blocks * 7
    U_data = np.zeros((n*m, max_rank))
    V_data = np.zeros((m*p, max_rank))
    W_data = np.zeros((p*n, max_rank))
    
    col = 0
    for a in range(n_blocks):
        for c in range(p_blocks):
            for jb in range(m_blocks):
                for a_coeffs, b_coeffs, c_coeffs in strassen_terms:
                    # A-side: A is n x m, block A[a, jb] has entries A[2*a+ar, 2*jb+ac]
                    for (ar, ac), val in a_coeffs.items():
                        row = 2*a + ar
                        col_a = 2*jb + ac
                        U_data[row*m + col_a, col] += val
                    # B-side: B is m x p, block B[jb, c] has entries B[2*jb+br, 2*c+bc]
                    for (br, bc), val in b_coeffs.items():
                        row_b = 2*jb + br
                        col_b = 2*c + bc
                        V_data[row_b*p + col_b, col] += val
                    # C-side: C is n x p, block C[a, c] has entries C[2*a+cr, 2*c+cc]
                    # W is indexed by (k, i) where k is output column (p-dim), i is output row (n-dim)
                    # So W[k*n + i, col] where k = 2*c+cc, i = 2*a+cr
                    for (cr, cc), val in c_coeffs.items():
                        out_row = 2*a + cr  # i index
                        out_col = 2*c + cc  # k index
                        W_data[out_col*n + out_row, col] += val
                    col += 1
    
    rank = col
    
    # Verify the decomposition
    U_np = U_data[:, :rank]
    V_np = V_data[:, :rank]
    W_np = W_data[:, :rank]
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_np[i*m+j] * V_np[j*p+k], W_np[k*n+i])
                expected = 1.0 if i == k else 0.0
                if abs(val - expected) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook
        r2 = n*m*p
        U2 = np.zeros((n*m, r2)); V2 = np.zeros((m*p, r2)); W2 = np.zeros((p*n, r2))
        cc = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U2[ii*m+jj, cc] = 1
                    V2[jj*p+kk, cc] = 1
                    W2[kk*n+ii, cc] = 1
                    cc += 1
        return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
    
    return {"U": U_np.tolist(), "V": V_np.tolist(), "W": W_np.tolist()}
# EVOLVE-BLOCK-END
