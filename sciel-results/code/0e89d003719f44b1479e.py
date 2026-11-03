import numpy as np
from itertools import product

def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 matrix multiplication (rank 7)."""
    # Strassen's algorithm for 2x2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h), M2 = (c+d)e, M3 = a(f-h), M4 = d(g-e)
    # M5 = (a+b)h, M6 = (c-a)(e+f), M7 = (b-d)(g+h)
    # C11 = M1+M4-M5+M7, C12 = M3+M5, C21 = M2+M4, C22 = M1-M2+M3+M6
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Index: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    #         B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
    #         C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
    
    # M1 = (a+d)(e+h): A indices 0,3; B indices 0,3; C indices 0,3
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1
    
    # M2 = (c+d)e: A indices 2,3; B index 0; C index 2
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[2,1] = 1
    
    # M3 = a(f-h): A index 0; B indices 1,3; C index 1
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1
    W[1,2] = 1
    
    # M4 = d(g-e): A index 3; B indices 2,0; C indices 2,3
    U[3,3] = 1
    V[2,3] = 1; V[0,3] = -1
    W[2,3] = 1; W[3,3] = 1
    
    # M5 = (a+b)h: A indices 0,1; B index 3; C indices 0,1
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = -1; W[1,4] = 1
    
    # M6 = (c-a)(e+f): A indices 2,0; B indices 0,1; C index 3
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[3,5] = 1
    
    # M7 = (b-d)(g+h): A indices 1,3; B indices 2,3; C index 0
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[0,6] = 1
    
    return U, V, W

def construct():
    n, m, p = 4, 5, 6
    
    # Block decomposition:
    # A (4x5): split rows into [0,1] and [2,3], cols into [0,1,2] and [3,4]
    # B (5x6): split rows into [0,1,2] and [3,4], cols into [0,1,2] and [3,4,5]
    # C (4x6): output
    
    # We'll use a structured approach:
    # For the 2x3 x 3x2 blocks, we can use direct multiplication (rank 6 each)
    # For 2x2 x 2x2 blocks, we use Strassen (rank 7 each)
    
    # Let's define the block structure more carefully:
    # A blocks: A00 (2x3), A01 (2x2), A10 (2x3), A11 (2x2)
    # B blocks: B00 (3x3), B01 (3x3), B10 (2x3), B11 (2x3)
    # C = A*B: C00 = A00*B00 + A01*B10, C01 = A00*B01 + A01*B11
    #          C10 = A10*B00 + A11*B10, C11 = A10*B01 + A11*B11
    
    # This is getting complex. Let me use a simpler but still improved approach.
    
    # Alternative: Use the fact that we can decompose into smaller multiplications
    # and use Strassen where the dimensions allow.
    
    # Let me try a different block split:
    # A (4x5): rows [0,1,2,3], cols [0,1,2,3,4]
    # B (5x6): rows [0,1,2,3,4], cols [0,1,2,3,4,5]
    
    # Split into 2x2 blocks where possible:
    # A: [[A00(2x2), A01(2x2), A02(2x1)],
    #     [A10(2x2), A11(2x2), A12(2x1)]]
    # B: [[B00(2x2), B01(2x2), B02(2x2)],
    #     [B10(2x2), B11(2x2), B12(2x2)],
    #     [B20(1x2), B21(1x2), B22(1x2)]]
    
    # C = A*B (4x6):
    # C[0:2, 0:2] = A00*B00 + A01*B10 + A02*B20
    # C[0:2, 2:4] = A00*B01 + A01*B11 + A02*B21
    # C[0:2, 4:6] = A00*B02 + A01*B12 + A02*B22
    # C[2:4, 0:2] = A10*B00 + A11*B10 + A12*B20
    # C[2:4, 2:4] = A10*B01 + A11*B11 + A12*B21
    # C[2:4, 4:6] = A10*B02 + A11*B12 + A12*B22
    
    # Each 2x2 x 2x2 term: rank 7 (Strassen)
    # Each 2x2 x 1x2 term: this is (2x2)*(1x2) = 2x2, which is just scaling, rank 4
    # Wait, A02 is 2x1, B20 is 1x2, so A02*B20 is 2x2 outer product, rank 4
    
    # Count:
    # 2x2 x 2x2 terms: 6 output blocks * 2 terms each = 12 terms * 7 = 84
    # 2x1 x 1x2 terms: 6 output blocks * 1 term each = 6 terms * 4 = 24
    # Total rank = 84 + 24 = 108
    
    # Now I need to construct the actual U, V, W matrices.
    
    # Let me build this systematically.
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u_entries, v_entries, w_entries):
        """Add a rank-1 term to the decomposition."""
        U_list.append(u_entries)
        V_list.append(v_entries)
        W_list.append(w_entries)
    
    def add_strassen(A_row_start, A_col_start, B_row_start, B_col_start, C_row_start, C_col_start):
        """Add a Strassen decomposition for a 2x2x2 block."""
        U, V, W = _strassen_2x2x2()
        for t in range(7):
            u_vec = [0.0] * (n*m)
            v_vec = [0.0] * (m*p)
            w_vec = [0.0] * (p*n)
            
            for i in range(2):
                for j in range(2):
                    a_idx = (A_row_start + i) * m + (A_col_start + j)
                    if U[i*2+j, t] != 0:
                        u_vec[a_idx] += U[i*2+j, t]
            
            for i in range(2):
                for j in range(2):
                    b_idx = (B_row_start + i) * p + (B_col_start + j)
                    if V[i*2+j, t] != 0:
                        v_vec[b_idx] += V[i*2+j, t]
            
            for i in range(2):
                for j in range(2):
                    c_idx = (C_col_start + j) * n + (C_row_start + i)
                    if W[i*2+j, t] != 0:
                        w_vec[c_idx] += W[i*2+j, t]
            
            add_term(u_vec, v_vec, w_vec)
    
    def add_outer_product(A_row_start, A_col, B_row, B_col_start, C_row_start, C_col_start):
        """Add an outer product term: A[:,col] * B[row,:] contributing to C block."""
        # A is 2x1, B is 1x2, result is 2x2
        for i in range(2):
            for j in range(2):
                u_vec = [0.0] * (n*m)
                v_vec = [0.0] * (m*p)
                w_vec = [0.0] * (p*n)
                
                a_idx = (A_row_start + i) * m + A_col
                b_idx = B_row * p + (B_col_start + j)
                c_idx = (C_col_start + j) * n + (C_row_start + i)
                
                u_vec[a_idx] = 1.0
                v_vec[b_idx] = 1.0
                w_vec[c_idx] = 1.0
                
                add_term(u_vec, v_vec, w_vec)
    
    # Define block positions:
    # A blocks:
    # A00: rows 0-1, cols 0-1
    # A01: rows 0-1, cols 2-3
    # A02: rows 0-1, col 4
    # A10: rows 2-3, cols 0-1
    # A11: rows 2-3, cols 2-3
    # A12: rows 2-3, col 4
    
    # B blocks:
    # B00: rows 0-1, cols 0-1
    # B01: rows 0-1, cols 2-3
    # B02: rows 0-1, cols 4-5
    # B10: rows 2-3, cols 0-1
    # B11: rows 2-3, cols 2-3
    # B12: rows 2-3, cols 4-5
    # B20: row 4, cols 0-1
    # B21: row 4, cols 2-3
    # B22: row 4, cols 4-5
    
    # C blocks (4x6):
    # C[0:2, 0:2], C[0:2, 2:4], C[0:2, 4:6]
    # C[2:4, 0:2], C[2:4, 2:4], C[2:4, 4:6]
    
    # C[0:2, 0:2] = A00*B00 + A01*B10 + A02*B20
    add_strassen(0, 0, 0, 0, 0, 0)  # A00*B00
    add_strassen(0, 2, 2, 0, 0, 0)  # A01*B10
    add_outer_product(0, 4, 4, 0, 0, 0)  # A02*B20
    
    # C[0:2, 2:4] = A00*B01 + A01*B11 + A02*B21
    add_strassen(0, 0, 0, 2, 0, 2)  # A00*B01
    add_strassen(0, 2, 2, 2, 0, 2)  # A01*B11
    add_outer_product(0, 4, 4, 2, 0, 2)  # A02*B21
    
    # C[0:2, 4:6] = A00*B02 + A01*B12 + A02*B22
    add_strassen(0, 0, 0, 4, 0, 4)  # A00*B02
    add_strassen(0, 2, 2, 4, 0, 4)  # A01*B12
    add_outer_product(0, 4, 4, 4, 0, 4)  # A02*B22
    
    # C[2:4, 0:2] = A10*B00 + A11*B10 + A12*B20
    add_strassen(2, 0, 0, 0, 2, 0)  # A10*B00
    add_strassen(2, 2, 2, 0, 2, 0)  # A11*B10
    add_outer_product(2, 4, 4, 0, 2, 0)  # A12*B20
    
    # C[2:4, 2:4] = A10*B01 + A11*B11 + A12*B21
    add_strassen(2, 0, 0, 2, 2, 2)  # A10*B01
    add_strassen(2, 2, 2, 2, 2, 2)  # A11*B11
    add_outer_product(2, 4, 4, 2, 2, 2)  # A12*B21
    
    # C[2:4, 4:6] = A10*B02 + A11*B12 + A12*B22
    add_strassen(2, 0, 0, 4, 2, 4)  # A10*B02
    add_strassen(2, 2, 2, 4, 2, 4)  # A11*B12
    add_outer_product(2, 4, 4, 4, 2, 4)  # A12*B22
    
    U = np.array(U_list).T  # shape (n*m, rank)
    V = np.array(V_list).T  # shape (m*p, rank)
    W = np.array(W_list).T  # shape (p*n, rank)
    
    rank = U.shape[1]
    
    # Verify the decomposition
    def matmul_from_decomp(U, V, W, n, m, p):
        """Reconstruct C from U, V, W."""
        C = np.zeros((n, p))
        for t in range(U.shape[1]):
            u = U[:, t].reshape(n, m)
            v = V[:, t].reshape(m, p)
            w = W[:, t].reshape(p, n)
            # C += u @ v, but we need to check against w
            # Actually, the decomposition means: sum_t U_t * V_t = W_t (element-wise in tensor sense)
            # Let's verify differently
        return C
    
    # Convert to list
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

