"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _strassen_2x2(A, B):
    """Strassen's 7-multiplication 2x2 matrix multiply. Returns 7 intermediate products."""
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    a, b = A[0,0], A[0,1]
    c, d = A[1,0], A[1,1]
    e, f = B[0,0], B[0,1]
    g, h = B[1,0], B[1,1]
    
    m1 = (a+d)*(e+h)
    m2 = (c+d)*e
    m3 = a*(f-h)
    m4 = d*(g-e)
    m5 = (a+b)*h
    m6 = (c-a)*(e+f)
    m7 = (b-d)*(g+h)
    
    return [m1, m2, m3, m4, m5, m6, m7]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (16, rank)  
    # W: (p*n, rank) = (8, rank)
    
    # Try to find a low-rank decomposition
    # Start with schoolbook rank 32, then try to reduce
    
    # Let's try rank 28 using Strassen blocks
    # A is 2x4, B is 4x4, C is 2x4
    # Partition: A = [A_left | A_right] each 2x2
    # B = [[B11, B12], [B21, B22]] each 2x2
    # C_left = A_left * B11 + A_right * B21
    # C_right = A_left * B12 + A_right * B22
    
    # For each 2x2*2x2 product, use Strassen (rank 7)
    # Total: 4 * 7 = 28
    
    rank = 28
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Indexing:
    # A[i, j] where i in [0,2), j in [0,4)
    # B[j, k] where j in [0,4), k in [0,4)
    # C[i, k] where i in [0,2), k in [0,4)
    # W index: k*n + i = k*2 + i
    
    # A_left: A[0:2, 0:2], A_right: A[0:2, 2:4]
    # B11: B[0:2, 0:2], B12: B[0:2, 2:4], B21: B[2:4, 0:2], B22: B[2:4, 2:4]
    
    # For each Strassen product, we assign 7 columns
    
    def add_strassen_block(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols, col_offset):
        """
        A_rows, A_cols: row/col indices of the 2x2 A block
        B_rows, B_cols: row/col indices of the 2x2 B block  
        C_rows, C_cols: row/col indices of the 2x2 C block
        col_offset: starting column index
        """
        # A block elements: A[i, j] for i in A_rows, j in A_cols
        # B block elements: B[j, k] for j in B_rows, k in B_cols
        # C block elements: C[i, k] for i in C_rows, k in C_cols
        
        a00 = (A_rows[0], A_cols[0])  # A[i0, j0]
        a01 = (A_rows[0], A_cols[1])  # A[i0, j1]
        a10 = (A_rows[1], A_cols[0])  # A[i1, j0]
        a11 = (A_rows[1], A_cols[1])  # A[i1, j1]
        
        b00 = (B_rows[0], B_cols[0])  # B[j0, k0]
        b01 = (B_rows[0], B_cols[1])  # B[j0, k1]
        b10 = (B_rows[1], B_cols[0])  # B[j1, k0]
        b11 = (B_rows[1], B_cols[1])  # B[j1, k1]
        
        c00 = (C_rows[0], C_cols[0])  # C[i0, k0]
        c01 = (C_rows[0], C_cols[1])  # C[i0, k1]
        c10 = (C_rows[1], C_cols[0])  # C[i1, k0]
        c11 = (C_rows[1], C_cols[1])  # C[i1, k1]
        
        # Strassen intermediates:
        # m1 = (a00+a11)*(b00+b11)
        # m2 = (a10+a11)*b00
        # m3 = a00*(b01-b11)
        # m4 = a11*(b10-b00)
        # m5 = (a00+a01)*b11
        # m6 = (a10-a00)*(b00+b01)
        # m7 = (a01-a11)*(b10+b11)
        
        # m1 = (a00+a11)*(b00+b11)
        col = col_offset + 0
        U[a00[0]*m + a00[1], col] += 1
        U[a11[0]*m + a11[1], col] += 1
        V[b00[0]*p + b00[1], col] += 1
        V[b11[0]*p + b11[1], col] += 1
        # C contribution: c00 += m1, c01 -= m3, c10 -= m2, c11 += m1 + m2 + m3 - m5 - m7
        # Actually: C00 = m1 - m3 - m5 + m7, C01 = m4 + m5, C10 = m2 + m3, C11 = m1 - m2 + m4 - m6
        W[c00[0]*n + c00[1], col] += 1  # m1 contributes to c00
        W[c11[0]*n + c11[1], col] += 1  # m1 contributes to c11
        
        # m2 = (a10+a11)*b00
        col = col_offset + 1
        U[a10[0]*m + a10[1], col] += 1
        U[a11[0]*m + a11[1], col] += 1
        V[b00[0]*p + b00[1], col] += 1
        W[c10[0]*n + c10[1], col] += 1  # m2 contributes to c10
        W[c11[0]*n + c11[1], col] += -1  # m2 contributes negatively to c11
        
        # m3 = a00*(b01-b11)
        col = col_offset + 2
        U[a00[0]*m + a00[1], col] += 1
        V[b01[0]*p + b01[1], col] += 1
        V[b11[0]*p + b11[1], col] += -1
        W[c00[0]*n + c00[1], col] += -1  # m3 contributes negatively to c00
        W[c10[0]*n + c10[1], col] += 1  # m3 contributes to c10
        
        # m4 = a11*(b10-b00)
        col = col_offset + 3
        U[a11[0]*m + a11[1], col] += 1
        V[b10[0]*p + b10[1], col] += 1
        V[b00[0]*p + b00[1], col] += -1
        W[c01[0]*n + c01[1], col] += 1  # m4 contributes to c01
        W[c11[0]*n + c11[1], col] += 1  # m4 contributes to c11
        
        # m5 = (a00+a01)*b11
        col = col_offset + 4
        U[a00[0]*m + a00[1], col] += 1
        U[a01[0]*m + a01[1], col] += 1
        V[b11[0]*p + b11[1], col] += 1
        W[c00[0]*n + c00[1], col] += -1  # m5 contributes negatively to c00
        W[c01[0]*n + c01[1], col] += 1  # m5 contributes to c01
        
        # m6 = (a10-a00)*(b00+b01)
        col = col_offset + 5
        U[a10[0]*m + a10[1], col] += 1
        U[a00[0]*m + a00[1], col] += -1
        V[b00[0]*p + b00[1], col] += 1
        V[b01[0]*p + b01[1], col] += 1
        W[c11[0]*n + c11[1], col] += -1  # m6 contributes negatively to c11
        
        # m7 = (a01-a11)*(b10+b11)
        col = col_offset + 6
        U[a01[0]*m + a01[1], col] += 1
        U[a11[0]*m + a11[1], col] += -1
        V[b10[0]*p + b10[1], col] += 1
        V[b11[0]*p + b11[1], col] += 1
        W[c00[0]*n + c00[1], col] += 1  # m7 contributes to c00
        # m7 does not directly contribute to other C entries in standard Strassen
        
        return col_offset + 7
    
    # Block 1: A_left * B11 -> C_left_part1
    # A_left: rows [0,1], cols [0,1]
    # B11: rows [0,1], cols [0,1]
    # C_left_part1: rows [0,1], cols [0,1]
    next_col = add_strassen_block([0,1], [0,1], [0,1], [0,1], [0,1], [0,1], 0)
    
    # Block 2: A_right * B21 -> C_left_part2
    # A_right: rows [0,1], cols [2,3]
    # B21: rows [2,3], cols [0,1]
    # C_left_part2: rows [0,1], cols [0,1]
    next_col = add_strassen_block([0,1], [2,3], [2,3], [0,1], [0,1], [0,1], next_col)
    
    # Block 3: A_left * B12 -> C_right_part1
    # A_left: rows [0,1], cols [0,1]
    # B12: rows [0,1], cols [2,3]
    # C_right_part1: rows [0,1], cols [2,3]
    next_col = add_strassen_block([0,1], [0,1], [0,1], [2,3], [0,1], [2,3], next_col)
    
    # Block 4: A_right * B22 -> C_right_part2
    # A_right: rows [0,1], cols [2,3]
    # B22: rows [2,3], cols [2,3]
    # C_right_part2: rows [0,1], cols [2,3]
    next_col = add_strassen_block([0,1], [2,3], [2,3], [2,3], [0,1], [2,3], next_col)
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(U.shape[1]):
                        total += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(total - 1.0) > 1e-10:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        rank = n*m*p
    else:
        rank = next_col
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
