import numpy as np

DIMENSIONS = [4, 4, 5]

def _strassen_decomp(n):
    if n == 1:
        U = np.array([[1]])
        V = np.array([[1]])
        W = np.array([[1]])
        return U, V, W
    
    m = n // 2
    U_rec, V_rec, W_rec = _strassen_decomp(m)
    R_rec = U_rec.shape[1]
    R = 7 * R_rec
    
    U = np.zeros((n*n, R))
    V = np.zeros((n*n, R))
    W = np.zeros((n*n, R))
    
    # Coefficients for Strassen
    # Blocks: 00, 01, 10, 11
    A_coeffs = [
        [1, 0, 0, 1], # P1: A00 + A11
        [0, 0, 1, 1], # P2: A10 + A11
        [1, 0, 0, 0], # P3: A00
        [0, 0, 0, 1], # P4: A11
        [1, 1, 0, 0], # P5: A00 + A01
        [-1, 0, 1, 0],# P6: A10 - A00
        [0, 1, 0, -1] # P7: A01 - A11
    ]
    B_coeffs = [
        [1, 0, 0, 1], # P1: B00 + B11
        [1, 0, 0, 0], # P2: B00
        [0, 0, 1, -1],# P3: B10 - B11
        [-1, 1, 0, 0],# P4: B01 - B00
        [0, 0, 0, 1], # P5: B11
        [1, 1, 0, 0], # P6: B00 + B01
        [0, 0, 1, 1]  # P7: B10 + B11
    ]
    # C_coeffs[block][k]
    C_coeffs = [
        [1, 0, 0, 1, -1, 0, 1], # C00: P1 + P4 - P5 + P7
        [0, 0, 1, 0, 1, 0, 0],  # C01: P3 + P5
        [0, 1, 0, 1, 0, 0, 0],  # C10: P2 + P4
        [1, -1, 1, 0, 0, 1, 0]  # C11: P1 - P2 + P3 + P6
    ]
    
    def get_U_index(block, i, j):
        if block == 0: return i * n + j
        if block == 1: return i * n + (m + j)
        if block == 2: return (m + i) * n + j
        if block == 3: return (m + i) * n + (m + j)
        
    def get_V_index(block, i, j):
        # V index is j * n + k (row j, col k)
        # In Strassen, V corresponds to B matrix.
        # B block indices: row i, col j.
        # So index is i * n + j.
        if block == 0: return i * n + j
        if block == 1: return i * n + (m + j)
        if block == 2: return (m + i) * n + j
        if block == 3: return (m + i) * n + (m + j)

    def get_W_index(block, i, k):
        # W index is k * n + i
        if block == 0: return k * n + i
        if block == 1: return (m + k) * n + i
        if block == 2: return k * n + (m + i)
        if block == 3: return (m + k) * n + (m + i)

    for k in range(7):
        col_offset = k * R_rec
        for c in range(R_rec):
            col = col_offset + c
            U_rec_col = U_rec[:, c]
            V_rec_col = V_rec[:, c]
            W_rec_col = W_rec[:, c]
            
            # Update U
            for b in range(4):
                coeff = A_coeffs[k][b]
                if coeff != 0:
                    for i in range(m):
                        for j in range(m):
                            row = get_U_index(b, i, j)
                            U[row, col] += coeff * U_rec[i * m + j, c]
                            
            # Update V
            for b in range(4):
                coeff = B_coeffs[k][b]
                if coeff != 0:
                    for i in range(m):
                        for j in range(m):
                            row = get_V_index(b, i, j)
                            V[row, col] += coeff * V_rec[i * m + j, c]
                            
            # Update W
            for b in range(4):
                coeff = C_coeffs[b][k]
                if coeff != 0:
                    for i in range(m):
                        for k_loc in range(m):
                            row = get_W_index(b, i, k_loc)
                            W[row, col] += coeff * W_rec[i * m + k_loc, c]
                            
    return U, V, W

def _mat_vec_decomp(n):
    # M_{n, n, 1}
    rank = n * n
    U = np.zeros((n*n, rank))
    V = np.zeros((n*1, rank))
    W = np.zeros((1*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(n):
            U[i * n + j, col] = 1
            V[j, col] = 1
            W[i, col] = 1
            col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=5
    
    # Part 1: 4x4x4 (Rank 49)
    U4, V4, W4 = _strassen_decomp(4)
    
    # Part 2: 4x4x1 (Rank 16)
    U1, V1, W1 = _mat_vec_decomp(4)
    
    R4 = U4.shape[1]
    R1 = U1.shape[1]
    R_total = R4 + R1
    
    U = np.zeros((n*m, R_total))
    V = np.zeros((m*p, R_total))
    W = np.zeros((p*n, R_total))
    
    # Copy U
    U[:, :R4] = U4
    U[:, R4:] = U1
    
    # Copy V
    # V4 corresponds to B cols 0..3.
    # V4 shape (16, 49). Index i * 4 + j.
    # V shape (20, R). Index i * 5 + k.
    # Map V4 row (i * 4 + j) to V row (i * 5 + j).
    for i in range(4):
        for j in range(4):
            V[i * 5 + j, :R4] = V4[i * 4 + j, :]
            
    # V1 corresponds to B col 4.
    # V1 shape (4, 16). Index i.
    # Map V1 row i to V row (i * 5 + 4).
    for i in range(4):
        V[i * 5 + 4, R4:] = V1[i, :]
        
    # Copy W
    # W4 corresponds to C cols 0..3.
    # W4 shape (16, 49). Index k * 4 + i.
    # W shape (20, R). Index k * 4 + i.
    # Rows 0..15 map directly.
    W[:16, :R4] = W4
    
    # W1 corresponds to C col 4.
    # W1 shape (4, 16). Index i.
    # Map W1 row i to W row (4 * 4 + i) = 16 + i.
    W[16:, R4:] = W1
    
    # Convert to lists
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
