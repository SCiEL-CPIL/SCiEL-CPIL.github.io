import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 decomposition using Strassen for 4x4 blocks and Strassen for 2x2 blocks
    rank = 49
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to add a linear form to a vector
    # Indices for A (i, j) -> i * 4 + j
    # Indices for B (j, k) -> j * 4 + k
    # Indices for C (k, i) -> k * 4 + i (based on W[k*n+i])
    
    # Strassen for 4x4 (Block level)
    # Blocks: 00, 01, 10, 11
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B01, B10, B11
    
    # We define the 7 block products S0..S6
    # Each Sk = Ak * Bk where Ak, Bk are 2x2 matrices of linear forms
    
    # Linear forms for A blocks
    # A00: indices 0,1,4,5 (i=0,1; j=0,1)
    # A01: indices 2,3,6,7 (i=0,1; j=2,3)
    # A10: indices 8,9,12,13 (i=2,3; j=0,1)
    # A11: indices 10,11,14,15 (i=2,3; j=2,3)
    
    # Linear forms for B blocks
    # B00: indices 0,4,1,5? No.
    # B is j*p+k. j=0..3, k=0..3.
    # B00 (j=0,1; k=0,1): indices 0, 1, 4, 5?
    # j=0, k=0 -> 0
    # j=0, k=1 -> 1
    # j=1, k=0 -> 4
    # j=1, k=1 -> 5
    # B01 (j=0,1; k=2,3): indices 2, 3, 6, 7
    # B10 (j=2,3; k=0,1): indices 8, 9, 12, 13
    # B11 (j=2,3; k=2,3): indices 10, 11, 14, 15
    
    # Let's create a function to get the 2x2 block of A or B as a matrix of linear forms (vectors of length 16)
    def get_A_block(i, j):
        # i, j in {0, 1}
        # Returns 2x2 matrix of vectors (16,)
        mat = np.zeros((2, 2, 16))
        for a in range(2):
            for b in range(2):
                idx_i = i * 2 + a
                idx_j = j * 2 + b
                vec = np.zeros(16)
                vec[idx_i * 4 + idx_j] = 1
                mat[a, b] = vec
        return mat

    def get_B_block(j, k):
        # j, k in {0, 1}
        # Returns 2x2 matrix of vectors (16,)
        mat = np.zeros((2, 2, 16))
        for b in range(2): # local row of block (corresponds to j offset)
            for c in range(2): # local col of block (corresponds to k offset)
                idx_j = j * 2 + b
                idx_k = k * 2 + c
                vec = np.zeros(16)
                vec[idx_j * 4 + idx_k] = 1
                mat[b, c] = vec
        return mat

    A00 = get_A_block(0, 0)
    A01 = get_A_block(0, 1)
    A10 = get_A_block(1, 0)
    A11 = get_A_block(1, 1)
    
    B00 = get_B_block(0, 0)
    B01 = get_B_block(0, 1)
    B10 = get_B_block(1, 0)
    B11 = get_B_block(1, 1)
    
    # Define the 7 Strassen terms for the 4x4 block multiplication
    # S0 = A00 * (B00 - B11)
    # S1 = (A00 + A01) * B11
    # S2 = (A10 + A11) * B00
    # S3 = A11 * (B10 - B00)
    # S4 = (A00 + A11) * (B00 + B11)
    # S5 = (A01 - A11) * (B10 + B11)
    # S6 = (A00 - A10) * (B00 + B01)
    
    block_terms = []
    block_terms.append((A00, B00 - B11)) # S0
    block_terms.append((A00 + A01, B11)) # S1
    block_terms.append((A10 + A11, B00)) # S2
    block_terms.append((A11, B10 - B00)) # S3
    block_terms.append((A00 + A11, B00 + B11)) # S4
    block_terms.append((A01 - A11, B10 + B11)) # S5
    block_terms.append((A00 - A10, B00 + B01)) # S6
    
    # Reconstruction coefficients for C blocks from S terms
    # C00 = S4 + S3 - S1 + S6
    # C01 = S0 + S1
    # C10 = S2 + S3
    # C11 = S4 + S0 - S2 + S5
    
    # c[I, J, k] where I, J in {0,1} are block indices of C, k is term index 0..6
    c_coeff = np.zeros((2, 2, 7))
    c_coeff[0, 0] = [0, -1, 0, 1, 1, 0, 1] # S4, S3, S1, S6
    c_coeff[0, 1] = [1, 1, 0, 0, 0, 0, 0]   # S0, S1
    c_coeff[1, 0] = [0, 0, 1, 1, 0, 0, 0]   # S2, S3
    c_coeff[1, 1] = [1, 0, -1, 0, 1, 1, 0]  # S4, S0, S2, S5
    
    # Strassen for 2x2
    # Inputs X, Y (2x2 matrices of vectors)
    # Returns list of 7 tuples (u_vec, v_vec, w_coeffs)
    # w_coeffs is 2x2 matrix of coefficients for the 4 outputs
    
    def strassen_2x2(X, Y):
        # X, Y are (2, 2, 16)
        # M0 = X00 * (Y00 - Y11)
        # M1 = (X00 + X01) * Y11
        # M2 = (X10 + X11) * Y00
        # M3 = X11 * (Y10 - Y00)
        # M4 = (X00 + X11) * (Y00 + Y11)
        # M5 = (X01 - X11) * (Y10 + Y11)
        # M6 = (X00 - X10) * (Y00 + Y01)
        
        terms = []
        terms.append((X[0,0], Y[0,0] - Y[1,1])) # M0
        terms.append((X[0,0] + X[0,1], Y[1,1])) # M1
        terms.append((X[1,0] + X[1,1], Y[0,0])) # M2
        terms.append((X[1,1], Y[1,0] - Y[0,0])) # M3
        terms.append((X[0,0] + X[1,1], Y[0,0] + Y[1,1])) # M4
        terms.append((X[0,1] - X[1,1], Y[1,0] + Y[1,1])) # M5
        terms.append((X[0,0] - X[1,0], Y[0,0] + Y[0,1])) # M6
        
        # Reconstruction
        # Z00 = M4 + M3 - M1 + M6
        # Z01 = M0 + M1
        # Z10 = M2 + M3
        # Z11 = M4 + M0 - M2 + M5
        
        # w_coeffs[t, a, b]
        w_coeffs = np.zeros((7, 2, 2))
        w_coeffs[0] = [[1, 1], [0, 1]] # M0: Z00+=1, Z01+=1, Z11+=1
        w_coeffs[1] = [[-1, 1], [0, 0]] # M1: Z00+=-1, Z01+=1
        w_coeffs[2] = [[0, 0], [1, -1]] # M2: Z10+=1, Z11+=-1
        w_coeffs[3] = [[1, 0], [1, 0]]  # M3: Z00+=1, Z10+=1
        w_coeffs[4] = [[1, 0], [0, 1]]  # M4: Z00+=1, Z11+=1
        w_coeffs[5] = [[0, 0], [0, 1]]  # M5: Z11+=1
        w_coeffs[6] = [[1, 0], [0, 0]]  # M6: Z00+=1
        
        return terms, w_coeffs

    col = 0
    for k in range(7):
        Ak, Bk = block_terms[k]
        terms, w_coeffs = strassen_2x2(Ak, Bk)
        for t in range(7):
            u_vec, v_vec = terms[t]
            U[:, col] = u_vec
            V[:, col] = v_vec
            
            # Compute W contribution
            W_col = np.zeros(16)
            for I in range(2):
                for J in range(2):
                    c_val = c_coeff[I, J, k]
                    if c_val == 0:
                        continue
                    for a in range(2):
                        for b in range(2):
                            w_val = w_coeffs[t, a, b]
                            if w_val == 0:
                                continue
                            # Output index in W: k_out * 4 + i_out
                            # i_out = I * 2 + a
                            # k_out = J * 2 + b
                            i_out = I * 2 + a
                            k_out = J * 2 + b
                            w_idx = k_out * 4 + i_out
                            W_col[w_idx] += c_val * w_val
            W[:, col] = W_col
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
