"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=5, p=5
    
    terms = []  # list of (U_col, V_col, W_col) where each is a sparse dict {index: value}
    
    def add_term(U_idx, V_idx, W_idx, U_val=1, V_val=1, W_val=1):
        terms.append((U_idx, U_val, V_idx, V_val, W_idx, W_val))
    
    # Block structure:
    # A (2x5) = [A1 (2x3) | A2 (2x2)]
    # B (5x5) = [B11 (3x3)  B12 (3x2)]
    #           [B21 (2x3)  B22 (2x2)]
    # C (2x5) = [C11 (2x3)  C12 (2x2)]
    #
    # C11 = A1*B11 + A2*B21
    # C12 = A1*B12 + A2*B22
    #
    # Index conventions:
    # A[i,j] -> U index: i*m + j = i*5 + j
    # B[j,k] -> V index: j*p + k = j*5 + k
    # C[k,i] -> W index: k*n + i = k*2 + i
    #
    # A1: i in 0..1, j in 0..2
    # A2: i in 0..1, j in 3..4
    # B11: j in 0..2, k in 0..2
    # B12: j in 0..2, k in 3..4
    # B21: j in 3..4, k in 0..2
    # B22: j in 3..4, k in 3..4
    # C11: i in 0..1, k in 0..2
    # C12: i in 0..1, k in 3..4
    
    # --- Block 1: A1(2x3) * B11(3x3) -> C11(2x3) ---
    # Schoolbook for 2x3x3 would be 18 terms.
    # Use Strassen-like for 2x3x3: 
    # Split B11 into columns: B11a (3x1), B11b (3x2)
    # C11a = A1 * B11a (2x3x1): 6 terms
    # C11b = A1 * B11b (2x3x2): 8 terms (using 2x2x2 Strassen per row pair)
    # Actually for 2x3x2: split A1 into rows? No, A1 is 2x3.
    # 2x3x2: rank 8 via: for each of 2 output cols, 2x3x1 = 6, but that's 12.
    # Better: 2x3x2. We can do: M = A1 * [B11b] where B11b is 3x2.
    # Split: for k in {3,4}: C[:,k] = sum_{j=0}^{2} A1[:,j] * B11b[j,k]
    # That's 6 terms per k, 12 total. Not great.
    # Alternative: use the known 2x3x2 = 8.
    # Let me just use schoolbook for simplicity first, then optimize.
    
    # For 2x3x3: use decomposition
    # Split B11 (3x3) into B11[:,0] (3x1), B11[:,1:3] (3x2)
    # A1 * B11[:,0]: 2x3x1 = 6 terms (schoolbook is optimal for x1)
    # A1 * B11[:,1:3]: 2x3x2
    # For 2x3x2: rank 8
    #   Split A1 (2x3) into A1[:,0:2] (2x2), A1[:,2] (2x1)
    #   A1[:,0:2] * B11[:,1:3] (2x2x2): rank 4 (Strassen... no, 2x2x2 rank is 4? No, Strassen is 7 for 3x3x3)
    #   Wait, 2x2x2 schoolbook is 4, and that's optimal.
    #   A1[:,2] * B11[:,1:3] (2x1x2): 4 terms
    #   Total: 4 + 4 = 8. 
    # So 2x3x3: 6 + 8 = 14. 
    
    # Block 1a: A1 * B11[:,0] -> C11[:,0] (2x3x1)
    for i in range(2):
        for j in range(3):
            add_term(i*5 + j, (j*5 + 0), 0*2 + i)
    
    # Block 1b: A1 * B11[:,1:3] -> C11[:,1:3] (2x3x2)
    # Sub-part: A1[:,0:2] * B11[:,1:3] (2x2x2) -> C11[:,1:3]
    for i in range(2):
        for j in range(2):  # j = 0,1 (local)
            for k in range(2):  # k = 0,1 (local) -> global k = 1+k
                add_term(i*5 + j, (j*5 + (1+k)), (1+k)*2 + i)
    
    # Sub-part: A1[:,2] * B11[:,1:3] (2x1x2) -> C11[:,1:3]
    for i in range(2):
        for k in range(2):  # global k = 1+k
            add_term(i*5 + 2, (2*5 + (1+k)), (1+k)*2 + i)
    
    # --- Block 2: A2(2x2) * B21(2x3) -> C11(2x3) ---
    # 2x2x3: rank 8
    # Split B21 (2x3) into B21[:,0] (2x1), B21[:,1:3] (2x2)
    # A2 * B21[:,0] (2x2x1): 4 terms
    # A2 * B21[:,1:3] (2x2x2): 4 terms
    # Total: 8
    
    # Block 2a: A2 * B21[:,0] -> C11[:,0] (2x2x1)
    for i in range(2):
        for j in range(2):  # local j -> global j = 3+j
            add_term(i*5 + (3+j), ((3+j)*5 + 0), 0*2 + i)
    
    # Block 2b: A2 * B21[:,1:3] -> C11[:,1:3] (2x2x2)
    for i in range(2):
        for j in range(2):  # local j -> global j = 3+j
            for k in range(2):  # local k -> global k = 1+k
                add_term(i*5 + (3+j), ((3+j)*5 + (1+k)), (1+k)*2 + i)
    
    # --- Block 3: A1(2x3) * B12(3x2) -> C12(2x2) ---
    # 2x3x2: rank 8
    # Split A1 (2x3) into A1[:,0:2] (2x2), A1[:,2] (2x1)
    # A1[:,0:2] * B12 (2x2x2): 4 terms
    # A1[:,2] * B12 (2x1x2): 4 terms
    # Total: 8
    
    # Block 3a: A1[:,0:2] * B12 -> C12 (2x2x2)
    for i in range(2):
        for j in range(2):  # local j -> global j = j
            for k in range(2):  # local k -> global k = 3+k
                add_term(i*5 + j, (j*5 + (3+k)), (3+k)*2 + i)
    
    # Block 3b: A1[:,2] * B12 -> C12 (2x1x2)
    for i in range(2):
        for k in range(2):  # global k = 3+k
            add_term(i*5 + 2, (2*5 + (3+k)), (3+k)*2 + i)
    
    # --- Block 4: A2(2x2) * B22(2x2) -> C12(2x2) ---
    # 2x2x2: rank 4 (schoolbook)
    for i in range(2):
        for j in range(2):  # local j -> global j = 3+j
            for k in range(2):  # local k -> global k = 3+k
                add_term(i*5 + (3+j), ((3+j)*5 + (3+k)), (3+k)*2 + i)
    
    # Count total terms
    rank = len(terms)
    
    # Build U, V, W matrices
    U_rows = n * m  # 10
    V_rows = m * p  # 25
    W_rows = p * n  # 10
    
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    for c, (U_idx, U_val, V_idx, V_val, W_idx, W_val) in enumerate(terms):
        U[U_idx][c] = U_val
        V[V_idx][c] = V_val
        W[W_idx][c] = W_val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
