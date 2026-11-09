"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank calculation:
    # A1 (2x1) * B1 (1x7) -> 14 terms
    # A2 (2x2) * B2 (2x7)
    #   B2 split into 3 (2x2) blocks and 1 (2x1) block
    #   3 * 7 (Strassen) + 4 (Schoolbook) = 21 + 4 = 25 terms
    # Total rank = 14 + 25 = 39
    rank = 39
    
    U = [[0] * rank for _ in range(n * m)] # 6 x 39
    V = [[0] * rank for _ in range(m * p)] # 21 x 39
    W = [[0] * rank for _ in range(p * n)] # 14 x 39
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val
        col += 1

    # Part 1: A1 * B1
    # A1 is A[:, 0] (rows 0,1 of U are indices 0, 1)
    # B1 is B[0, :] (cols 0..6 of V are indices 0..6)
    # Output C[:, 0] (rows 0..6 of W? No, W is k*n+i. i=0,1. k=0..6)
    # W index for C[k, 0] is k*2 + 0 = 2k
    # W index for C[k, 1] is k*2 + 1 = 2k + 1
    
    for k in range(p): # 0..6
        # Term A[0,0] * B[0,k] -> C[k,0]
        add_term(0, 1, k, 1, 2*k, 1)
        # Term A[1,0] * B[0,k] -> C[k,1]
        add_term(1, 1, k, 1, 2*k + 1, 1)
        
    # Part 2: A2 * B2
    # A2 is A[:, 1:3]
    # U indices:
    # A[0,1] -> 0*3+1 = 1
    # A[0,2] -> 0*3+2 = 2
    # A[1,1] -> 1*3+1 = 4
    # A[1,2] -> 1*3+2 = 5
    
    u_00 = 1
    u_01 = 2
    u_10 = 4
    u_11 = 5
    
    # B2 is B[1:3, :]
    # V indices for row 1 (j=1): 7+k
    # V indices for row 2 (j=2): 14+k
    
    # Blocks of B2:
    # Block 1: k=0,1
    # Block 2: k=2,3
    # Block 3: k=4,5
    # Block 4: k=6
    
    def add_strassen_block(k1, k2):
        nonlocal col
        # V indices
        v_00 = 7 + k1
        v_01 = 7 + k2
        v_10 = 14 + k1
        v_11 = 14 + k2
        
        # W indices
        w_00 = 2 * k1 + 0 # C[k1, 0]
        w_01 = 2 * k2 + 0 # C[k2, 0]
        w_10 = 2 * k1 + 1 # C[k1, 1]
        w_11 = 2 * k2 + 1 # C[k2, 1]
        
        # M1: (A00+A11)(B00+B11)
        # U: u_00+1, u_11+1
        # V: v_00+1, v_11+1
        # W: w_00+1, w_11+1
        # We need to represent this as a single rank-1 term? No, Strassen uses 7 multiplications.
        # Each multiplication is a rank-1 term in the tensor decomposition.
        # The "multiplication" is the column.
        # So for M1, we create one column where U has coeffs for A, V has coeffs for B, W has coeffs for C.
        
        # M1
        U[u_00][col] = 1
        U[u_11][col] = 1
        V[v_00][col] = 1
        V[v_11][col] = 1
        W[w_00][col] = 1
        W[w_11][col] = 1
        col += 1
        
        # M2: (A10+A11)B00
        # U: u_10+1, u_11+1
        # V: v_00+1
        # W: w_10+1, w_11-1
        U[u_10][col] = 1
        U[u_11][col] = 1
        V[v_00][col] = 1
        W[w_10][col] = 1
        W[w_11][col] = -1
        col += 1
        
        # M3: A00(B10-B11)
        # U: u_00+1
        # V: v_10+1, v_11-1
        # W: w_01+1, w_10+1
        U[u_00][col] = 1
        V[v_10][col] = 1
        V[v_11][col] = -1
        W[w_01][col] = 1
        W[w_10][col] = 1
        col += 1
        
        # M4: A11(B01-B00)
        # U: u_11+1
        # V: v_01+1, v_00-1
        # W: w_00+1, w_10+1
        U[u_11][col] = 1
        V[v_01][col] = 1
        V[v_00][col] = -1
        W[w_00][col] = 1
        W[w_10][col] = 1
        col += 1
        
        # M5: (A00+A01)B11
        # U: u_00+1, u_01+1
        # V: v_11+1
        # W: w_00-1, w_01+1
        U[u_00][col] = 1
        U[u_01][col] = 1
        V[v_11][col] = 1
        W[w_00][col] = -1
        W[w_01][col] = 1
        col += 1
        
        # M6: (A10-A00)(B00+B01)
        # U: u_10+1, u_00-1
        # V: v_00+1, v_01+1
        # W: w_11+1
        U[u_10][col] = 1
        U[u_00][col] = -1
        V[v_00][col] = 1
        V[v_01][col] = 1
        W[w_11][col] = 1
        col += 1
        
        # M7: (A01-A11)(B10+B11)
        # U: u_01+1, u_11-1
        # V: v_10+1, v_11+1
        # W: w_00+1
        U[u_01][col] = 1
        U[u_11][col] = -1
        V[v_10][col] = 1
        V[v_11][col] = 1
        W[w_00][col] = 1
        col += 1

    add_strassen_block(0, 1)
    add_strassen_block(2, 3)
    add_strassen_block(4, 5)
    
    # Part 3: A2 * B24 (k=6)
    # B24 is B[1:3, 6]
    # v_00 = 7+6 = 13
    # v_10 = 14+6 = 20
    # W indices: w_00 = 2*6+0 = 12, w_10 = 2*6+1 = 13
    
    v_00 = 13
    v_10 = 20
    w_00 = 12
    w_10 = 13
    
    # Term 1: A00 * B00 -> C00
    U[u_00][col] = 1
    V[v_00][col] = 1
    W[w_00][col] = 1
    col += 1
    
    # Term 2: A01 * B10 -> C00
    U[u_01][col] = 1
    V[v_10][col] = 1
    W[w_00][col] = 1
    col += 1
    
    # Term 3: A10 * B00 -> C10
    U[u_10][col] = 1
    V[v_00][col] = 1
    W[w_10][col] = 1
    col += 1
    
    # Term 4: A11 * B10 -> C10
    U[u_11][col] = 1
    V[v_10][col] = 1
    W[w_10][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
