"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will build a decomposition of rank 54.
    # Split m=5 into 1 and 4.
    # Part 1: M_{2x1x6} (Rank 12)
    # Part 2: M_{2x4x6} (Rank 42)
    #   Split p=6 into 4 and 2.
    #   Part 2a: M_{2x4x4} (Rank 28)
    #     Split m=4 into 2 and 2.
    #     Part 2a1: M_{2x2x4} (Rank 14)
    #       Split p=4 into 2 and 2.
    #       Part 2a11: M_{2x2x2} (Rank 7)
    #       Part 2a12: M_{2x2x2} (Rank 7)
    #     Part 2a2: M_{2x2x4} (Rank 14)
    #       Split p=4 into 2 and 2.
    #       Part 2a21: M_{2x2x2} (Rank 7)
    #       Part 2a22: M_{2x2x2} (Rank 7)
    #   Part 2b: M_{2x4x2} (Rank 14)
    #     Split m=4 into 2 and 2.
    #     Part 2b1: M_{2x2x2} (Rank 7)
    #     Part 2b2: M_{2x2x2} (Rank 7)
    
    total_rank = 12 + 28 + 14
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val
        col += 1

    # Helper to add a Strassen block M_{2x2x2} at offsets i0, j0, k0
    def add_strassen(i0, j0, k0):
        # Strassen products for 2x2 matrices
        # A is 2x2, B is 2x2
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We add 7 terms.
        # Term 1: M1
        # U: A00=1, A11=1. V: B00=1, B11=1. W: C00=1, C11=1.
        add_term(i0*m + j0, j0*p + k0, k0*n + i0, 1, 1, 1)
        add_term(i0*m + (j0+1), (j0+1)*p + k0, k0*n + i0, 0, 0, 0) # Placeholder, need to handle sums correctly
        
        # Actually, it's easier to just define the vectors for each of the 7 products.
        # The "add_term" above is for a single rank-1 component.
        # A Strassen product like M1 is a single rank-1 component in the tensor decomposition?
        # Yes, because it is (linear combo of A) * (linear combo of B) -> (linear combo of C).
        # So for each of the 7 products, we add ONE column to U, V, W.
        
        # Let's restart the helper.
        pass

    # Correct Helper
    def add_strassen_block(i0, j0, k0):
        nonlocal col
        
        # Product 1: (A00+A11)(B00+B11) -> C00, C11
        # U: A00=1, A11=1
        # V: B00=1, B11=1
        # W: C00=1, C11=1
        U[i0*m + j0][col] = 1
        U[(i0+1)*m + (j0+1)][col] = 1
        V[j0*p + k0][col] = 1
        V[(j0+1)*p + (k0+1)][col] = 1
        W[k0*n + i0][col] = 1
        W[(k0+1)*n + (i0+1)][col] = 1
        col += 1
        
        # Product 2: (A10+A11)B00 -> C10
        U[(i0+1)*m + j0][col] = 1
        U[(i0+1)*m + (j0+1)][col] = 1
        V[j0*p + k0][col] = 1
        W[k0*n + (i0+1)][col] = 1
        col += 1
        
        # Product 3: A00(B01-B11) -> C01
        U[i0*m + j0][col] = 1
        V[j0*p + (k0+1)][col] = 1
        V[(j0+1)*p + (k0+1)][col] = -1
        W[(k0+1)*n + i0][col] = 1
        col += 1
        
        # Product 4: A11(B10-B00) -> C00, C10
        U[(i0+1)*m + (j0+1)][col] = 1
        V[(j0+1)*p + k0][col] = 1
        V[j0*p + k0][col] = -1
        W[k0*n + i0][col] = 1
        W[k0*n + (i0+1)][col] = 1
        col += 1
        
        # Product 5: (A00+A01)B11 -> C00, C01
        U[i0*m + j0][col] = 1
        U[i0*m + (j0+1)][col] = 1
        V[(j0+1)*p + (k0+1)][col] = 1
        W[k0*n + i0][col] = -1 # C00 gets -M5
        W[(k0+1)*n + i0][col] = 1 # C01 gets M5
        col += 1
        
        # Product 6: (A10-A00)(B00+B01) -> C11
        U[(i0+1)*m + j0][col] = 1
        U[i0*m + j0][col] = -1
        V[j0*p + k0][col] = 1
        V[j0*p + (k0+1)][col] = 1
        W[(k0+1)*n + (i0+1)][col] = 1
        col += 1
        
        # Product 7: (A01-A11)(B10+B11) -> C00, C11
        U[i0*m + (j0+1)][col] = 1
        U[(i0+1)*m + (j0+1)][col] = -1
        V[(j0+1)*p + k0][col] = 1
        V[(j0+1)*p + (k0+1)][col] = 1
        W[k0*n + i0][col] = 1
        W[(k0+1)*n + (i0+1)][col] = -1
        col += 1

    # Part 1: M_{2x1x6}
    # A col 0, B row 0.
    # i in 0..1, j=0, k in 0..5
    for i in range(2):
        for k in range(6):
            U[i*m + 0][col] = 1
            V[0*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1
            
    # Part 2: M_{2x4x6}
    # A cols 1..4, B rows 1..4.
    # Split p=6 into 4 (k=0..3) and 2 (k=4..5).
    
    # Part 2a: M_{2x4x4} (k=0..3)
    # Split m=4 into 2 (j=1,2) and 2 (j=3,4).
    
    # Part 2a1: M_{2x2x4} (j=1,2; k=0..3)
    # Split p=4 into 2 (k=0,1) and 2 (k=2,3).
    add_strassen_block(0, 1, 0) # i=0..1, j=1..2, k=0..1
    add_strassen_block(0, 1, 2) # i=0..1, j=1..2, k=2..3
    
    # Part 2a2: M_{2x2x4} (j=3,4; k=0..3)
    # Split p=4 into 2 (k=0,1) and 2 (k=2,3).
    add_strassen_block(0, 3, 0) # i=0..1, j=3..4, k=0..1
    add_strassen_block(0, 3, 2) # i=0..1, j=3..4, k=2..3
    
    # Part 2b: M_{2x4x2} (k=4..5)
    # Split m=4 into 2 (j=1,2) and 2 (j=3,4).
    
    # Part 2b1: M_{2x2x2} (j=1,2; k=4..5)
    add_strassen_block(0, 1, 4)
    
    # Part 2b2: M_{2x2x2} (j=3,4; k=4..5)
    add_strassen_block(0, 3, 4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
