import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 36
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_222(i_base, j_base, k_base, offset):
        nonlocal col
        # i_base: 0 or 1 (row in A)
        # j_base: 0 or 2 (col in A)
        # k_base: 0 or 2 (col in B) - wait, B cols are 0..4. 
        # The block is 2x2x2. 
        # A indices: i_base, i_base+1 for rows; j_base, j_base+1 for cols.
        # B indices: j_base, j_base+1 for rows; k_base, k_base+1 for cols.
        
        # P1 = (A00 + A11)(B00 + B11)
        U[i_base*m + j_base][col] += 1
        U[(i_base+1)*m + (j_base+1)][col] += 1
        V[j_base*p + k_base][col] += 1
        V[(j_base+1)*p + (k_base+1)][col] += 1
        W[k_base*n + i_base][col] += 1
        W[(k_base+1)*n + (i_base+1)][col] += 1
        col += 1
        
        # P2 = (A10 + A11)B00
        U[(i_base+1)*m + j_base][col] += 1
        U[(i_base+1)*m + (j_base+1)][col] += 1
        V[j_base*p + k_base][col] += 1
        W[k_base*n + (i_base+1)][col] += 1
        col += 1
        
        # P3 = A00(B01 - B11)
        U[i_base*m + j_base][col] += 1
        V[j_base*p + (k_base+1)][col] += 1
        V[(j_base+1)*p + (k_base+1)][col] += -1
        W[(k_base+1)*n + i_base][col] += 1
        col += 1
        
        # P4 = A11(B10 - B00)
        U[(i_base+1)*m + (j_base+1)][col] += 1
        V[(j_base+1)*p + k_base][col] += 1
        V[j_base*p + k_base][col] += -1
        W[k_base*n + (i_base+1)][col] += 1
        col += 1
        
        # P5 = (A00 + A01)B11
        U[i_base*m + j_base][col] += 1
        U[i_base*m + (j_base+1)][col] += 1
        V[(j_base+1)*p + (k_base+1)][col] += 1
        W[(k_base+1)*n + i_base][col] += 1
        col += 1
        
        # P6 = (A10 - A00)(B00 + B01)
        U[(i_base+1)*m + j_base][col] += 1
        U[i_base*m + j_base][col] += -1
        V[j_base*p + k_base][col] += 1
        V[j_base*p + (k_base+1)][col] += 1
        W[k_base*n + (i_base+1)][col] += 1
        W[(k_base+1)*n + (i_base+1)][col] += 1
        col += 1
        
        # P7 = (A01 - A11)(B10 + B11)
        U[i_base*m + (j_base+1)][col] += 1
        U[(i_base+1)*m + (j_base+1)][col] += -1
        V[(j_base+1)*p + k_base][col] += 1
        V[(j_base+1)*p + (k_base+1)][col] += 1
        W[k_base*n + i_base][col] += 1
        W[(k_base+1)*n + i_base][col] += 1
        col += 1

    def add_strassen_221(i_base, j_base, k_base, offset):
        nonlocal col
        # 2x2x1 multiplication
        # A is 2x2, B is 2x1.
        # C is 2x1.
        # C0 = A00*B0 + A01*B1
        # C1 = A10*B0 + A11*B1
        # This is 4 products.
        
        # P1 = A00 * B0
        U[i_base*m + j_base][col] += 1
        V[j_base*p + k_base][col] += 1
        W[k_base*n + i_base][col] += 1
        col += 1
        
        # P2 = A01 * B1
        U[i_base*m + (j_base+1)][col] += 1
        V[(j_base+1)*p + k_base][col] += 1
        W[k_base*n + i_base][col] += 1
        col += 1
        
        # P3 = A10 * B0
        U[(i_base+1)*m + j_base][col] += 1
        V[j_base*p + k_base][col] += 1
        W[k_base*n + (i_base+1)][col] += 1
        col += 1
        
        # P4 = A11 * B1
        U[(i_base+1)*m + (j_base+1)][col] += 1
        V[(j_base+1)*p + k_base][col] += 1
        W[k_base*n + (i_base+1)][col] += 1
        col += 1

    # Block 1: A_L B_T1 (i=0,1; j=0,1; k=0,1)
    add_strassen_222(0, 0, 0, 0)
    
    # Block 2: A_L B_T2 (i=0,1; j=0,1; k=2,3)
    add_strassen_222(0, 0, 2, 0)
    
    # Block 3: A_L B_T3 (i=0,1; j=0,1; k=4)
    add_strassen_221(0, 0, 4, 0)
    
    # Block 4: A_R B_B1 (i=0,1; j=2,3; k=0,1)
    # Note: A_R uses j=2,3. B_B1 uses rows j=2,3 of B? 
    # Wait. B is 4x5. Rows 0,1 are B_T. Rows 2,3 are B_B.
    # So for B_B, the "row index" in the 2x5 block is 0,1 which maps to global 2,3.
    # So j_base for B is 2.
    # A_R uses columns 2,3 of A. So j_base for A is 2.
    # k_base for B_B1 is 0.
    add_strassen_222(0, 2, 0, 0) # i_base=0, j_base=2 (A cols), k_base=0 (B cols)
    # Wait, add_strassen_222(i_base, j_base_A, k_base_B)
    # My function signature was add_strassen_222(i_base, j_base, k_base, offset)
    # I used j_base for both A and B in the function logic?
    # Let's check the function.
    # U[i_base*m + j_base] -> A row i_base, col j_base.
    # V[j_base*p + k_base] -> B row j_base, col k_base.
    # This assumes A and B use the same j_base index for the "inner" dimension block.
    # For A_L B_T1: A cols 0,1. B rows 0,1. j_base=0. Correct.
    # For A_R B_B1: A cols 2,3. B rows 2,3. j_base=2. Correct.
    # But k_base is the column index in B.
    # For B_B1, cols are 0,1. So k_base=0.
    # So add_strassen_222(0, 2, 0, 0) is correct for A_R B_B1.
    
    # Block 5: A_R B_B2 (i=0,1; j=2,3; k=2,3)
    add_strassen_222(0, 2, 2, 0)
    
    # Block 6: A_R B_B3 (i=0,1; j=2,3; k=4)
    add_strassen_221(0, 2, 4, 0)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
