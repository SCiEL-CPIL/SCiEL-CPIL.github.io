"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Decomposition:
    # A is 2x3. Split into A1 (2x2) and A2 (2x1).
    # B is 3x8. Split into B1 (2x8) and B2 (1x8).
    # C = A1 * B1 + A2 * B2.
    # A1*B1 is 2x2x8. Split into four 2x2x2 blocks. Rank 4*7=28.
    # A2*B2 is 2x1x8. Split into four 2x1x2 blocks. Rank 4*4=16.
    # Total rank 44.
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add Strassen 2x2x2 block
    # A indices: i in 0..1, j in 0..1 (local)
    # B indices: j in 0..1 (local), k in 0..1 (local)
    # C indices: i in 0..1, k in 0..1 (local)
    # Global indices:
    # A: i*m + j
    # B: j*p + k
    # W: k*n + i
    
    # Strassen 2x2x2 terms
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B11)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    def add_strassen_block(offset_A_i, offset_A_j, offset_B_j, offset_B_k, col_start):
        # Local indices 0,1
        # A rows: offset_A_i, offset_A_i+1
        # A cols: offset_A_j, offset_A_j+1
        # B rows: offset_B_j, offset_B_j+1
        # B cols: offset_B_k, offset_B_k+1
        
        # M1
        U[offset_A_i * m + offset_A_j][col_start] = 1
        U[(offset_A_i + 1) * m + (offset_A_j + 1)][col_start] = 1
        V[offset_B_j * p + offset_B_k][col_start] = 1
        V[(offset_B_j + 1) * p + (offset_B_k + 1)][col_start] = 1
        W[offset_B_k * n + offset_A_i][col_start] = 1
        W[(offset_B_k + 1) * n + (offset_A_i + 1)][col_start] = 1
        # Note: W indices for M1 contribute to C00 and C11?
        # C00 = M1... C11 = M1...
        # W[k*n+i, r].
        # For M1, we add to C00 and C11? No, M1 is a scalar product.
        # The term is U * V * W.
        # U has 1s at (0,0) and (1,1).
        # V has 1s at (0,0) and (1,1).
        # Product is A00*B00 + A11*B11.
        # We need W to place this in C00 and C11?
        # No, W is a matrix.
        # W[0, r] contributes to C00?
        # C[i,k] = sum U[i,j,r] V[j,k,r] W[k,i,r]?
        # Convention: W[k*n+i, r].
        # So W[k*n+i, r] is the coefficient for C[i,k].
        # For M1 = (A00+A11)(B00+B11).
        # This product appears in C00 and C11.
        # So W[0*n+0, r] = 1 and W[1*n+1, r] = 1.
        # Wait, my previous assignment was:
        # W[offset_B_k * n + offset_A_i][col_start] = 1
        # W[(offset_B_k + 1) * n + (offset_A_i + 1)][col_start] = 1
        # This sets W[0, r]=1 and W[1, r]=1 (if n=2).
        # W[0] is C[0,0]. W[1] is C[1,0]? No.
        # W[k*n+i].
        # k=0, i=0 -> 0. C[0,0].
        # k=1, i=1 -> 1*2+1=3. C[1,1].
        # So W[0]=1, W[3]=1.
        # My code: W[0][col]=1, W[3][col]=1. Correct.
        
        # M2 = (A10+A11)B00
        col_start += 1
        U[(offset_A_i + 1) * m + offset_A_j][col_start] = 1
        U[(offset_A_i + 1) * m + (offset_A_j + 1)][col_start] = 1
        V[offset_B_j * p + offset_B_k][col_start] = 1
        W[(offset_B_k + 1) * n + (offset_A_i + 1)][col_start] = 1 # C11? No, M2 is in C10 and C11?
        # C10 = M2 + M4. C11 = M1 - M2...
        # So M2 contributes to C10 and C11?
        # C10 index: k=0, i=1 -> 0*2+1=1.
        # C11 index: k=1, i=1 -> 1*2+1=3.
        # So W[1]=1, W[3]=-1?
        # We can have negative coefficients.
        # Let's just add terms with +1 and handle signs in W.
        # Actually, it's easier to define the 7 terms and their W coefficients directly.
        
        # This manual construction is error prone.
        # I will use the known rank 44 structure but implement it carefully.
        
        pass

    # Given the complexity, I will stick to the schoolbook rank 48 if I can't be sure of 44.
    # But the prompt says 44 is measured.
    # I will try to implement the 44 rank decomposition properly.
    
    # Let's restart the construction logic.
    # A1 (2x2) * B1 (2x8).
    # B1 is split into 4 blocks of 2x2.
    # For each block, use Strassen (7 terms).
    # A2 (2x1) * B2 (1x8).
    # B2 is split into 4 blocks of 1x2.
    # For each block, use schoolbook (4 terms).
    
    col = 0
    # Part 1: A1 * B1
    # A1 rows 0,1. Cols 0,1.
    # B1 rows 0,1. Cols 0..7.
    for b in range(4):
        # B block cols 2b, 2b+1
        # Strassen 2x2x2
        # M1
        U[0][col] = 1; U[3][col] = 1 # A00, A11 (indices 0, 3 in 2x3 matrix? No. A is 2x3. A[0,0]=0, A[1,1]=3? No. A[i*m+j]. A[0,0]=0. A[1,1]=1*3+1=4? No. m=3. A[0,0]=0. A[1,1]=4. A[0,1]=1. A[1,0]=3.
        # A1 is A[0:2, 0:2].
        # A[0,0] -> U[0]. A[0,1] -> U[1].
        # A[1,0] -> U[3]. A[1,1] -> U[4].
        
        # Let's use local indices.
        # A1: i in 0..1, j in 0..1. Global U[i*m+j].
        # B1: j in 0..1, k in 0..1. Global V[j*p+k].
        # C: i in 0..1, k in 0..1. Global W[k*n+i].
        
        # M1 = (A00+A11)(B00+B11)
        U[0][col] = 1
        U[4][col] = 1
        V[0][col] = 1
        V[1][col] = 1
        W[0][col] = 1
        W[3][col] = 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[3][col] = 1
        U[4][col] = 1
        V[0][col] = 1
        W[1][col] = 1 # C10
        W[3][col] = -1 # C11
        col += 1
        
        # M3 = A00(B10-B00)
        U[0][col] = 1
        V[2][col] = 1 # B10 (j=1, k=0)
        V[0][col] = -1 # B00
        W[2][col] = 1 # C01
        W[3][col] = 1 # C11
        col += 1
        
        # M4 = A11(B01-B11)
        U[4][col] = 1
        V[1][col] = 1 # B01 (j=0, k=1)
        V[3][col] = -1 # B11 (j=1, k=1)
        W[0][col] = 1 # C00
        W[1][col] = 1 # C10
        col += 1
        
        # M5 = (A00+A01)B11
        U[0][col] = 1
        U[1][col] = 1
        V[3][col] = 1
        W[0][col] = -1 # C00
        W[2][col] = 1 # C01
        col += 1
        
        # M6 = (A10-A00)(B00+B10)
        U[3][col] = 1
        U[0][col] = -1
        V[0][col] = 1
        V[2][col] = 1
        W[3][col] = 1 # C11
        col += 1
        
        # M7 = (A01-A11)(B01+B11)
        U[1][col] = 1
        U[4][col] = -1
        V[1][col] = 1
        V[3][col] = 1
        W[0][col] = 1 # C00
        W[2][col] = -1 # C01
        col += 1

    # Part 2: A2 * B2
    # A2 is A[0:2, 2:3]. U[2], U[5].
    # B2 is B[2:3, 0:8]. V[2*p+k].
    # C is same.
    for b in range(4):
        # B block cols 2b, 2b+1.
        # Schoolbook 2x1x2.
        # C00 = A0 B0
        U[2][col] = 1
        V[2*p + 2*b][col] = 1
        W[0][col] = 1
        col += 1
        # C01 = A0 B1
        U[2][col] = 1
        V[2*p + 2*b + 1][col] = 1
        W[2][col] = 1
        col += 1
        # C10 = A1 B0
        U[5][col] = 1
        V[2*p + 2*b][col] = 1
        W[1][col] = 1
        col += 1
        # C11 = A1 B1
        U[5][col] = 1
        V[2*p + 2*b + 1][col] = 1
        W[3][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
