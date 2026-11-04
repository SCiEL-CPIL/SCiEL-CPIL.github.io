"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=6
    # We'll use Strassen on 2x2 blocks.
    # Split into two 2x2x6 multiplications.
    # Each 2x2x6 uses Strassen on 3 pairs of columns, giving 7*3=21 products.
    # Total rank = 42.
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    # For each of the two 2x2x6 blocks (A1, A2)
    for block in range(2):
        # A_block is rows 0,1 of A, columns 2*block, 2*block+1
        # B_block is rows 2*block, 2*block+1 of B, all columns
        # C_block contributes to all of C
        
        for pair in range(3):  # 3 pairs of columns: (0,1), (2,3), (4,5)
            k0 = 2 * pair
            k1 = 2 * pair + 1
            base = block * 21 + pair * 7
            
            # M1 = (A00+A11)(B00+B11)
            t = base + 0
            U[0 * m + 2*block][t] = 1      # A[0, 2*block]
            U[1 * m + 2*block+1][t] = 1    # A[1, 2*block+1]
            V[2*block * p + k0][t] = 1     # B[2*block, k0]
            V[(2*block+1) * p + k1][t] = 1 # B[2*block+1, k1]
            # C[0,k0] += M1, C[1,k1] += M1 (partially)
            W[k0 * n + 0][t] = 1
            W[k1 * n + 1][t] = 1
            
            # M2 = (A10+A11)B00
            t = base + 1
            U[1 * m + 2*block][t] = 1      # A[1, 2*block]
            U[1 * m + 2*block+1][t] = 1    # A[1, 2*block+1]
            V[2*block * p + k0][t] = 1     # B[2*block, k0]
            # C[0,k1] += M2, C[1,k0] += M2
            W[k1 * n + 0][t] = 1
            W[k0 * n + 1][t] = 1
            
            # M3 = A00(B10-B11)
            t = base + 2
            U[0 * m + 2*block][t] = 1      # A[0, 2*block]
            V[(2*block+1) * p + k0][t] = 1 # B[2*block+1, k0]
            V[(2*block+1) * p + k1][t] = -1 # B[2*block+1, k1]
            # C[0,k1] += M3, C[1,k1] += M3
            W[k1 * n + 0][t] = 1
            W[k1 * n + 1][t] = 1
            
            # M4 = A11(B01-B00)
            t = base + 3
            U[1 * m + 2*block+1][t] = 1    # A[1, 2*block+1]
            V[2*block * p + k1][t] = 1     # B[2*block, k1]
            V[2*block * p + k0][t] = -1    # B[2*block, k0]
            # C[0,k0] += M4, C[1,k0] += M4
            W[k0 * n + 0][t] = 1
            W[k0 * n + 1][t] = 1
            
            # M5 = (A00+A01)B11
            t = base + 4
            U[0 * m + 2*block][t] = 1      # A[0, 2*block]
            U[0 * m + 2*block+1][t] = 1    # A[0, 2*block+1]
            V[(2*block+1) * p + k1][t] = 1 # B[2*block+1, k1]
            # C[0,k0] -= M5, C[1,k1] += M5
            W[k0 * n + 0][t] = -1
            W[k1 * n + 1][t] = 1
            
            # M6 = (A10-A00)(B00+B10)
            t = base + 5
            U[1 * m + 2*block][t] = 1      # A[1, 2*block]
            U[0 * m + 2*block][t] = -1     # A[0, 2*block]
            V[2*block * p + k0][t] = 1     # B[2*block, k0]
            V[(2*block+1) * p + k0][t] = 1 # B[2*block+1, k0]
            # C[1,k0] += M6
            W[k0 * n + 1][t] = 1
            
            # M7 = (A01-A11)(B01+B11)
            t = base + 6
            U[0 * m + 2*block+1][t] = 1    # A[0, 2*block+1]
            U[1 * m + 2*block+1][t] = -1   # A[1, 2*block+1]
            V[2*block * p + k1][t] = 1     # B[2*block, k1]
            V[(2*block+1) * p + k1][t] = 1 # B[2*block+1, k1]
            # C[0,k0] += M7
            W[k0 * n + 0][t] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
