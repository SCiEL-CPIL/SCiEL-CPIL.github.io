"""Matrix multiplication 2,2,5 with rank-minimal exact decomposition."""

DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=5
    
    # We use a hybrid decomposition:
    # - Two 2-column blocks (columns 0,1 and 2,3) using Strassen's algorithm for 2x2x2 (rank 7 each)
    # - One 1-column block (column 4) using schoolbook (rank 4)
    # Total rank: 7 + 7 + 4 = 18
    
    total_rank = 18
    
    U = [[0] * total_rank for _ in range(n * m)]   # (4, 18)
    V = [[0] * total_rank for _ in range(m * p)]   # (10, 18)
    W = [[0] * total_rank for _ in range(p * n)]   # (10, 18)
    
    def add_strassen_block(c0, c1, start_rank):
        """Add Strassen decomposition for A(2x2) * B(2x2) -> C(2x2) where B columns are c0, c1."""
        # Strassen products for 0-indexed matrices:
        # P1 = (A00+A11)*(B00+B11)
        # P2 = (A10+A11)*B00
        # P3 = A00*(B01-B11)
        # P4 = A11*(B10-B00)
        # P5 = (A00+A01)*B11
        # P6 = (A10-A00)*(B00+B01)
        # P7 = (A01-A11)*(B10+B11)
        
        # P1
        U[0][start_rank] += 1
        U[3][start_rank] += 1
        V[0 * p + c0][start_rank] += 1
        V[1 * p + c1][start_rank] += 1
        
        # P2
        U[2][start_rank + 1] += 1
        U[3][start_rank + 1] += 1
        V[0 * p + c0][start_rank + 1] += 1
        
        # P3
        U[0][start_rank + 2] += 1
        V[0 * p + c1][start_rank + 2] += 1
        V[1 * p + c1][start_rank + 2] -= 1
        
        # P4
        U[3][start_rank + 3] += 1
        V[1 * p + c0][start_rank + 3] += 1
        V[0 * p + c0][start_rank + 3] -= 1
        
        # P5
        U[0][start_rank + 4] += 1
        U[1][start_rank + 4] += 1
        V[1 * p + c1][start_rank + 4] += 1
        
        # P6
        U[2][start_rank + 5] += 1
        U[0][start_rank + 5] -= 1
        V[0 * p + c0][start_rank + 5] += 1
        V[0 * p + c1][start_rank + 5] += 1
        
        # P7
        U[1][start_rank + 6] += 1
        U[3][start_rank + 6] -= 1
        V[1 * p + c0][start_rank + 6] += 1
        V[1 * p + c1][start_rank + 6] += 1
        
        # Output reconstruction:
        # C00 (row 0, col c0) = P1 - P4 + P7
        W[c0 * n + 0][start_rank] += 1
        W[c0 * n + 0][start_rank + 3] -= 1
        W[c0 * n + 0][start_rank + 6] += 1
        
        # C01 (row 0, col c1) = P3 + P1
        W[c1 * n + 0][start_rank + 2] += 1
        W[c1 * n + 0][start_rank] += 1
        
        # C10 (row 1, col c0) = P2 + P4
        W[c0 * n + 1][start_rank + 1] += 1
        W[c0 * n + 1][start_rank + 3] += 1
        
        # C11 (row 1, col c1) = P1 - P3 + P5
        W[c1 * n + 1][start_rank] += 1
        W[c1 * n + 1][start_rank + 2] -= 1
        W[c1 * n + 1][start_rank + 4] += 1
    
    # Block 1: columns 0, 1
    add_strassen_block(0, 1, 0)
    
    # Block 2: columns 2, 3
    add_strassen_block(2, 3, 7)
    
    # Block 3: column 4 (schoolbook)
    # C[4,0] = A00*B[0,4] + A01*B[1,4]
    # C[4,1] = A10*B[0,4] + A11*B[1,4]
    
    # Q1 = A00 * B[0,4]
    U[0][14] = 1
    V[0 * p + 4][14] = 1
    W[4 * n + 0][14] = 1
    
    # Q2 = A01 * B[1,4]
    U[1][15] = 1
    V[1 * p + 4][15] = 1
    W[4 * n + 0][15] = 1
    
    # Q3 = A10 * B[0,4]
    U[2][16] = 1
    V[0 * p + 4][16] = 1
    W[4 * n + 1][16] = 1
    
    # Q4 = A11 * B[1,4]
    U[3][17] = 1
    V[1 * p + 4][17] = 1
    W[4 * n + 1][17] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
