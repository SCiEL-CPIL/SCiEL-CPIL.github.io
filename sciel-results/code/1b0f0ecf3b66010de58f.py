"""Matrix multiplication 2,2,11 - Strassen block decomposition"""
DIMENSIONS = [2, 2, 11]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=11
    
    # We decompose B (2x11) into 5 blocks of size 2x2 and 1 block of size 2x1
    # For each 2x2 block: use Strassen (rank 7)
    # For the 2x1 block: use standard multiplication (rank 4)
    # Total rank: 5*7 + 4 = 39
    
    # Strassen's algorithm for 2x2 * 2x2:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We'll build U, V, W matrices
    # U: shape (n*m, rank) = (4, 39)
    # V: shape (m*p, rank) = (22, 39)
    # W: shape (p*n, rank) = (22, 39)
    
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper: add a Strassen block for B columns starting at offset
    def add_strassen_block(b_offset):
        nonlocal col
        # b_offset is 0-based column index in B where this block starts
        # B has rows j=0,1 and columns k=b_offset, b_offset+1
        
        # M1 = (A11 + A22) * (B11 + B22)
        # A11 = A[0,0] = U index 0
        # A22 = A[1,1] = U index 3
        # B11 = B[0, b_offset] = V index b_offset
        # B22 = B[1, b_offset+1] = V index 2 + b_offset + 1 = b_offset + 3
        
        U[0][col] = 1
        U[3][col] = 1
        V[b_offset][col] = 1
        V[b_offset + 3][col] = 1
        # Output: C11 += M1, C22 += M1
        # C11 is at W index: row k=0, col i=0 -> 0*2+0 = 0
        # C22 is at W index: row k=1, col i=1 -> 1*2+1 = 3
        W[0][col] = 1
        W[3][col] = 1
        col += 1
        
        # M2 = (A21 + A22) * B11
        # A21 = A[1,0] = U index 2
        # A22 = A[1,1] = U index 3
        # B11 = B[0, b_offset] = V index b_offset
        U[2][col] = 1
        U[3][col] = 1
        V[b_offset][col] = 1
        # Output: C21 += M2
        # C21 is at W index: row k=0, col i=1 -> 0*2+1 = 1
        W[1][col] = 1
        col += 1
        
        # M3 = A11 * (B12 - B22)
        # A11 = A[0,0] = U index 0
        # B12 = B[0, b_offset+1] = V index b_offset + 1
        # B22 = B[1, b_offset+1] = V index b_offset + 3
        U[0][col] = 1
        V[b_offset + 1][col] = 1
        V[b_offset + 3][col] = -1
        # Output: C12 += M3, C22 += M3
        # C12 is at W index: row k=1, col i=0 -> 1*2+0 = 2
        # C22 is at W index: row k=1, col i=1 -> 1*2+1 = 3
        W[2][col] = 1
        W[3][col] = 1
        col += 1
        
        # M4 = A22 * (B21 - B11)
        # A22 = A[1,1] = U index 3
        # B21 = B[1, b_offset] = V index b_offset + 2
        # B11 = B[0, b_offset] = V index b_offset
        U[3][col] = 1
        V[b_offset + 2][col] = 1
        V[b_offset][col] = -1
        # Output: C11 += M4, C21 += M4
        # C11 is at W index: 0
        # C21 is at W index: 1
        W[0][col] = 1
        W[1][col] = 1
        col += 1
        
        # M5 = (A11 + A12) * B22
        # A11 = A[0,0] = U index 0
        # A12 = A[0,1] = U index 1
        # B22 = B[1, b_offset+1] = V index b_offset + 3
        U[0][col] = 1
        U[1][col] = 1
        V[b_offset + 3][col] = 1
        # Output: C11 -= M5, C12 += M5
        W[0][col] = -1
        W[2][col] = 1
        col += 1
        
        # M6 = (A21 - A11) * (B11 + B12)
        # A21 = A[1,0] = U index 2
        # A11 = A[0,0] = U index 0
        # B11 = B[0, b_offset] = V index b_offset
        # B12 = B[0, b_offset+1] = V index b_offset + 1
        U[2][col] = 1
        U[0][col] = -1
        V[b_offset][col] = 1
        V[b_offset + 1][col] = 1
        # Output: C22 += M6
        W[3][col] = 1
        col += 1
        
        # M7 = (A12 - A22) * (B21 + B22)
        # A12 = A[0,1] = U index 1
        # A22 = A[1,1] = U index 3
        # B21 = B[1, b_offset] = V index b_offset + 2
        # B22 = B[1, b_offset+1] = V index b_offset + 3
        U[1][col] = 1
        U[3][col] = -1
        V[b_offset + 2][col] = 1
        V[b_offset + 3][col] = 1
        # Output: C11 += M7
        W[0][col] = 1
        col += 1
    
    # Add 5 Strassen blocks for B columns 0-9
    for block in range(5):
        add_strassen_block(block * 2)
    
    # Add standard multiplication for B column 10 (2x1 block)
    # C[i,k] = sum_j A[i,j] B[j,k]
    # For k=10: C[i,10] = A[i,0]*B[0,10] + A[i,1]*B[1,10]
    
    # Term 1: A[0,0] * B[0,10] -> C[0,10]
    U[0][col] = 1
    V[10][col] = 1
    W[10 * 2 + 0][col] = 1  # k=10, i=0 -> 20+0=20
    col += 1
    
    # Term 2: A[0,1] * B[1,10] -> C[0,10]
    U[1][col] = 1
    V[11][col] = 1
    W[20][col] = 1
    col += 1
    
    # Term 3: A[1,0] * B[0,10] -> C[1,10]
    U[2][col] = 1
    V[10][col] = 1
    W[10 * 2 + 1][col] = 1  # k=10, i=1 -> 20+1=21
    col += 1
    
    # Term 4: A[1,1] * B[1,10] -> C[1,10]
    U[3][col] = 1
    V[11][col] = 1
    W[21][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
