"""Matrix multiplication tensor decomposition for 3x4 x 4x8 with Strassen-based block decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def _strassen_222():
    """Return (U, V, W) for 2x2 * 2x2 using Strassen (rank 7)."""
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = A*B
    # Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 - M4 + M5 + M6
    # C12 = M3 + M4
    # C21 = M2 + M5
    # C22 = M1 - M3 + M2 + M7
    #
    # Indices: A[i,j] with i=0,1 j=0,1; B[j,k] with j=0,1 k=0,1; C[i,k] with i=0,1 k=0,1
    # U[i*m+j, r] where m=2
    # V[j*p+k, r] where p=2
    # W[k*n+i, r] where n=2 (note: W index is k*n+i per task convention)
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (a+d)(e+h)
    # a=A[0,0], d=A[1,1], e=B[0,0], h=B[1,1]
    # U: A[0,0]->idx 0, A[1,1]->idx 3
    # V: B[0,0]->idx 0, B[1,1]->idx 3
    # W: C[0,0]->idx k*n+i = 0*2+0=0, C[1,1]->idx 1*2+1=3
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1
    # C11 += M1, C22 += M1
    W[0,0] = 1  # C11
    W[3,0] = 1  # C22
    # But we need to handle multiple contributions to same C element
    # Let me use a different approach: store contributions explicitly
    
    # Let me redo this more carefully with a list of (U_idx, V_idx, W_idx, coeff)
    # Actually, let me just directly construct the matrices
    
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # C00 = M1 - M4 + M5 + M6
    # C01 = M3 + M4
    # C10 = M2 + M5
    # C11 = M1 - M3 + M2 + M7
    
    # U indices: A00=0, A01=1, A10=2, A11=3
    # V indices: B00=0, B01=1, B10=2, B11=3
    # W indices: C00=0, C01=1, C10=2, C11=3  (k*n+i: C[i,k] -> k*2+i)
    # Wait: W[k*n+i] where C[i,k], so C[0,0]->W[0], C[0,1]->W[2], C[1,0]->W[1], C[1,1]->W[3]
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (A00+A11)(B00+B11)
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    # Contributes to C00 (+1), C11 (+1)
    W[0,0] = 1   # C00
    W[3,0] = 1   # C11
    
    # M2: (A10+A11)*B00
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    # Contributes to C10 (+1)
    W[1,1] = 1   # C10
    
    # M3: A00*(B01-B11)
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1
    # Contributes to C01 (+1)
    W[2,2] = 1   # C01
    
    # M4: A11*(B10-B00)
    U[3,3] = 1
    V[2,3] = 1; V[0,3] = -1
    # Contributes to C00 (-1), C01 (+1)
    W[0,3] = -1  # C00
    W[2,3] = 1   # C01
    
    # M5: (A00+A01)*B11
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    # Contributes to C00 (+1), C10 (+1)
    W[0,4] = 1   # C00
    W[1,4] = 1   # C10
    
    # M6: (A10-A00)*(B00+B01)
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    # Contributes to C00 (+1)
    W[0,5] = 1   # C00
    
    # M7: (A01-A11)*(B10+B11)
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    # Contributes to C11 (+1)
    W[3,6] = 1   # C11
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 120
    
    # We'll decompose the 3x4 * 4x8 multiplication into blocks.
    # Split the 3x4 matrix A into row blocks: [2x4] and [1x4]
    # Split the 4x8 matrix B into column blocks: [4x4] and [4x4]
    # Also split the 4 columns of A and 4 rows of B into [2] and [2]
    
    # Blocks:
    # C[0:2, 0:4] = A[0:2, 0:2]*B[0:2, 0:4] + A[0:2, 2:4]*B[2:4, 0:4]
    # C[0:2, 4:8] = A[0:2, 0:2]*B[0:2, 4:8] + A[0:2, 2:4]*B[2:4, 4:8]
    # C[2:3, 0:4] = A[2:3, 0:2]*B[0:2, 0:4] + A[2:3, 2:4]*B[2:4, 0:4]
    # C[2:3, 4:8] = A[2:3, 0:2]*B[0:2, 4:8] + A[2:3, 2:4]*B[2:4, 4:8]
    
    # Each sub-multiplication:
    # 2x2 * 2x4 -> can be done as two 2x2*2x2 using Strassen: rank 7+7=14
    # 1x2 * 2x4 -> can be done as two 1x2*2x2: rank 3+3=6 (or better)
    
    # Let's use:
    # For 2x2 * 2x4: split B into two 2x2 blocks, use Strassen for each: rank 14
    # For 1x2 * 2x4: split B into two 2x2 blocks, use rank 3 for each (1x2*2x2): rank 6
    
    # Actually, let me think about 1x2 * 2x2 more carefully.
    # A = [a, b], B = [[e, f], [g, h]]
    # C = [ae+bg, af+bh]
    # This is just 2 separate scalar multiplications of length-2 dot products.
    # We can do it with rank 4: one term per product.
    # Or we can try to do better.
    # Actually, 1x2 * 2x2 has tensor rank 4 (it's just two independent length-2 dot products).
    
    # Let me just use the naive approach for 1x2*2x2: rank 4
    # And Strassen for 2x2*2x2: rank 7
    
    # Total rank:
    # 4 blocks of 2x2*2x4 (each rank 14): 4*14 = 56
    # 4 blocks of 1x2*2x4 (each rank 6): 4*6 = 24
    # Total: 80
    
    # Wait, let me recount the blocks:
    # C[0:2, 0:4]: A[0:2,0:2]*B[0:2,0:4] (2x2*2x4, rank 14) + A[0:2,2:4]*B[2:4,0:4] (2x2*2x4, rank 14)
    # C[0:2, 4:8]: A[0:2,0:2]*B[0:2,4:8] (2x2*2x4, rank 14) + A[0:2,2:4]*B[2:4,4:8] (2x2*2x4, rank 14)
    # C[2:3, 0:4]: A[2:3,0:2]*B[0:2,0:4] (1x2*2x4, rank 6) + A[2:3,2:4]*B[2:4,0:4] (1x2*2x4, rank 6)
    # C[2:3, 4:8]: A[2:3,0:2]*B[0:2,4:8] (1x2*2x4, rank 6) + A[2:3,2:4]*B[2:4,4:8] (1x2*2x4, rank 6)
    
    # Total rank = 4*14 + 4*6 = 56 + 24 = 80
    
    # Let's build this.
    
    # Helper: create 2x2*2x4 decomposition using two Strassen 2x2*2x2
    def _strassen_22x24():
        """Return (U, V, W) for 2x2 * 2x4 using two Strassen blocks."""
        U1, V1, W1 = _strassen_222()  # for left 2x2 block
        U2, V2, W2 = _strassen_222()  # for right 2x2 block
        
        # U: shape (4, 14) - first 7 cols for left, next 7 for right
        U = np.zeros((4, 14))
        U[:, :7] = U1
        U[:, 7:] = U2
        
        # V: shape (8, 14) - rows 0-3 for left B block (rows 0-1 of full B), rows 4-7 for right
        V = np.zeros((8, 14))
        V[:4, :7] = V1
        V[4:, 7:] = V2
        
        # W: shape (8, 14) - C has 2 rows x 4 cols, W index = k*n+i = k*2+i
        # Left block: C[0:2, 0:2] -> W[0:4, :7]
        # Right block: C[0:2, 2:4] -> W[4:8, 7:]
        W = np.zeros((8, 14))
        W[:4, :7] = W1
        W[4:, 7:] = W2
        
        return U, V, W
    
    # Helper: create 1x2 * 2x4 decomposition
    def _naive_12x24():
        """Return (U, V, W) for 1x2 * 2x4 using naive rank 6."""
        # A = [a, b], B is 2x4
        # C = [a*B[0,0] + b*B[1,0], a*B[0,1] + b*B[1,1], a*B[0,2] + b*B[1,2], a*B[0,3] + b*B[1,3]]
        # Rank 6: 3 terms per output element (but actually 2 terms each, so rank 4 would suffice if we can't share)
        # Actually: C_k = a*B[0,k] + b*B[1,k]
        # This needs rank 4: (a,B[0,0]), (b,B[1,0]), (a,B[0,1]), (b,B[1,1])... no that's 4 for one output.
        # For 4 outputs: 8 terms? No.
        # C_0 = a*b00 + b*b10
        # C_1 = a*b01 + b*b11
        # C_2 = a*b02 + b*b12
        # C_3 = a*b03 + b*b13
        # Each C_k needs 2 terms, total 8 terms with no sharing. Rank 8.
        # But wait, can we do better? The tensor is a sum of 4 independent 1x1*1x2*1x2 tensors.
        # Each 1x1*1x2*1x2 has rank 2. Total rank 8.
        # Hmm, that's not great. Let me just use rank 8.
        
        # Actually, let me reconsider. 1x2 * 2x4:
        # U: shape (2, rank) - indices for a, b
        # V: shape (8, rank) - indices for B[0,0]..B[0,3], B[1,0]..B[1,3]
        # W: shape (4, rank) - indices for C[0]..C[3]
        
        # Naive: rank 8
        U = np.zeros((2, 8))
        V = np.zeros((8, 8))
        W = np.zeros((4, 8))
        
        for k in range(4):
            # C_k = a*B[0,k] + b*B[1,k]
            U[0, k] = 1    # a
            U[1, k+4] = 1  # b
            V[k, k] = 1        # B[0,k]
            V[k+4, k+4] = 1    # B[1,k]
            W[k, k] = 1
            W[k, k+4] = 1
        
        return U, V, W
    
    # Now build the full decomposition
    # Total rank = 4*14 + 4*8 = 56 + 32 = 88
    
    rank = 88
    U_full = np.zeros((12, rank))  # n*m = 3*4 = 12
    V_full = np.zeros((32, rank))  # m*p = 4*8 = 32
    W_full = np.zeros((24, rank))  # p*n = 8*3 = 24
    
    col = 0
    
    # Block 1: C[0:2, 0:4] += A[0:2, 0:2] * B[0:2, 0:4]
    # A[0:2, 0:2]: rows 0-1, cols 0-1 of A -> U indices 0-3
    # B[0:2, 0:4]: rows 0-1, cols 0-3 of B -> V indices 0-7
    # C[0:2, 0:4]: rows 0-1, cols 0-3 of C -> W indices k*3+i for i=0,1, k=0..3
    U1, V1, W1 = _strassen_22x24()
    U_full[0:4, col:col+14] = U1
    V_full[0:8, col:col+14] = V1
    # W mapping: W1 has indices for 2x4 C, where W1[k*2+i, r]
    # Full W: W_full[k*3+i, r] for i in 0,1 and k in 0..3
    for k in range(4):
        for i in range(2):
            W_full[k*3+i, col:col+14] = W1[k*2+i, :]
    col += 14
    
    # Block 2: C[0:2, 0:4] += A[0:2, 2:4] * B[2:4, 0:4]
    # A[0:2, 2:4]: rows 0-1, cols 2-3 of A -> U indices 2-5
    # B[2:4, 0:4]: rows 2-3, cols 0-3 of B -> V indices 8-15
    U1, V1, W1 = _strassen_22x24()
    U_full[2:6, col:col+14] = U1
    V_full[8:16, col:col+14] = V1
    for k in range(4):
        for i in range(2):
            W_full[k*3+i, col:col+14] = W1[k*2+i, :]
    col += 14
    
    # Block 3: C[0:2, 4:8] += A[0:2, 0:2] * B[0:2, 4:8]
    # A[0:2, 0:2]: U indices 0-3
    # B[0:2, 4:8]: rows 0-1, cols 4-7 of B -> V indices 4-11
    U1, V1, W1 = _strassen_22x24()
    U_full[0:4, col:col+14] = U1
    V_full[4:12, col:col+14] = V1
    # C[0:2, 4:8]: k in 4..7, i in 0,1
    for k in range(4):
        for i in range(2):
            W_full[(k+4)*3+i, col:col+14] = W1[k*2+i, :]
    col += 14
    
    # Block 4: C[0:2, 4:8] += A[0:2, 2:4] * B[2:4, 4:8]
    # A[0:2, 2:4]: U indices 2-5
    # B[2:4, 4:8]: rows 2-3, cols 4-7 of B -> V indices 12-19
    U1, V1, W1 = _strassen_22x24()
    U_full[2:6, col:col+14] = U1
    V_full[12:20, col:col+14] = V1
    for k in range(4):
        for i in range(2):
            W_full[(k+4)*3+i, col:col+14] = W1[k*2+i, :]
    col += 14
    
    # Block 5: C[2:3, 0:4] += A[2:3, 0:2] * B[0:2, 0:4]
    # A[2:3, 0:2]: row 2, cols 0-1 of A -> U indices 8-9
    # B[0:2, 0:4]: V indices 0-7
    U1, V1, W1 = _naive_12x24()
    U_full[8:10, col:col+8] = U1
    V_full[0:8, col:col+8] = V1
    # C[2:3, 0:4]: i=2, k=0..3
    for k in range(4):
        W_full[k*3+2, col:col+8] = W1[k, :]
    col += 8
    
    # Block 6: C[2:3, 0:4] += A[2:3, 2:4] * B[2:4, 0:4]
    # A[2:3, 2:4]: row 2, cols 2-3 of A -> U indices 10-11
    # B[2:4, 0:4]: V indices 8-15
    U1, V1, W1 = _naive_12x24()
    U_full[10:12, col:col+8] = U1
    V_full[8:16, col:col+8] = V1
    for k in range(4):
        W_full[k*3+2, col:col+8] = W1[k, :]
    col += 8
    
    # Block 7: C[2:3, 4:8] += A[2:3, 0:2] * B[0:2, 4:8]
    # A[2:3, 0:2]: U indices 8-9
    # B[0:2, 4:8]: V indices 4-11
    U1, V1, W1 = _naive_12x24()
    U_full[8:10, col:col+8] = U1
    V_full[4:12, col:col+8] = V1
    for k in range(4):
        W_full[(k+4)*3+2, col:col+8] = W1[k, :]
    col += 8
    
    # Block 8: C[2:3, 4:8] += A[2:3, 2:4] * B[2:4, 4:8]
    # A[2:3, 2:4]: U indices 10-11
    # B[2:4, 4:8]: V indices 12-19
    U1, V1, W1 = _naive_12x24()
    U_full[10:12, col:col+8] = U1
    V_full[12:20, col:col+8] = V1
    for k in range(4):
        W_full[(k+4)*3+2, col:col+8] = W1[k, :]
    col += 8
    
    # Verify col == rank
    if col != rank:
        raise ValueError(f"Column mismatch: {col} != {rank}")
    
    # Convert to lists
    U_list = U_full.tolist()
    V_list = V_full.tolist()
    W_list = W_full.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
