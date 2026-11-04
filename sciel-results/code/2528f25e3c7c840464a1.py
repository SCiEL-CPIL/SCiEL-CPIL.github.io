"""Matrix multiplication 4x5x6 rank minimization."""
import time
import sys

DIMENSIONS = [4, 5, 6]

def _strassen_2x2x2():
    """
    Return U, V, W for M_{2,2,2} using Strassen's algorithm (rank 7).
    
    Convention: U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    For n=m=p=2:
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
    
    Strassen's 7 products:
    M1 = (A00+A11)*(B00+B11)
    M2 = (A10+A11)*B00
    M3 = A00*(B01-B11)
    M4 = A11*(B10-B00)
    M5 = (A00+A01)*B11
    M6 = (A10-A00)*(B00+B01)
    M7 = (A01-A11)*(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # U indices: i*2+j for i,j in {0,1}
    # V indices: j*2+k for j,k in {0,1}  
    # W indices: k*2+i for k,i in {0,1}
    
    # M1 = (A00+A11)*(B00+B11)
    # U: A00=1, A11=1 -> U[0][0]=1, U[3][0]=1
    # V: B00=1, B11=1 -> V[0][0]=1, V[3][0]=1
    # W: C00 gets +1, C11 gets -1 -> W[0][0]=1, W[3][0]=-1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = -1
    
    # M2 = (A10+A11)*B00
    # U: A10=1, A11=1 -> U[2][1]=1, U[3][1]=1
    # V: B00=1 -> V[0][1]=1
    # W: C10 gets +1 -> W[2][1]=1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # M3 = A00*(B01-B11)
    # U: A00=1 -> U[0][2]=1
    # V: B01=1, B11=-1 -> V[1][2]=1, V[3][2]=-1
    # W: C01 gets +1 -> W[1][2]=1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # M4 = A11*(B10-B00)
    # U: A11=1 -> U[3][3]=1
    # V: B10=1, B00=-1 -> V[2][3]=1, V[0][3]=-1
    # W: C00 gets +1, C10 gets +1 -> W[0][3]=1, W[2][3]=1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A01)*B11
    # U: A00=1, A01=1 -> U[0][4]=1, U[1][4]=1
    # V: B11=1 -> V[3][4]=1
    # W: C00 gets -1, C01 gets +1 -> W[0][4]=-1, W[1][4]=1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A10-A00)*(B00+B01)
    # U: A10=1, A00=-1 -> U[2][5]=1, U[0][5]=-1
    # V: B00=1, B01=1 -> V[0][5]=1, V[1][5]=1
    # W: C11 gets +1 -> W[3][5]=1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)*(B10+B11)
    # U: A01=1, A11=-1 -> U[1][6]=1, U[3][6]=-1
    # V: B10=1, B11=1 -> V[2][6]=1, V[3][6]=1
    # W: C00 gets +1 -> W[0][6]=1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _schoolbook(n, m, p):
    """Return U, V, W for M_{n,m,p} using schoolbook (rank n*m*p)."""
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][c] = 1
                V[j*p + k][c] = 1
                W[k*n + i][c] = 1
                c += 1
    return U, V, W

def _combine_blocks(blocks):
    """
    Combine a list of (U, V, W, i_off, j_off, k_off, n_b, m_b, p_b) blocks
    into a single (U, V, W) for the full problem.
    
    Each block contributes to the full tensor with appropriate index offsets.
    """
    if not blocks:
        return [], [], []
    
    # Determine total rank
    total_rank = sum(len(b[0][0]) for b in blocks if b[0])
    
    # Get full dimensions from the first block's context
    # We need to know n, m, p for the full problem
    # This should be passed in or inferred
    
    return None  # This needs more context

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # Strategy: Use block decomposition with Strassen for 2x2x2 sub-blocks
    # Split: rows 4 -> 2+2, cols 6 -> 3+3, middle 5 -> 2+3
    # This gives 4 sub-problems:
    #   M_{2,2,3}, M_{2,3,3}, M_{2,2,3}, M_{2,3,3}
    # 
    # For M_{2,2,3}: split cols 3 -> 2+1
    #   M_{2,2,2} (Strassen, rank 7) + M_{2,2,1} (schoolbook, rank 4) = 11
    # For M_{2,3,3}: split rows 2 -> 1+1, cols 3 -> 2+1
    #   This gets complex. Let's use schoolbook for M_{2,3,3} = rank 18
    # 
    # Total: 2*(11) + 2*(18) = 22 + 36 = 58... wait, that's not right.
    # 
    # Actually, the blocks are independent in the sense that they contribute to
    # different parts of C. Let me reconsider.
    #
    # C[i,k] = sum_j A[i,j] B[j,k]
    # If we split i into i1+i2, j into j1+j2, k into k1+k2:
    # C[i1,k1] = sum_{j1} A[i1,j1]B[j1,k1] + sum_{j2} A[i1,j2]B[j2,k1]
    # C[i1,k2] = sum_{j1} A[i1,j1]B[j1,k2] + sum_{j2} A[i1,j2]B[j2,k2]
    # etc.
    #
    # So we have 4 independent sub-multiplications:
    # C[0:2, 0:3] involves A[0:2, 0:2] * B[0:2, 0:3] and A[0:2, 2:5] * B[2:5, 0:3]
    # This is not simply a block decomposition of independent sub-problems.
    #
    # The correct block decomposition is:
    # C = A * B where A is 4x5, B is 5x6
    # Split A into [A_top; A_bottom] (2x5 + 2x5)
    # Split B into [B_left B_right] (5x3 + 5x3)
    # C = [A_top * B_left, A_top * B_right; A_bottom * B_left, A_bottom * B_right]
    #
    # So we have 4 independent sub-multiplications:
    # 1. A_top (2x5) * B_left (5x3) -> C[0:2, 0:3]
    # 2. A_top (2x5) * B_right (5x3) -> C[0:2, 3:6]
    # 3. A_bottom (2x5) * B_left (5x3) -> C[2:4, 0:3]
    # 4. A_bottom (2x5) * B_right (5x3) -> C[2:4, 3:6]
    #
    # Each is M_{2,5,3}. If we can do M_{2,5,3} with rank < 30, we save.
    #
    # For M_{2,5,3}: split middle 5 -> 2+3
    # = M_{2,2,3} + M_{2,3,3}
    # M_{2,2,3}: split cols 3 -> 2+1, use Strassen for M_{2,2,2} part
    #   M_{2,2,2} (rank 7) + M_{2,2,1} (rank 4) = 11
    # M_{2,3,3}: schoolbook = 18
    # Total for M_{2,5,3}: 11 + 18 = 29
    #
    # Total for M_{4,5,6}: 4 * 29 = 116
    #
    # That's an improvement from 120 to 116!
    
    # Let's implement this carefully.
    
    def get_M223():
        """M_{2,2,3} using Strassen for M_{2,2,2} + schoolbook for M_{2,2,1}"""
        # Split cols: 3 -> 2+1
        # Part 1: M_{2,2,2} (Strassen, rank 7)
        U1, V1, W1 = _strassen_2x2x2()
        # Part 2: M_{2,2,1} (schoolbook, rank 4)
        U2, V2, W2 = _schoolbook(2, 2, 1)
        
        # Combine: total rank = 7 + 4 = 11
        # U: shape (4, 11)
        # V: shape (6, 11)  [since m*p = 2*3 = 6]
        # W: shape (6, 11)  [since p*n = 3*2 = 6]
        
        rank_total = 7 + 4
        U = [[0]*rank_total for _ in range(4)]
        V = [[0]*rank_total for _ in range(6)]
        W = [[0]*rank_total for _ in range(6)]
        
        # Fill from U1, V1, W1 (for cols 0,1)
        for i in range(4):
            for c in range(7):
                U[i][c] = U1[i][c]
        for j in range(2):
            for k in range(2):
                for c in range(7):
                    V[j*3 + k][c] = V1[j*2 + k][c]
        for k in range(2):
            for i in range(2):
                for c in range(7):
                    W[k*2 + i][c] = W1[k*2 + i][c]
        
        # Fill from U2, V2, W2 (for col 2)
        for i in range(4):
            for c in range(4):
                U[i][7 + c] = U2[i][c]
        for j in range(2):
            for k in range(1):
                for c in range(4):
                    V[j*3 + 2 + k][7 + c] = V2[j*1 + k][c]
        for k in range(1):
            for i in range(2):
                for c in range(4):
                    W[2*2 + i][7 + c] = W2[k*2 + i][c]
        
        return U, V, W
    
    def get_M233():
        """M_{2,3,3} using schoolbook (rank 18)"""
        return _schoolbook(2, 3, 3)
    
    def get_M253():
        """M_{2,5,3} = M_{2,2,3} + M_{2,3,3}"""
        U1, V1, W1 = get_M223()
        U2, V2, W2 = get_M233()
        
        rank_total = 11 + 18
        U = [[0]*rank_total for _ in range(10)]  # 2*5 = 10
        V = [[0]*rank_total for _ in range(15)]  # 5*3 = 15
        W = [[0]*rank_total for _ in range(6)]   # 3*2 = 6
        
        # Fill from U1, V1, W1 (for middle indices 0,1)
        for i in range(2):
            for j in range(2):
                for c in range(11):
                    U[i*5 + j][c] = U1[i*2 + j][c]
        for j in range(2):
            for k in range(3):
                for c in range(11):
                    V[j*3 + k][c] = V1[j*3 + k][c]
        for k in range(3):
            for i in range(2):
                for c in range(11):
                    W[k*2 + i][c] = W1[k*2 + i][c]
        
        # Fill from U2, V2, W2 (for middle indices 2,3,4)
        for i in range(2):
            for j in range(3):
                for c in range(18):
                    U[i*5 + 2 + j][11 + c] = U2[i*3 + j][c]
        for j in range(3):
            for k in range(3):
                for c in range(18):
                    V[(2 + j)*3 + k][11 + c] = V2[j*3 + k][c]
        for k in range(3):
            for i in range(2):
                for c in range(18):
                    W[k*2 + i][11 + c] = W2[k*2 + i][c]
        
        return U, V, W
    
    # Now build M_{4,5,6} from 4 copies of M_{2,5,3}
    # C[0:2, 0:3] = A[0:2, :] * B[:, 0:3]
    # C[0:2, 3:6] = A[0:2, :] * B[:, 3:6]
    # C[2:4, 0:3] = A[2:4, :] * B[:, 0:3]
    # C[2:4, 3:6] = A[2:4, :] * B[:, 3:6]
    
    U1, V1, W1 = get_M253()  # For C[0:2, 0:3]
    U2, V2, W2 = get_M253()  # For C[0:2, 3:6]
    U3, V3, W3 = get_M253()  # For C[2:4, 0:3]
    U4, V4, W4 = get_M253()  # For C[2:4, 3:6]
    
    rank_total = 4 * 29
    U = [[0]*rank_total for _ in range(20)]  # 4*5 = 20
    V = [[0]*rank_total for _ in range(30)]  # 5*6 = 30
    W = [[0]*rank_total for _ in range(24)]  # 6*4 = 24
    
    # Block 1: C[0:2, 0:3]
    for i in range(2):
        for j in range(5):
            for c in range(29):
                U[i*5 + j][c] = U1[i*5 + j][c]
    for j in range(5):
        for k in range(3):
            for c in range(29):
                V[j*6 + k][c] = V1[j*3 + k][c]
    for k in range(3):
        for i in range(2):
            for c in range(29):
                W[k*4 + i][c] = W1[k*2 + i][c]
    
    # Block 2: C[0:2, 3:6]
    for i in range(2):
        for j in range(5):
            for c in range(29):
                U[i*5 + j][29 + c] = U2[i*5 + j][c]
    for j in range(5):
        for k in range(3):
            for c in range(29):
                V[j*6 + 3 + k][29 + c] = V2[j*3 + k][c]
    for k in range(3):
        for i in range(2):
            for c in range(29):
                W[(3 + k)*4 + i][29 + c] = W2[k*2 + i][c]
    
    # Block 3: C[2:4, 0:3]
    for i in range(2):
        for j in range(5):
            for c in range(29):
                U[(2 + i)*5 + j][58 + c] = U3[i*5 + j][c]
    for j in range(5):
        for k in range(3):
            for c in range(29):
                V[j*6 + k][58 + c] = V3[j*3 + k][c]
    for k in range(3):
        for i in range(2):
            for c in range(29):
                W[k*4 + 2 + i][58 + c] = W3[k*2 + i][c]
    
    # Block 4: C[2:4, 3:6]
    for i in range(2):
        for j in range(5):
            for c in range(29):
                U[(2 + i)*5 + j][87 + c] = U4[i*5 + j][c]
    for j in range(5):
        for k in range(3):
            for c in range(29):
                V[j*6 + 3 + k][87 + c] = V4[j*3 + k][c]
    for k in range(3):
        for i in range(2):
            for c in range(29):
                W[(3 + k)*4 + 2 + i][87 + c] = W4[k*2 + i][c]
    
    return {"U": U, "V": V, "W": W}
