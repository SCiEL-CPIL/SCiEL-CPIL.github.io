"""Combined program: Strassen-based decomposition for 4x4 * 4x9 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START

def _build_strassen_2x2():
    """
    Build Strassen decomposition for 2x2 x 2x2 matrix multiplication.
    Returns U (4,7), V (4,7), W (4,7) where:
    U indexed by (i,k) [row of A, col of B], V by (k,j) [col of A, row of B], W by (j,i) [row of C, col of C]
    Wait, the convention is:
    U: (n*m, rank) -> U[i*m+j, r] where i is row of A, j is col of A (which is row of B)
    V: (m*p, rank) -> V[j*p+k, r] where j is col of A (row of B), k is col of B
    W: (p*n, rank) -> W[k*n+i, r] where k is col of B, i is row of A (row of C)
    
    For 2x2x2:
    n=2, m=2, p=2
    U index: i*2+j
    V index: j*2+k
    W index: k*2+i
    
    Strassen for 2x2:
    A = [[A11, A12], [A21, A22]]
    B = [[B11, B12], [B21, B22]]
    C = [[C11, C12], [C21, C22]]
    
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B12-B22)
    M4 = A22(B21-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    Let's map this to the tensor decomposition.
    The tensor T_{(i,j), (j,k), (k,i)} = 1.
    For a rank-1 term r:
    U_{i,j}^r V_{j,k}^r W_{k,i}^r
    
    This corresponds to the contribution to C_{i,k} from the term involving A_{i,j} and B_{j,k}.
    Actually, the decomposition is:
    sum_r U_{i,j}^r V_{j,k}^r W_{k,i}^r = 1 if the indices match the "diagonal" of the multiplication?
    No, the condition is that for all i,j,k, sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r} = 1.
    Wait, this is only true for the "consistent" indices.
    The tensor T is defined such that T_{u,v,w} = 1 if u=i*m+j, v=j*p+k, w=k*n+i for some i,j,k.
    Since the mapping (i,j,k) -> (u,v,w) is a bijection from {0..n-1}x{0..m-1}x{0..p-1} to the support of T,
    we just need to ensure that for each (i,j,k), the sum is 1, and for all other (u,v,w), the sum is 0.
    
    In the Strassen decomposition, we are constructing U, V, W such that this holds.
    
    Let's define the Strassen 2x2 U, V, W matrices.
    U is 4x7, V is 4x7, W is 4x7.
    
    M1:
    A part: A11, A22 -> U[0,0]=1, U[3,0]=1 (indices 0=0*2+0, 3=1*2+1)
    B part: B11, B22 -> V[0,0]=1, V[3,0]=1 (indices 0=0*2+0, 3=1*2+1)
    C part: C11, C22 -> W[0,0]=1, W[3,0]=1 (indices 0=0*2+0, 3=1*2+1)
    
    M2:
    A part: A21, A22 -> U[2,1]=1, U[3,1]=1 (indices 2=1*2+0, 3=1*2+1)
    B part: B11 -> V[0,1]=1
    C part: C21, C22 -> W[2,1]=1, W[3,1]=-1 (indices 2=1*2+0, 3=1*2+1)
    Wait, C21 is at (i=1, k=0) -> W index k*n+i = 0*2+1 = 2.
    C22 is at (i=1, k=1) -> W index k*n+i = 1*2+1 = 3.
    So W[2,1]=1, W[3,1]=-1.
    
    M3:
    A part: A11 -> U[0,2]=1
    B part: B12, B22 -> V[1,2]=1, V[3,2]=-1 (indices 1=0*2+1, 3=1*2+1)
    C part: C12, C22 -> W[1,2]=1, W[3,2]=1 (indices 1=0*2+1, 3=1*2+1)
    C12 is at (i=0, k=1) -> W index 1*2+0 = 2? No, W index is k*n+i.
    For C12: i=0, k=1 -> W index 1*2+0 = 2.
    For C22: i=1, k=1 -> W index 1*2+1 = 3.
    So W[2,2]=1, W[3,2]=1.
    Let's re-check W indices.
    W index = k*n + i.
    C11: i=0, k=0 -> 0
    C12: i=0, k=1 -> 1*2+0 = 2
    C21: i=1, k=0 -> 0*2+1 = 1
    C22: i=1, k=1 -> 1*2+1 = 3
    
    So for M3:
    C12 -> W[2,2]=1
    C22 -> W[3,2]=1
    
    M4:
    A part: A22 -> U[3,3]=1
    B part: B21, B11 -> V[2,3]=1, V[0,3]=-1 (indices 2=1*2+0, 0=0*2+0)
    C part: C11, C21 -> W[0,3]=1, W[1,3]=1
    C11 -> W[0,3]=1
    C21 -> W[1,3]=1
    
    M5:
    A part: A11, A12 -> U[0,4]=1, U[1,4]=1 (indices 0=0*2+0, 1=0*2+1)
    B part: B22 -> V[3,4]=1
    C part: C11, C12 -> W[0,4]=-1, W[2,4]=1
    C11 -> W[0,4]=-1
    C12 -> W[2,4]=1
    
    M6:
    A part: A21, A11 -> U[2,5]=1, U[0,5]=-1
    B part: B11, B12 -> V[0,5]=1, V[1,5]=1
    C part: C22 -> W[3,5]=1
    
    M7:
    A part: A12, A22 -> U[1,6]=1, U[3,6]=-1
    B part: B21, B22 -> V[2,6]=1, V[3,6]=1
    C part: C11 -> W[0,6]=1
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_4x4x4_strassen():
    """
    Build Strassen-based rank-49 decomposition for 4x4 x 4x4.
    Uses 2x2 blocks, each multiplied using the 2x2 Strassen (rank 7).
    Total rank = 7 * 7 = 49.
    """
    U2, V2, W2 = _build_strassen_2x2()
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # Strassen at 4x4 level (blocks are 2x2)
    # A blocks: (0,0), (0,1), (1,0), (1,1)
    # B blocks: (0,0), (0,1), (1,0), (1,1)
    # C blocks: (0,0), (0,1), (1,0), (1,1)
    
    big_mults = [
        {
            'A': [((0,0), 1), ((1,1), 1)],
            'B': [((0,0), 1), ((1,1), 1)],
            'C': [((0,0), 1), ((1,1), 1)]
        },
        {
            'A': [((1,0), 1), ((1,1), 1)],
            'B': [((0,0), 1)],
            'C': [((0,1), 1), ((1,1), -1)]
        },
        {
            'A': [((0,0), 1)],
            'B': [((0,1), 1), ((1,1), -1)],
            'C': [((1,0), 1), ((1,1), 1)]
        },
        {
            'A': [((1,1), 1)],
            'B': [((1,0), 1), ((0,0), -1)],
            'C': [((0,0), 1), ((0,1), 1)]
        },
        {
            'A': [((0,0), 1), ((0,1), 1)],
            'B': [((1,1), 1)],
            'C': [((0,0), -1), ((1,0), 1)]
        },
        {
            'A': [((1,0), 1), ((0,0), -1)],
            'B': [((0,0), 1), ((0,1), 1)],
            'C': [((1,1), 1)]
        },
        {
            'A': [((0,1), 1), ((1,1), -1)],
            'B': [((1,0), 1), ((1,1), 1)],
            'C': [((0,0), 1)]
        }
    ]
    
    for big_idx, bm in enumerate(big_mults):
        for r2 in range(7):
            col = big_idx * 7 + r2
            
            # Update U
            for (bi, bj), sign_a in bm['A']:
                for i2 in range(2):
                    for k2 in range(2):
                        u_val = U2[i2*2+k2, r2]
                        if u_val != 0:
                            I = 2*bi + i2
                            J = 2*bj + k2
                            U[I*4+J, col] += sign_a * u_val
            
            # Update V
            for (bk, bj), sign_b in bm['B']:
                for k2 in range(2):
                    for j2 in range(2):
                        v_val = V2[k2*2+j2, r2]
                        if v_val != 0:
                            J = 2*bk + k2
                            K = 2*bj + j2
                            V[J*4+K, col] += sign_b * v_val
            
            # Update W
            for (bi, bj), sign_c in bm['C']:
                for i2 in range(2):
                    for j2 in range(2):
                        w_val = W2[i2*2+j2, r2]
                        if w_val != 0:
                            I = 2*bi + i2
                            J = 2*bj + j2
                            W[I*4+J, col] += sign_c * w_val
                            
    return U, V, W

def _schoolbook_fallback(n=4, m=4, p=9):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}

def _validate_numpy(U, V, W, n, m, p):
    """Validate using NumPy for speed."""
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    rank = U_np.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_np[i*m+j, :] * V_np[j*p+k, :] * W_np[k*n+i, :])
                if abs(val - 1) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Build 4x4x4 Strassen decomposition
    U444, V444, W444 = _build_4x4x4_strassen()
    
    # Convert to lists for the final output format
    U444_list = U444.tolist()
    V444_list = V444.tolist()
    W444_list = W444.tolist()
    
    # Validate 4x4x4
    if not _validate_numpy(U444_list, V444_list, W444_list, 4, 4, 4):
        return _schoolbook_fallback()
    
    # Build 4x4x9 decomposition
    # We split the 9 columns of B into 4 + 4 + 1.
    # Block 1: K=0..3 (4x4x4 Strassen)
    # Block 2: K=4..7 (4x4x4 Strassen)
    # Block 3: K=8 (4x4x1 schoolbook, rank 16)
    
    total_rank = 49 + 49 + 16
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(36)]
    W = [[0]*total_rank for _ in range(36)]
    
    # Block 1: columns 0-3
    for I in range(4):
        for J in range(4):
            for c in range(49):
                U[I*4+J][c] = U444_list[I*4+J][c]
    for J in range(4):
        for K in range(4):
            for c in range(49):
                V[J*9+K][c] = V444_list[J*4+K][c]
    for K in range(4):
        for I in range(4):
            for c in range(49):
                W[K*4+I][c] = W444_list[K*4+I][c]
    
    # Block 2: columns 4-7
    offset2 = 49
    for I in range(4):
        for J in range(4):
            for c in range(49):
                U[I*4+J][offset2+c] = U444_list[I*4+J][c]
    for J in range(4):
        for K in range(4):
            for c in range(49):
                V[J*9+(K+4)][offset2+c] = V444_list[J*4+K][c]
    for K in range(4):
        for I in range(4):
            for c in range(49):
                W[(K+4)*4+I][offset2+c] = W444_list[K*4+I][c]
    
    # Block 3: column 8 (schoolbook, rank 16)
    offset3 = 98
    for idx in range(16):
        I = idx // 4
        J = idx % 4
        U[I*4+J][offset3+idx] = 1
        V[J*9+8][offset3+idx] = 1
        W[8*4+I][offset3+idx] = 1
    
    # Final validation
    if not _validate_numpy(U, V, W, 4, 4, 9):
        return _schoolbook_fallback()
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
