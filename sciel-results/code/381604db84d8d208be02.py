import time
import numpy as np

def _verify(U, V, W, n, m, p, num_trials=20):
    """Verify tensor decomposition: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1"""
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build the full tensor T from U,V,W
    T = np.zeros((n, m, p))
    for c in range(rank):
        for i in range(n):
            for j in range(m):
                u_val = U_np[i*m+j, c]
                if u_val == 0:
                    continue
                for k in range(p):
                    v_val = V_np[j*p+k, c]
                    if v_val == 0:
                        continue
                    w_val = W_np[k*n+i, c]
                    if w_val == 0:
                        continue
                    T[i, j, k] += u_val * v_val * w_val
    
    # For matrix multiplication, T[i,j,k] should be 1 for all valid i,j,k
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    
    # Additional verification with random matrices
    for trial in range(num_trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_decomp = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    u_val = 0.0
                    v_val = 0.0
                    for j in range(m):
                        u_val += U_np[i*m+j, c] * A[i, j]
                        v_val += V_np[j*p+k, c] * B[j, k]
                    total += u_val * v_val * W_np[k*n+i, c]
                C_decomp[i, k] = total
        
        if not np.allclose(C_decomp, C_expected, atol=1e-8):
            return False
    
    return True

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _block_decomposition(n, m, p):
    """
    Block decomposition of n×m×p matrix multiplication.
    
    Strategy: Partition indices into blocks and use known low-rank decompositions
    for each block, combining them with appropriate index mappings.
    
    For 2×3×3:
    - Split j into {0,1} and {2}, k into {0,1} and {2}
    - Block (j∈{0,1}, k∈{0,1}): 2×2×2, rank 7 (Strassen)
    - Block (j∈{0,1}, k=2):      2×2×1, rank 3
    - Block (j=2, k∈{0,1}):      2×1×2, rank 3  
    - Block (j=2, k=2):          2×1×1, rank 2
    Total: 7 + 3 + 3 + 2 = 15
    """
    if n != 2 or m != 3 or p != 3:
        return None
    
    # Strassen for 2×2×2
    def strassen_2x2x2():
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
        # M1 = (A00+A11)(B00+B11)
        U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
        # M2 = (A10+A11)B00
        U[2][1]=1; U[3][1]=1; V[0][1]=1; W[2][1]=1; W[3][1]=-1
        # M3 = A00(B10-B11)
        U[0][2]=1; V[2][2]=1; V[3][2]=-1; W[1][2]=1; W[3][2]=1
        # M4 = A11(B01-B00)
        U[3][3]=1; V[0][3]=-1; V[1][3]=1; W[0][3]=-1; W[2][3]=1
        # M5 = (A00+A01)B11
        U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=1; W[1][4]=1
        # M6 = (A10-A00)(B00+B01)
        U[0][5]=-1; U[2][5]=1; V[0][5]=1; V[1][5]=1; W[3][5]=1
        # M7 = (A01-A11)(B10+B11)
        U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1
        return U, V, W
    
    # 2×2×1 decomposition (rank 3)
    # A is 2×2, B is 2×1, C is 2×1
    # c0 = a00*b0 + a01*b1
    # c1 = a10*b0 + a11*b1
    # p0 = a00*b0
    # p1 = a11*b1
    # p2 = (a01+a10)*(b0+b1)
    # c0 = p0 + a01*b1 = p0 + (p2 - a10*b0 - a11*b1 - a01*b0) ... 
    # Actually: c0 = a00*b0 + a01*b1
    # p2 = a01*b0 + a01*b1 + a10*b0 + a10*b1
    # c0 = p0 + a01*b1
    # a01*b1 = p2 - a01*b0 - a10*b0 - a10*b1
    # This doesn't simplify cleanly. Let me use a different rank-3 formula.
    #
    # Known rank-3 for 2×2×1:
    # p0 = a00 * b0
    # p1 = a11 * b1
    # p2 = (a01 + a10) * (b0 + b1)
    # c0 = a00*b0 + a01*b1 = p0 + a01*b1
    # Note: a01*b1 = (a01+a10)*(b0+b1) - a01*b0 - a10*b0 - a10*b1 - a11*b1 + a11*b1
    #      = p2 - a01*b0 - a10*(b0+b1)
    # This still has extra terms. 
    #
    # Let me just verify if rank 3 is actually achievable or use rank 4.
    # For safety, I'll use rank 4 (schoolbook) for the 2×2×1 and 2×1×2 blocks.
    # Total: 7 + 4 + 4 + 2 = 17
    #
    # Actually, let me try a different approach. The rank of M_{2,2,1} is known to be 3.
    # Let me look up the correct formula.
    #
    # For multiplying a 2×2 matrix by a 2×1 vector:
    # c0 = a00*b0 + a01*b1
    # c1 = a10*b0 + a11*b1
    #
    # Rank-3 decomposition:
    # p0 = a00 * b0
    # p1 = a11 * b1  
    # p2 = (a01 + a10) * (b0 + b1)
    #
    # c0 = a00*b0 + a01*b1
    #    = p0 + a01*b1
    #    = p0 + a01*(b0+b1) - a01*b0
    #    = p0 + [a01*(b0+b1)] - a01*b0
    # But a01*b0 is not directly available. However:
    # a01*b0 = a01*b0
    # And p2 = a01*b0 + a01*b1 + a10*b0 + a10*b1
    # So a01*b1 = p2 - a01*b0 - a10*b0 - a10*b1
    # c0 = p0 + p2 - a01*b0 - a10*b0 - a10*b1
    #    = p0 + p2 - a01*b0 - a10*(b0+b1)
    # Still not clean.
    #
    # I think the issue is that I'm not getting the right combination. Let me just use
    # the direct approach with rank 4 for these smaller blocks to ensure correctness.
    # Total rank: 7 + 4 + 4 + 2 = 17
    #
    # Alternatively, I can try to find a better global decomposition.
    #
    # For now, let's use the safe block decomposition with rank 17.
    
    def schoolbook_2x2x1():
        U = [[0]*4 for _ in range(4)]
        V = [[0]*4 for _ in range(2)]
        W = [[0]*4 for _ in range(2)]
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(1):
                    U[i*2+j][col] = 1
                    V[j*1+k][col] = 1
                    W[k*2+i][col] = 1
                    col += 1
        return U, V, W
    
    def schoolbook_2x1x2():
        U = [[0]*4 for _ in range(2)]
        V = [[0]*4 for _ in range(2)]
        W = [[0]*4 for _ in range(4)]
        col = 0
        for i in range(2):
            for j in range(1):
                for k in range(2):
                    U[i*1+j][col] = 1
                    V[j*2+k][col] = 1
                    W[k*2+i][col] = 1
                    col += 1
        return U, V, W
    
    def schoolbook_2x1x1():
        U = [[0]*2 for _ in range(2)]
        V = [[0]*2 for _ in range(1)]
        W = [[0]*2 for _ in range(2)]
        col = 0
        for i in range(2):
            for j in range(1):
                for k in range(1):
                    U[i*1+j][col] = 1
                    V[j*1+k][col] = 1
                    W[k*2+i][col] = 1
                    col += 1
        return U, V, W
    
    U1, V1, W1 = strassen_2x2x2()
    U2, V2, W2 = schoolbook_2x2x1()
    U3, V3, W3 = schoolbook_2x1x2()
    U4, V4, W4 = schoolbook_2x1x1()
    
    total_rank = 7 + 4 + 4 + 2
    
    U_full = [[0]*total_rank for _ in range(6)]
    V_full = [[0]*total_rank for _ in range(9)]
    W_full = [[0]*total_rank for _ in range(6)]
    
    col_off = 0
    # Block 1: A[:,0:2] * B[0:2,0:2] -> C[:,0:2]
    for r in range(4):
        for c in range(7):
            if U1[r][c] != 0: U_full[r][col_off+c] = U1[r][c]
            if V1[r][c] != 0: V_full[r][col_off+c] = V1[r][c]
            if W1[r][c] != 0: W_full[r][col_off+c] = W1[r][c]
    col_off += 7
    
    # Block 2: A[:,0:2] * B[0:2,2] -> C[:,2]
    for r in range(4):
        for c in range(4):
            if U2[r][c] != 0: U_full[r][col_off+c] = U2[r][c]
    for r in range(2):
        for c in range(4):
            if V2[r][c] != 0: V_full[r*3+2][col_off+c] = V2[r][c]
    for r in range(2):
        for c in range(4):
            if W2[r][c] != 0: W_full[3+r][col_off+c] = W2[r][c]
    col_off += 4
    
    # Block 3: A[:,2] * B[2,0:2] -> C[:,0:2]
    for r in range(2):
        for c in range(4):
            if U3[r][c] != 0: U_full[2+r][col_off+c] = U3[r][c]
    for r in range(2):
        for c in range(4):
            if V3[r][c] != 0: V_full[6+r][col_off+c] = V3[r][c]
    for r in range(4):
        for c in range(4):
            if W3[r][c] != 0: W_full[r][col_off+c] = W3[r][c]
    col_off += 4
    
    # Block 4: A[:,2] * B[2,2] -> C[:,2]
    for r in range(2):
        for c in range(2):
            if U4[r][c] != 0: U_full[2+r][col_off+c] = U4[r][c]
            if V4[r][c] != 0: V_full[8][col_off+c] = V4[r][c]
            if W4[r][c] != 0: W_full[3+r][col_off+c] = W4[r][c]
    col_off += 2
    
    return U_full, V_full, W_full

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 2, 3, 3
    
    # Try block decomposition first
    block_result = _block_decomposition(n, m, p)
    if block_result is not None:
        U, V, W = block_result
        if _verify(U, V, W, n, m, p, num_trials=10):
            return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
