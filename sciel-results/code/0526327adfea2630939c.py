"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time

def _strassen_2x2x2():
    """Return U, V, W (each 4x7) for 2x2x2 Strassen multiplication.
    Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c].
    C[i,k] = sum_c W[k*2+i, c] * U[i*2+j, c] * V[j*2+k, c]
    Actually the tensor is T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
    where C[i,k] = sum_j A[i,j]*B[j,k] = sum_c W[k,i,c] * (sum_j U[i,j,c]*A[i,j]) * (sum_k' ... )
    
    Wait, let me be precise. The multiplication tensor T has T[i,j,k] = delta_{i,k} for the identity?
    No. For matrix multiplication C = A*B, C[i,k] = sum_j A[i,j] B[j,k].
    The tensor T_{i,j,k} = delta_{i,k}? No, that's not right either.
    
    The standard formulation: T[i,j,k] = 1 if i,j,k are the indices such that C[i,k] uses A[i,j]*B[j,k].
    Actually T is a 3-tensor where T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k].
    So T[i,j,k] = 1 always? No.
    
    Let me think again. C[i,k] = sum_j A[i,j] B[j,k].
    We want C[i,k] = sum_c W[k,i,c] * U[i,j,c] * V[j,k,c] ... no.
    
    The correct formulation: C[i,k] = sum_c (sum_j U[i,j,c] A[i,j]) * (sum_j V[j,k,c] B[j,k]) * W[k,i,c]
    
    Hmm, that's not quite standard. Let me use the standard tensor decomposition.
    
    The matrix multiplication tensor T has entries T[i,j,k] = 1 if and only if the product A[i,j]*B[j,k] contributes to C[i,k].
    So T[i,j,k] = 1 for all valid i,j,k? No, that can't be right because then the rank would be n*m*p.
    
    Actually, I think the convention is:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    We want to write this as:
    C[i,k] = sum_c W[k,i,c] * (sum_j U[i,j,c] * A[i,j]) * (sum_j V[j,k,c] * B[j,k])
    
    Wait, that's not dimensionally consistent in the standard way. Let me look at the seed program.
    
    In the seed: U[i*m+j][column] = 1, V[j*p+k][column] = 1, W[k*n+i][column] = 1
    So for each (i,j,k) triple, we have a rank-1 term.
    C[i,k] = sum_{j} A[i,j] B[j,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] A[i,j]) * (sum_j V[j*p+k, c] B[j,k])
    
    For the schoolbook, each c corresponds to a unique (i,j,k) triple, so:
    sum_j U[i*m+j, c] A[i,j] = A[i,j_c] (only one j is nonzero)
    sum_j V[j*p+k, c] B[j,k] = B[j_c, k] (same j_c)
    W[k*n+i, c] = 1
    
    So C[i,k] gets contribution A[i,j_c] * B[j_c, k] for each c where i_c=i, k_c=k.
    Summing over all c with i_c=i, k_c=k gives sum_j A[i,j] B[j,k]. Correct.
    
    So for Strassen 2x2x2:
    A is 2x2, B is 2x2, C is 2x2.
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    
    Strassen:
    M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    M2 = (A[1,0]+A[1,1])*B[0,0]
    M3 = A[0,0]*(B[0,1]-B[1,1])
    M4 = A[1,1]*(B[1,0]-B[0,0])
    M5 = (A[0,0]+A[0,1])*B[1,1]
    M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    
    C[0,0] = M1 - M4 + M5 - M7
    C[0,1] = M3 + M5
    C[1,0] = M2 + M4
    C[1,1] = M1 - M2 + M3 + M6
    
    Now, each M_c is of the form (sum_j U[i,j,c] A[i,j]) * (sum_j V[j,k,c] B[j,k])
    
    For M1: (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    U[0,0,1]=1, U[1,1,1]=1, others 0 for U.
    V[0,0,1]=1, V[1,1,1]=1, others 0 for V.
    
    W[k,i,c] for M1: C[0,0] gets +1, C[0,1] gets 0, C[1,0] gets 0, C[1,1] gets +1.
    W[0,0,1]=1 (k=0,i=0), W[1,1,1]=1 (k=1,i=1)
    
    For M2: (A[1,0]+A[1,1])*B[0,0]
    U[1,0,2]=1, U[1,1,2]=1
    V[0,0,2]=1
    C[1,0] gets +1, C[1,1] gets -1
    W[0,1,2]=1, W[1,1,2]=-1
    
    For M3: A[0,0]*(B[0,1]-B[1,1])
    U[0,0,3]=1
    V[0,1,3]=1, V[1,1,3]=-1
    C[0,1] gets +1, C[1,1] gets +1
    W[1,0,3]=1, W[1,1,3]=1
    
    For M4: A[1,1]*(B[1,0]-B[0,0])
    U[1,1,4]=1
    V[1,0,4]=1, V[0,0,4]=-1
    C[0,0] gets -1, C[1,0] gets +1
    W[0,0,4]=-1, W[0,1,4]=1
    
    For M5: (A[0,0]+A[0,1])*B[1,1]
    U[0,0,5]=1, U[0,1,5]=1
    V[1,1,5]=1
    C[0,0] gets +1, C[0,1] gets +1
    W[0,0,5]=1, W[1,0,5]=1
    
    For M6: (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    U[1,0,6]=1, U[0,0,6]=-1
    V[0,0,6]=1, V[0,1,6]=1
    C[1,1] gets +1
    W[1,1,6]=1
    
    For M7: (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    U[0,1,7]=1, U[1,1,7]=-1
    V[1,0,7]=1, V[1,1,7]=1
    C[0,0] gets -1
    W[0,0,7]=-1
    
    Let me verify C[0,0] = M1 - M4 + M5 - M7:
    W[0,0,1]=1, W[0,0,4]=-1, W[0,0,5]=1, W[0,0,7]=-1. Correct.
    C[0,1] = M3 + M5:
    W[1,0,3]=1, W[1,0,5]=1. Correct.
    C[1,0] = M2 + M4:
    W[0,1,2]=1, W[0,1,4]=1. Correct.
    C[1,1] = M1 - M2 + M3 + M6:
    W[1,1,1]=1, W[1,1,2]=-1, W[1,1,3]=1, W[1,1,6]=1. Correct.
    """
    U = [[0]*7 for _ in range(4)]  # i*2+j
    V = [[0]*7 for _ in range(4)]  # j*2+k
    W = [[0]*7 for _ in range(4)]  # k*2+i
    
    # M1: (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    U[0*2+0][0] = 1; U[1*2+1][0] = 1
    V[0*2+0][0] = 1; V[1*2+1][0] = 1
    W[0*2+0][0] = 1; W[1*2+1][0] = 1
    
    # M2: (A[1,0]+A[1,1])*B[0,0]
    U[1*2+0][1] = 1; U[1*2+1][1] = 1
    V[0*2+0][1] = 1
    W[0*2+1][1] = 1; W[1*2+1][1] = -1
    
    # M3: A[0,0]*(B[0,1]-B[1,1])
    U[0*2+0][2] = 1
    V[0*2+1][2] = 1; V[1*2+1][2] = -1
    W[1*2+0][2] = 1; W[1*2+1][2] = 1
    
    # M4: A[1,1]*(B[1,0]-B[0,0])
    U[1*2+1][3] = 1
    V[1*2+0][3] = 1; V[0*2+0][3] = -1
    W[0*2+0][3] = -1; W[0*2+1][3] = 1
    
    # M5: (A[0,0]+A[0,1])*B[1,1]
    U[0*2+0][4] = 1; U[0*2+1][4] = 1
    V[1*2+1][4] = 1
    W[0*2+0][4] = 1; W[1*2+0][4] = 1
    
    # M6: (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    U[1*2+0][5] = 1; U[0*2+0][5] = -1
    V[0*2+0][5] = 1; V[0*2+1][5] = 1
    W[1*2+1][5] = 1
    
    # M7: (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    U[0*2+1][6] = 1; U[1*2+1][6] = -1
    V[1*2+0][6] = 1; V[1*2+1][6] = 1
    W[0*2+0][6] = -1
    
    return U, V, W

def _schoolbook_2x2x1():
    """Return U, V, W (each 4x4) for 2x2x1 schoolbook.
    C[i,0] = sum_j A[i,j] B[j,0]
    """
    U = [[0]*4 for _ in range(4)]
    V = [[0]*4 for _ in range(4)]
    W = [[0]*4 for _ in range(4)]
    col = 0
    for i in range(2):
        for j in range(2):
            U[i*2+j][col] = 1
            V[j*2+0][col] = 1
            W[0*2+i][col] = 1
            col += 1
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # Partition m=4 into 2+2, p=5 into 2+3
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] (2x2x2)
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2] (2x2x2)
    # Block 3: A[0:2, 0:2] * B[0:2, 2:5] -> C[0:2, 2:5] (2x2x3)
    # Block 4: A[0:2, 2:4] * B[2:4, 2:5] -> C[0:2, 2:5] (2x2x3)
    
    # For 2x2x3, we can use Strassen(2x2x2) + schoolbook(2x2x1) = 7+4 = 11 rank
    # But wait, the known optimal for 2x2x3 is actually 11? Let me think...
    # Actually, 2x2x3 multiplication: C[i,k] = sum_j A[i,j] B[j,k] for k=0,1,2
    # This is 3 separate 2x2x1 multiplications sharing the same A.
    # We can do: for each k, C[:,k] = A * B[:,k]
    # Using Strassen-like: we can combine the B columns.
    # Actually, rank of 2x2x3 is 11 (I believe).
    # Let's just use 2 Strassen blocks won't work since we need 3 columns.
    # Use Strassen for 2 columns (rank 7) + schoolbook for 1 column (rank 4) = 11.
    
    # Total rank: 7 + 7 + 11 + 11 = 36
    
    # But wait, the directive says partition p into 2+3 and use rank-10 for 2x2x3.
    # Is the rank of 2x2x3 actually 10? Let me check.
    # The multiplication tensor for 2x2x3 has rank 11 according to some sources, but maybe 10?
    # Actually, I recall that the rank of 2x2x3 is 11. Let me just use 11.
    # Hmm, but the directive mentions "known optimal rank-10 decomposition for the 2x2x3 tensor".
    # Let me try to construct a rank-10 decomposition if possible, otherwise use 11.
    
    # For now, let me implement with 2x2x2 (rank 7) and 2x2x1 (rank 4) blocks.
    # Block 1: 2x2x2 Strassen (rank 7)
    # Block 2: 2x2x2 Strassen (rank 7)
    # Block 3: 2x2x3 = 2x2x2 Strassen (rank 7) + 2x2x1 schoolbook (rank 4) = rank 11
    # Block 4: 2x2x3 = 2x2x2 Strassen (rank 7) + 2x2x1 schoolbook (rank 4) = rank 11
    # Total: 7+7+11+11 = 36
    
    # Let me build this.
    
    U_list = []
    V_list = []
    W_list = []
    
    # Helper to add a block
    def add_block(U_blk, V_blk, W_blk, row_offset_U, row_offset_V, row_offset_W):
        """Add a block's factors to the global lists."""
        r = len(U_blk[0]) if U_blk else 0
        for i in range(len(U_blk)):
            U_list.append([0]*r)
        for i in range(len(V_blk)):
            V_list.append([0]*r)
        for i in range(len(W_blk)):
            W_list.append([0]*r)
        for c in range(r):
            for i in range(len(U_blk)):
                U_list[row_offset_U + i][c] = U_blk[i][c]
            for i in range(len(V_blk)):
                V_list[row_offset_V + i][c] = V_blk[i][c]
            for i in range(len(W_blk)):
                W_list[row_offset_W + i][c] = W_blk[i][c]
    
    # This approach is getting complicated. Let me just build the full matrices directly.
    
    total_rank = 36
    U = [[0]*total_rank for _ in range(n*m)]  # 8 rows
    V = [[0]*total_rank for _ in range(m*p)]  # 20 rows
    W = [[0]*total_rank for _ in range(p*n)]  # 10 rows
    
    col = 0
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # U indices: i in 0..1, j in 0..1 -> row i*4+j (since m=4)
    # V indices: j in 0..1, k in 0..1 -> row j*5+k (since p=5)
    # W indices: k in 0..1, i in 0..1 -> row k*2+i (since n=2)
    U1, V1, W1 = _strassen_2x2x2()
    for c in range(7):
        for i in range(2):
            for j in range(2):
                U[i*4 + j][col + c] = U1[i*2 + j][c]
        for j in range(2):
            for k in range(2):
                V[j*5 + k][col + c] = V1[j*2 + k][c]
        for k in range(2):
            for i in range(2):
                W[k*2 + i][col + c] = W1[k*2 + i][c]
    col += 7
    
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # U indices: i in 0..1, j in 2..3 -> row i*4+j
    # V indices: j in 2..3, k in 0..1 -> row j*5+k
    # W indices: k in 0..1, i in 0..1 -> row k*2+i
    U2, V2, W2 = _strassen_2x2x2()
    for c in range(7):
        for i in range(2):
            for j in range(2):
                U[i*4 + (j+2)][col + c] = U2[i*2 + j][c]
        for j in range(2):
            for k in range(2):
                V[(j+2)*5 + k][col + c] = V2[j*2 + k][c]
        for k in range(2):
            for i in range(2):
                W[k*2 + i][col + c] = W2[k*2 + i][c]
    col += 7
    
    # Block 3: A[0:2, 0:2] * B[0:2, 2:5] -> C[0:2, 2:5]
    # This is 2x2x3: A is 2x2, B is 2x3, C is 2x3
    # Split into 2x2x2 (k=2,3) + 2x2x1 (k=4)
    
    # 2x2x2 part: B columns 2,3
    U3a, V3a, W3a = _strassen_2x2x2()
    for c in range(7):
        for i in range(2):
            for j in range(2):
                U[i*4 + j][col + c] = U3a[i*2 + j][c]
        for j in range(2):
            for k in range(2):
                V[j*5 + (k+2)][col + c] = V3a[j*2 + k][c]
        for k in range(2):
            for i in range(2):
                W[(k+2)*2 + i][col + c] = W3a[k*2 + i][c]
    col += 7
    
    # 2x2x1 part: B column 4
    U3b, V3b, W3b = _schoolbook_2x2x1()
    for c in range(4):
        for i in range(2):
            for j in range(2):
                U[i*4 + j][col + c] = U3b[i*2 + j][c]
        for j in range(2):
            V[j*5 + 4][col + c] = V3b[j*2 + 0][c]
        for i in range(2):
            W[4*2 + i][col + c] = W3b[0*2 + i][c]
    col += 4
    
    # Block 4: A[0:2, 2:4] * B[2:4, 2:5] -> C[0:2, 2:5]
    # 2x2x2 part: B columns 2,3, A columns 2,3
    U4a, V4a, W4a = _strassen_2x2x2()
    for c in range(7):
        for i in range(2):
            for j in range(2):
                U[i*4 + (j+2)][col + c] = U4a[i*2 + j][c]
        for j in range(2):
            for k in range(2):
                V[(j+2)*5 + (k+2)][col + c] = V4a[j*2 + k][c]
        for k in range(2):
            for i in range(2):
                W[(k+2)*2 + i][col + c] = W4a[k*2 + i][c]
    col += 7
    
    # 2x2x1 part: B column 4, A columns 2,3
    U4b, V4b, W4b = _schoolbook_2x2x1()
    for c in range(4):
        for i in range(2):
            for j in range(2):
                U[i*4 + (j+2)][col + c] = U4b[i*2 + j][c]
        for j in range(2):
            V[(j+2)*5 + 4][col + c] = V4b[j*2 + 0][c]
        for i in range(2):
            W[4*2 + i][col + c] = W4b[0*2 + i][c]
    col += 4
    
    assert col == total_rank, f"col={col}, total_rank={total_rank}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
