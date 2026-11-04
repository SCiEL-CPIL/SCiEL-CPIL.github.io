"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _validate(U, V, W, n, m, p):
    """Check that U * V * W reconstructs the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Build the tensor T[i,j,k] = 1 if and only if output (i,k) uses input (i,j) and (j,k)
    # The multiplication tensor M_{i,j,k} = delta_{i,i} * delta_{j,j} ... 
    # Actually: (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor is T[i,j,k] = 1 for all valid i,j,k (it's the identity in the middle)
    # Wait no. The tensor for matrix multiplication C = A @ B where A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has indices (i, j, k) and T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # But the factorization is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, let me re-read the contract.
    # U: shape (n*m, rank) -> U[i*m + j, r]
    # V: shape (m*p, rank) -> V[j*p + k, r]
    # W: shape (p*n, rank) -> W[k*n + i, r]
    # And the tensor is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all valid (i,j,k)
    # Because (A*B)[i,k] = sum_j A[i,j]*B[j,k], the "tensor" that we're decomposing is such that
    # for each (i,j,k), the contribution is A[i,j]*B[j,k]*C[i,k]... 
    # No. The matrix multiplication tensor T satisfies T[i,j,k] = 1 for all i,j,k.
    # Let me verify with the schoolbook: U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    # sum over col: for a fixed (i,j,k), only the column where (i,j,k) matches contributes, giving 1.
    # So yes, T[i,j,k] = 1 for all valid (i,j,k).
    
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Check shapes
    if U_arr.shape != (n*m, rank) or V_arr.shape != (m*p, rank) or W_arr.shape != (p*n, rank):
        return False
    
    # For each (i,j,k), check sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    # This is a 3D tensor contraction. We can do it efficiently.
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    
    # We need: for all i,j,k: sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r] == 1
    # Let's compute the full tensor and check
    # This could be expensive for large rank, but n=2,m=3,p=8 so max 48 elements
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U3[i,j,:], V3[j,k,:] * W3[k,i,:])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _make_trivial(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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

def _direct_sum(U1, V1, W1, U2, V2, W2):
    """Combine two decompositions via direct sum (concatenate columns)."""
    rank1 = len(U1[0])
    rank2 = len(U2[0])
    U = [row[:] + [0]*rank2 for row in U1] + [[0]*rank1 + row[:] for row in U2]
    V = [row[:] + [0]*rank2 for row in V1] + [[0]*rank1 + row[:] for row in V2]
    W = [row[:] + [0]*rank2 for row in W1] + [[0]*rank1 + row[:] for row in W2]
    return U, V, W

def _strassen_2x2x2():
    """Return Strassen decomposition for 2x2x2 matrix multiplication (rank 7)."""
    # Strassen's 7 products for 2x2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[i,j],[k,l]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)(e)
    # m3 = (a)(f-h)
    # m4 = (d)(g-c)
    # m5 = (a+b)(h)
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C11 = m1+m4-m5+m7
    # C12 = m3+m5
    # C21 = m2+m4
    # C22 = m1-m2+m3+m6
    
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r] with n=m=p=2
    # i in {0,1}, j in {0,1}, k in {0,1}
    # U index: i*2+j, V index: j*2+k, W index: k*2+i
    
    # We need: sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] = 1 for all i,j,k
    
    # Let's use the standard Strassen formulas.
    # The 7 rank-1 terms:
    # r=0: m1 = (a+d)(e+h) -> U[0,0]=1,a; U[1,1]=1,d; V[0,0]=1,e; V[1,1]=1,h; W[0,0]=1,i; W[1,1]=1,l
    #   Actually: m1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    #   U[0,0]=1, U[1,1]=1 (for the A part: a and d)
    #   V[0,0]=1, V[1,1]=1 (for the B part: e and h)
    #   W[0,0]=1, W[1,1]=1 (for the C part: i and l)
    
    # Let me be more careful. The tensor T[i,j,k]=1 for 2x2x2.
    # Strassen gives 7 rank-1 tensors whose sum equals T.
    # Each rank-1 tensor is u * v * w where u is a 4-vector, v is a 4-vector, w is a 4-vector.
    
    # Standard Strassen (using the common formulation):
    # Let A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # p1 = (a+d)(e+h)
    # p2 = (c+d)(e)
    # p3 = a(f-h)
    # p4 = d(g-c)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    
    # C[0,0] = p1 + p4 - p5 + p7
    # C[0,1] = p3 + p5
    # C[1,0] = p2 + p4
    # C[1,1] = p1 - p2 + p3 + p6
    
    # For each p_r, we need U, V, W vectors such that sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = T[i,j,k]
    # T[i,j,k] = 1 for all i,j,k.
    
    # p1 = (a+d)(e+h): U[0,0]=1, U[1,1]=1, U[0,1]=0, U[1,0]=0
    #                        V[0,0]=1, V[1,1]=1, V[0,1]=0, V[1,0]=0
    #                        W[0,0]=1, W[1,1]=1, W[0,1]=0, W[1,0]=0
    # p2 = (c+d)(e): U[1,0]=1, U[1,1]=1, U[0,0]=0, U[0,1]=0
    #                     V[0,0]=1, V[0,1]=0, V[1,0]=0, V[1,1]=0
    #                     W: C[1,0] gets p2, so W[1,0]=1 (k=1,i=0 -> W[1*2+0]=W[2]=1)
    # p3 = a(f-h): U[0,0]=1
    #                  V[0,1]=1, V[1,1]=-1
    #                  W: C[0,1] gets p3, so W[0,1]=1 (k=0,i=1 -> W[0*2+1]=W[1]=1)
    # p4 = d(g-c): U[1,1]=1
    #                  V[1,0]=1, V[0,0]=-1
    #                  W: C[1,0] gets p4, so W[1,0]=1
    # p5 = (a+b)h: U[0,0]=1, U[0,1]=1
    #                   V[1,1]=1
    #                   W: C[0,0] gets -p5, C[0,1] gets p5
    #                   W[0,0]=-1, W[0,1]=1
    # p6 = (c-a)(e+f): U[1,0]=1, U[0,0]=-1
    #                      V[0,0]=1, V[0,1]=1
    #                      W: C[1,1] gets p6, so W[1,1]=1
    # p7 = (b-d)(g+h): U[0,1]=1, U[1,1]=-1
    #                      V[1,0]=1, V[1,1]=1
    #                      W: C[0,0] gets p7, so W[0,0]=1
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # p1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # p2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1  # k=1, i=0 -> W[1*2+0] = W[2]
    
    # p3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1  # k=0, i=1 -> W[0*2+1] = W[1]
    
    # p4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1  # k=1, i=0 -> W[2]
    
    # p5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1  # k=0,i=0 -> W[0]=-1; k=0,i=1 -> W[1]=1
    
    # p6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1  # k=1, i=1 -> W[1*2+1] = W[3]
    
    # p7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1  # k=0, i=0 -> W[0]
    
    return U, V, W

def construct():
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Strategy: Use Strassen for the 2x2x2 blocks within the multiplication.
    # The 2x3x8 multiplication can be viewed as:
    # A (2x3) * B (3x8) = C (2x8)
    # Split B into two 3x4 blocks: B = [B1 | B2]
    # C = A * B1 | A * B2
    # Each A*B_i is 2x3 * 3x4 = 2x4
    # Further split: A = [A_left (2x1) | A_mid (2x1) | A_right (2x1)]
    # Actually, let's think differently.
    
    # Approach: Split the 3 middle dimension.
    # A is 2x3, B is 3x8.
    # We can split B into B1 (3x4) and B2 (3x4).
    # Then C = A*B1 (2x4) concatenated with A*B2 (2x4).
    # For each 2x3 * 3x4 multiplication, we can further decompose.
    
    # Alternative: Use the additive property of tensor rank for direct sums.
    # Split the output into independent parts.
    
    # Let's try: split the 8 columns into two groups of 4.
    # Part 1: 2x3 * 3x4 (output 2x4)
    # Part 2: 2x3 * 3x4 (output 2x4)
    # Total rank = rank(2x3x4) + rank(2x3x4) = 2 * rank(2x3x4)
    
    # For 2x3x4: split the 4 columns into two groups of 2.
    # 2x3 * 3x2 (output 2x2) twice.
    # rank(2x3x4) = 2 * rank(2x3x2)
    
    # For 2x3x2: A is 2x3, B is 3x2, C is 2x2.
    # Split A into A1 (2x1) and A2 (2x2).
    # C = A1*B1 + A2*B2 where B1 is first column of B, B2 is second column.
    # Wait, B is 3x2, so B1 is 3x1, B2 is 3x1.
    # C[:,0] = A * B[:,0] (2x1)
    # C[:,1] = A * B[:,1] (2x1)
    # These are independent (different output columns), so:
    # rank(2x3x2) = rank(2x3x1) + rank(2x3x1)
    
    # For 2x3x1: A is 2x3, B is 3x1, C is 2x1.
    # This is just a matrix-vector product, which is rank 2*3*1 = 6 (schoolbook).
    # Can we do better? The tensor T[i,j,0] = 1 for i in {0,1}, j in {0,1,2}.
    # This is the tensor for computing [a0*b0+a1*b1+a2*b2, a0*b0+a1*b1+a2*b2]... 
    # No wait, C[0] = A[0,0]*B[0] + A[0,1]*B[1] + A[0,2]*B[2]
    # C[1] = A[1,0]*B[0] + A[1,1]*B[1] + A[1,2]*B[2]
    # The tensor is T[i,j,0] = 1 for all i,j. This is just the outer product structure.
    # rank = 2*3 = 6 (can't do better since each output depends on all 3 inputs independently).
    
    # So rank(2x3x2) = 6 + 6 = 12
    # rank(2x3x4) = 12 + 12 = 24
    # rank(2x3x8) = 24 + 24 = 48 (back to schoolbook!)
    
    # This approach doesn't help. Let me try Strassen on the 2x2 substructure.
    
    # Split the 2 rows: A = [A_top (1x3); A_bottom (1x3)]
    # Split the 8 columns: B = [B_1 (3x4) | B_2 (3x4)]
    # C_top = A_top * B (1x8)
    # C_bottom = A_bottom * B (1x8)
    # These are independent, so rank = rank(1x3x8) + rank(1x3x8) = 2 * 3 * 8 = 48. Still schoolbook.
    
    # Better: Use Strassen on the 2x2 blocks by splitting both rows and columns.
    # A = [[A11 (1x2), A12 (1x1)], [A21 (1x2), A22 (1x1)]]
    # B = [[B11 (2x4)], [B12 (2x4)], [B21 (1x4)], [B22 (1x4)]]
    # This is getting complicated. Let me use the known result that Strassen gives rank 7 for 2x2x2.
    
    # Strategy: Split the 3 middle dimension into 1+2.
    # A = [A1 (2x1) | A2 (2x2)]
    # B = [B1 (1x8); B2 (2x8)]
    # C = A1*B1 + A2*B2 (2x8)
    # rank = rank(2x1x8) + rank(2x2x8)
    # rank(2x1x8) = 2*1*8 = 16 (schoolbook, can't improve)
    # For rank(2x2x8): split the 8 columns into four groups of 2.
    # rank(2x2x8) = 4 * rank(2x2x2) = 4 * 7 = 28 (using Strassen)
    # Total rank = 16 + 28 = 44.
    
    # Let's verify this and construct it.
    
    # First, construct the 2x1x8 part (schoolbook, rank 16)
    U1 = [[0]*16 for _ in range(2*1)]
    V1 = [[0]*16 for _ in range(1*8)]
    W1 = [[0]*16 for _ in range(8*2)]
    col = 0
    for i in range(2):
        for k in range(8):
            U1[i][col] = 1
            V1[k][col] = 1
            W1[k*2+i][col] = 1
            col += 1
    
    # Now construct the 2x2x8 part using Strassen on four 2x2x2 blocks
    # Split the 8 columns into 4 groups of 2: columns 0-1, 2-3, 4-5, 6-7
    U2_blocks = []
    V2_blocks = []
    W2_blocks = []
    for block_idx in range(4):
        U_s, V_s, W_s = _strassen_2x2x2()
        U2_blocks.append(U_s)
        V2_blocks.append(V_s)
        W2_blocks.append(W_s)
    
    # Combine the 4 Strassen blocks
    U2 = []
    V2 = []
    W2 = []
    total_rank_2 = 0
    for b in range(4):
        r = len(U2_blocks[b][0])
        for i in range(4):
            if i < len(U2):
                U2[i].extend(U2_blocks[b][i])
            else:
                U2.append([0]*total_rank_2 + U2_blocks[b][i][:])
        for i in range(4):
            if i < len(V2):
                V2[i].extend(V2_blocks[b][i])
            else:
                V2.append([0]*total_rank_2 + V2_blocks[b][i][:])
        for i in range(4):
            if i < len(W2):
                W2[i].extend(W2_blocks[b][i])
            else:
                W2.append([0]*total_rank_2 + W2_blocks[b][i][:])
        total_rank_2 += r
    
    # Now combine part 1 (2x1x8) and part 2 (2x2x8)
    # Part 1: U1 is 2x16, V1 is 8x16, W1 is 16x16
    # Part 2: U2 is 4x28, V2 is 4x28, W2 is 4x28
    # 
    # The full problem: A is 2x3, B is 3x8
    # A = [A1 (2x1) | A2 (2x2)]
    # B = [B1 (1x8); B2 (2x8)]
    # 
    # U for full: indices (i, j) where i in {0,1}, j in {0,1,2}
    #   j=0 -> part 1, j=1,2 -> part 2
    # V for full: indices (j, k) where j in {0,1,2}, k in {0,...,7}
    #   j=0 -> part 1, j=1,2 -> part 2
    # W for full: indices (k, i) where k in {0,...,7}, i in {0,1}
    
    rank_1 = 16
    rank_2 = 28
    total_rank = rank_1 + rank_2
    
    U_full = [[0]*total_rank for _ in range(2*3)]  # 6 rows
    V_full = [[0]*total_rank for _ in range(3*8)]  # 24 rows
    W_full = [[0]*total_rank for _ in range(8*2)]  # 16 rows
    
    # Fill part 1 (j=0)
    for i in range(2):
        for r in range(rank_1):
            U_full[i*3 + 0][r] = U1[i][r]
    for k in range(8):
        for r in range(rank_1):
            V_full[0*8 + k][r] = V1[k][r]
    for k in range(8):
        for i in range(2):
            for r in range(rank_1):
                W_full[k*2 + i][r] = W1[k*2 + i][r]
    
    # Fill part 2 (j=1,2)
    for i in range(2):
        for j in range(2):
            for r in range(rank_2):
                U_full[i*3 + (1+j)][rank_1 + r] = U2[i*2 + j][r]
    for j in range(2):
        for k in range(8):
            for r in range(rank_2):
                V_full[(1+j)*8 + k][rank_1 + r] = V2[j*2 + 0][r] if False else 0
    # Wait, V2 is for 2x2x8 where the middle dimension is 2 (j in {0,1} for the 2x2 part)
    # V2[j2*2 + k2, r] where j2 in {0,1}, k2 in {0,1} for each 2x2x2 block
    # But we have 4 blocks, each 2x2x2, combined into 2x2x8
    # V2 has shape (2*8, 28) = (16, 28)? No, V2 should be (m*p, rank) = (2*8, 28) = (16, 28)
    # Wait, I made an error. Let me reconsider.
    
    # For 2x2x8: n=2, m=2, p=8
    # U: (n*m, rank) = (4, 28)
    # V: (m*p, rank) = (16, 28)
    # W: (p*n, rank) = (16, 28)
    
    # But I constructed V2 as 4x28, which is wrong. Let me redo.
    
    # For each 2x2x2 Strassen block handling columns 2*b to 2*b+1:
    # U_block: (4, 7), V_block: (4, 7), W_block: (4, 7)
    # To combine 4 blocks into 2x2x8:
    # U_full_2: (4, 28) - same for all blocks
    # V_full_2: (16, 28) - V_block for block b goes into rows [2*b : 2*b+2, :]
    # W_full_2: (16, 28) - W_block for block b goes into rows [2*b : 2*b+2, :]
    
    U2_correct = [[0]*28 for _ in range(4)]
    V2_correct = [[0]*28 for _ in range(16)]
    W2_correct = [[0]*28 for _ in range(16)]
    
    for b in range(4):
        U_s, V_s, W_s = _strassen_2x2x2()
        for i in range(4):
            for r in range(7):
                U2_correct[i][b*7+r] = U_s[i][r]
        for j in range(2):
            for k in range(2):
                for r in range(7):
                    V2_correct[(j*2+k)*1 + b*2 + 0][b*7+r] = 0  # placeholder
                    # V_s[j*2+k, r] goes to V2_correct[(j*2+k)*1 + 2*b + 0... ]
                    # Actually V for 2x2x2 has shape (m*p, rank) = (4, 7), indexed by j*2+k
                    # For the full 2x2x8, V has shape (16, 28), indexed by j*8+k where j in {0,1}, k in {0..7}
                    # Block b handles k in {2b, 2b+1}
                    # So V_s[j*2+k2, r] -> V2_correct[j*8 + 2*b + k2, b*7+r]
        for j in range(2):
            for k2 in range(2):
                for r in range(7):
                    V2_correct[j*8 + 2*b + k2][b*7+r] = V_s[j*2+k2][r]
        for k2 in range(2):
            for i in range(2):
                for r in range(7):
                    W2_correct[k2*2 + i][b*7+r] = 0  # placeholder
                    # W_s[k2*2+i, r] -> W2_correct[(2*b+k2)*2 + i, b*7+r]
        for k2 in range(2):
            for i in range(2):
                for r in range(7):
                    W2_correct[(2*b+k2)*2 + i][b*7+r] = W_s[k2*2+i][r]
    
    # Now fill U_full, V_full, W_full
    U_full = [[0]*total_rank for _ in range(6)]
    V_full = [[0]*total_rank for _ in range(24)]
    W_full = [[0]*total_rank for _ in range(16)]
    
    # Part 1: j=0
    for i in range(2):
        for r in range(rank_1):
            U_full[i*3 + 0][r] = U1[i][r]
    for k in range(8):
        for r in range(rank_1):
            V_full[0*8 + k][r] = V1[k][r]
    for k in range(8):
        for i in range(2):
            for r in range(rank_1):
                W_full[k*2 + i][r] = W1[k*2 + i][r]
    
    # Part 2: j=1,2
    for i in range(2):
        for j2 in range(2):
            for r in range(rank_2):
                U_full[i*3 + (1+j2)][rank_1 + r] = U2_correct[i*2 + j2][r]
    for j2 in range(2):
        for k in range(8):
            for r in range(rank_2):
                V_full[(1+j2)*8 + k][rank_1 + r] = V2_correct[j2*8 + k][r]
    for k in range(8):
        for i in range(2):
            for r in range(rank_2):
                W_full[k*2 + i][rank_1 + r] = W2_correct[k*2 + i][r]
    
    result = {"U": U_full, "V": V_full, "W": W_full}
    
    # Validate
    if _validate(U_full, V_full, W_full, n, m, p):
        return result
    
    # Fallback to schoolbook
    return _make_trivial(n, m, p)
# EVOLVE-BLOCK-END
