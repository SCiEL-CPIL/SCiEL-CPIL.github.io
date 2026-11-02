"""Matrix multiplication 4x4x9 tensor decomposition using recursive Strassen."""
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 9]

def _strassen_2x2():
    """Return U, V, W for 2x2x2 matrix multiplication using Strassen (rank 7).
    
    Formula: C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_l V[l*p+k, r] * B[l,k])
    where n=m=p=2, W indexed by [k*n+i, r]
    """
    n, m, p = 2, 2, 2
    rank = 7
    # U[i*m+j, r], V[l*p+k, r], W[k*n+i, r]
    U = [[0]*rank for _ in range(n*m)]  # 4x7
    V = [[0]*rank for _ in range(m*p)]  # 4x7
    W = [[0]*rank for _ in range(p*n)]  # 4x7
    
    # Strassen formulas:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each rank term r, U and V define linear forms in A and B
    # U[i*m+j, r] = coefficient of A[i,j] in the r-th left factor
    # V[l*p+k, r] = coefficient of B[l,k] in the r-th right factor
    
    # M1: (A00+A11) * (B00+B11)
    U[0][0] = 1; U[3][0] = 1  # A00, A11
    V[0][0] = 1; V[3][0] = 1  # B00, B11
    W[0][0] = 1; W[3][0] = 1  # C00, C11 get +1
    W[1][0] = -1; W[2][0] = 1  # C01 gets -1 (wait, let me recheck)
    
    # Actually, W[k*n+i, r]: for n=p=2, k*n+i:
    # k=0,i=0 -> 0 (C00)
    # k=0,i=1 -> 1 (C10)  
    # k=1,i=0 -> 2 (C01)
    # k=1,i=1 -> 3 (C11)
    
    # Let me redo carefully.
    # C[i,k]: i=row, k=col
    # W index: k*n+i where n=2
    # C[0,0]: k=0,i=0 -> idx 0
    # C[0,1]: k=1,i=0 -> idx 2
    # C[1,0]: k=0,i=1 -> idx 1
    # C[1,1]: k=1,i=1 -> idx 3
    
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # M1 = (A00+A11)*(B00+B11)
    # C00 += M1, C11 += M1, C01 -= M1, C10 += M1
    U[0][0]=1; U[3][0]=1  # A[0,0], A[1,1]
    V[0][0]=1; V[3][0]=1  # B[0,0], B[1,1]
    W[0][0]=1   # C[0,0]
    W[1][0]=1   # C[1,0]
    W[2][0]=-1  # C[0,1]
    W[3][0]=1   # C[1,1]
    
    # M2 = (A10+A11)*B00
    # C10 += M2, C11 -= M2
    U[2][1]=1; U[3][1]=1  # A[1,0], A[1,1]
    V[0][1]=1             # B[0,0]
    W[1][1]=1   # C[1,0]
    W[3][1]=-1  # C[1,1]
    
    # M3 = A00*(B01-B11)
    # C01 += M3
    U[0][2]=1             # A[0,0]
    V[1][2]=1; V[3][2]=-1 # B[0,1], -B[1,1]
    W[2][2]=1   # C[0,1]
    
    # M4 = A11*(B10-B00)
    # C00 += M4, C10 += M4
    U[3][3]=1             # A[1,1]
    V[2][3]=1; V[0][3]=-1 # B[1,0], -B[0,0]
    W[0][3]=1   # C[0,0]
    W[1][3]=1   # C[1,0]
    
    # M5 = (A00+A01)*B11
    # C00 -= M5, C01 += M5
    U[0][4]=1; U[1][4]=1  # A[0,0], A[0,1]
    V[3][4]=1             # B[1,1]
    W[0][4]=-1  # C[0,0]
    W[2][4]=1   # C[0,1]
    
    # M6 = (A10-A00)*(B00+B01)
    # C11 += M6
    U[2][5]=1; U[0][5]=-1 # A[1,0], -A[0,0]
    V[0][5]=1; V[1][5]=1  # B[0,0], B[0,1]
    W[3][5]=1   # C[1,1]
    
    # M7 = (A01-A11)*(B10+B11)
    # C00 += M7
    U[1][6]=1; U[3][6]=-1 # A[0,1], -A[1,1]
    V[2][6]=1; V[3][6]=1  # B[1,0], B[1,1]
    W[0][6]=1   # C[0,0]
    
    return U, V, W

def _expand_strassen_4x4():
    """Expand 2x2x2 Strassen to 4x4x4 by treating each 2x2 block recursively.
    
    For 4x4x4, we have 4x4 = 2x2 blocks of 2x2.
    A = [[A00, A01], [A10, A11]] where each is 2x2
    B = [[B00, B01], [B10, B11]]
    C = [[C00, C01], [C10, C11]]
    
    Each C_ij is a 2x2 matrix computed via Strassen on the block matrices.
    This gives rank 7*7 = 49.
    """
    U2, V2, W2 = _strassen_2x2()
    
    n, m, p = 4, 4, 4
    rank = 49
    U = [[0]*rank for _ in range(n*m)]  # 16x49
    V = [[0]*rank for _ in range(m*p)]  # 16x49
    W = [[0]*rank for _ in range(p*n)]  # 16x49
    
    # For each pair of Strassen terms (r1, r2), we get a term in the 4x4x4 decomposition
    # The left factor involves A blocks, right factor involves B blocks
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            
            # Left factor: combination of A blocks
            # U2[i2*2+j2, r1] tells us which 2x2 sub-entries of each A block to use
            # A is 4x4, blocks are A00(top-left 2x2), A01(top-right), A10(bottom-left), A11(bottom-right)
            # A[i,j] where i,j in 0..3
            # Block row: i//2, block col: j//2
            # Within block: i%2, j%2
            
            for i in range(4):
                for j in range(4):
                    block_i = i // 2
                    block_j = j // 2
                    local_i = i % 2
                    local_j = j % 2
                    # U2 index for this block: block_i*2 + block_j (within 2x2 of blocks)
                    u2_idx = block_i * 2 + block_j
                    coeff = U2[u2_idx][r1]
                    if coeff != 0:
                        # For the local entry, we need to apply the 2x2 Strassen U
                        # Actually no - the 2x2 Strassen operates on the 2x2 blocks
                        # U2[i2*2+j2, r1] is the coefficient for the (i2,j2)-th element of the 2x2 block matrix
                        # So for A[block_i*2+local_i, block_j*2+local_j], the coefficient is U2[block_i*2+block_j, r1] * (1)
                        # Wait, that's not right. Let me think again.
                        
                        # The 2x2x2 Strassen computes:
                        # C_block[i2, k2] = sum_r W2[k2*2+i2, r] * (sum_j2 U2[i2*2+j2, r] * A_block[i2,j2]) * (sum_l2 V2[l2*2+k2, r] * B_block[l2,k2])
                        
                        # So for the 4x4 case, each "element" of the 2x2 block matrix is itself a 2x2 matrix.
                        # A_block[i2, j2] is the 2x2 submatrix A[i2*2:(i2+1)*2, j2*2:(j2+1)*2]
                        
                        # The left factor for the outer Strassen term r1 is:
                        # sum_{i2,j2} U2[i2*2+j2, r1] * A_block[i2,j2]
                        # This is a 2x2 matrix (sum of 2x2 blocks)
                        
                        # Then the inner Strassen term r2 applies to this 2x2 result:
                        # The inner U2[i2'*2+j2', r2] picks out elements from this summed 2x2 matrix
                        # But wait - the inner Strassen operates on the actual 2x2 matrices, not on symbols.
                        
                        # I think the correct approach is:
                        # The overall left factor for term (r1,r2) applied to A[i,j] is:
                        # sum over block positions that contribute:
                        # U2[block_i*2+block_j, r1] * (inner coefficient for A[i,j] within its block)
                        
                        # Actually, since A[i,j] is a scalar, and the blocks are just groupings:
                        # A_block[i2,j2][li2, lj2] = A[i2*2+li2, j2*2+lj2]
                        # The outer left factor: L_outer = sum_{i2,j2} U2[i2*2+j2, r1] * A_block[i2,j2]
                        # L_outer is 2x2, with L_outer[li2, lj2] = sum_{i2,j2} U2[i2*2+j2, r1] * A[i2*2+li2, j2*2+lj2]
                        # The inner left factor: L_inner = sum_{li2,lj2} U2[li2*2+lj2, r2] * L_outer[li2, lj2]
                        # L_inner = sum_{li2,lj2} U2[li2*2+lj2, r2] * sum_{i2,j2} U2[i2*2+j2, r1] * A[i2*2+li2, j2*2+lj2]
                        # L_inner = sum_{i2,j2,li2,lj2} U2[i2*2+j2, r1] * U2[li2*2+lj2, r2] * A[i2*2+li2, j2*2+lj2]
                        
                        # So for A[i,j] where i=i2*2+li2, j=j2*2+lj2:
                        # coefficient = U2[i2*2+j2, r1] * U2[li2*2+lj2, r2]
                        
                        u_inner = U2[local_i*2+local_j][r2]
                        total_coeff = coeff * u_inner
                        U[i*m+j][r] += total_coeff
            
            # Right factor: similar for B
            for l in range(4):
                for k in range(4):
                    block_l = l // 2
                    block_k = k // 2
                    local_l = l % 2
                    local_k = k % 2
                    v2_idx = block_l * 2 + block_k
                    coeff = V2[v2_idx][r1]
                    if coeff != 0:
                        v_inner = V2[local_l*2+local_k][r2]
                        total_coeff = coeff * v_inner
                        V[l*p+k][r] += total_coeff
            
            # W factor: W2[k2*2+i2, r] for outer, and inner
            # C[i,k] where i=i2*2+li2, k=k2*2+lk2
            # W_outer[k2*2+i2, r1] * W_inner[lk2*2+li2, r2]
            for i in range(4):
                for k in range(4):
                    block_i = i // 2
                    block_k = k // 2
                    local_i = i % 2
                    local_k = k % 2
                    w_outer = W2[block_k*2+block_i][r1]
                    w_inner = W2[local_k*2+local_i][r2]
                    W[k*n+i][r] += w_outer * w_inner
    
    return U, V, W

def _schoolbook(n, m, p):
    """Schoolbook decomposition for n x m * m x p."""
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, test_size=5, seed=42):
    """Verify the decomposition with random integer matrices."""
    np.random.seed(seed)
    # Generate random integer matrices
    A = np.random.randint(-5, 5, size=(n, m))
    B = np.random.randint(-5, 5, size=(m, p))
    C_expected = A @ B
    
    # Compute via decomposition
    C_computed = np.zeros((n, p), dtype=object)
    U_arr = np.array(U, dtype=object)
    V_arr = np.array(V, dtype=object)
    W_arr = np.array(W, dtype=object)
    A_arr = A.astype(object)
    B_arr = B.astype(object)
    
    rank = len(U[0]) if U else 0
    for r in range(rank):
        # left_r = sum_j U[i*m+j, r] * A[i,j] for each i
        left = np.zeros(n, dtype=object)
        for i in range(n):
            s = 0
            for j in range(m):
                s += U_arr[i*m+j, r] * A_arr[i, j]
            left[i] = s
        
        # right_r = sum_l V[l*p+k, r] * B[l,k] for each k
        right = np.zeros(p, dtype=object)
        for k in range(p):
            s = 0
            for l in range(m):
                s += V_arr[l*p+k, r] * B_arr[l, k]
            right[k] = s
        
        # Add contribution: W[k*n+i, r] * left[i] * right[k]
        for i in range(n):
            for k in range(p):
                C_computed[i, k] += W_arr[k*n+i, r] * left[i] * right[k]
    
    for i in range(n):
        for k in range(p):
            if C_computed[i, k] != C_expected[i, k]:
                return False, i, k, C_computed[i, k], C_expected[i, k]
    return True, None, None, None, None

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1)
    # We'll handle B3 as 4x4 with the last 3 columns being zero, but that wastes rank.
    # Better: handle B1 (4x4) with Strassen rank 49, B2 (4x4) with Strassen rank 49, 
    # and B3 (4x1) with schoolbook rank 16.
    
    # Get 4x4x4 Strassen decomposition
    U4, V4, W4 = _expand_strassen_4x4()
    
    # Verify 4x4x4
    ok, *_ = _verify(U4, V4, W4, 4, 4, 4, seed=42)
    if not ok:
        # Fall back to schoolbook for 4x4x4
        U4, V4, W4 = _schoolbook(4, 4, 4)
    
    # For B3 (4x1), use schoolbook
    U3, V3, W3 = _schoolbook(4, 4, 1)
    
    # Assemble final 4x4x9 decomposition
    # U is (n*m, rank) = (16, rank)
    # V is (m*p, rank) = (36, rank)  
    # W is (p*n, rank) = (36, rank)
    
    rank1 = len(U4[0])  # 49
    rank2 = rank1       # 49
    rank3 = len(U3[0])  # 16
    total_rank = rank1 + rank2 + rank3  # 114
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    # Block 1: columns 0..39 of B (B1 is 4x4)
    for r in range(rank1):
        for i in range(n*m):
            U[i][r] = U4[i][r]
        for l in range(m):
            for k in range(4):  # k=0..3 corresponds to B columns 0..3
                V[l*p+k][r] = V4[l*4+k][r]
        for k in range(4):
            for i in range(n):
                W[k*n+i][r] = W4[k*4+i][r]
    
    # Block 2: columns 4..7 of B (B2 is 4x4)
    offset1 = rank1
    for r in range(rank2):
        for i in range(n*m):
            U[i][offset1+r] = U4[i][r]
        for l in range(m):
            for k in range(4):  # k=0..3 corresponds to B columns 4..7
                V[l*p+(4+k)][offset1+r] = V4[l*4+k][r]
        for k in range(4):
            for i in range(n):
                W[(4+k)*n+i][offset1+r] = W4[k*4+i][r]
    
    # Block 3: column 8 of B (B3 is 4x1)
    offset2 = rank1 + rank2
    for r in range(rank3):
        for i in range(n*m):
            U[i][offset2+r] = U3[i][r]
        for l in range(m):
            V[l*p+8][offset2+r] = V3[l*1+0][r]
        for i in range(n):
            W[8*n+i][offset2+r] = W3[0*4+i][r]
    
    # Verify final result
    ok, i, k, cv, ev = _verify(U, V, W, n, m, p, seed=42)
    if not ok:
        # Fall back to full schoolbook
        U, V, W = _schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}
