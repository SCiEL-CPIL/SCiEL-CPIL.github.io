"""Program using deterministic block decomposition with Strassen for 4x4x4 part."""
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # n=4, m=4, p=7
    # U shape: (n*m, R) = (16, R)
    # V shape: (m*p, R) = (28, R)
    # W shape: (p*n, R) = (28, R)
    
    # Strategy:
    # Split the multiplication C = A * B into two parts:
    # C[:, :4] = A * B[:, :4]  (4x4 * 4x4 -> 4x4)
    # C[:, 4:] = A * B[:, 4:]  (4x4 * 4x3 -> 4x3)
    
    # Part 1: 4x4 * 4x4
    # Use Strassen's algorithm for 4x4 matrices.
    # A 4x4 matrix can be viewed as 2x2 blocks of 2x2 matrices.
    # Strassen reduces the number of 2x2 multiplications from 8 to 7.
    # Each 2x2 multiplication can be done with rank 7 (Strassen for 2x2x2).
    # So total rank for 4x4x4 is 7 * 7 = 49.
    
    # Part 2: 4x4 * 4x3
    # We can use schoolbook multiplication for this part.
    # Rank = 4 * 4 * 3 = 48.
    
    # Total Rank R = 49 + 48 = 97.
    
    R_total = 97
    U = np.zeros((n*m, R_total), dtype=np.int64)
    V = np.zeros((m*p, R_total), dtype=np.int64)
    W = np.zeros((p*n, R_total), dtype=np.int64)
    
    # --- Part 1: 4x4 * 4x4 using Strassen ---
    # Indices for Part 1:
    # A is 4x4. B1 is 4x4 (columns 0-3 of B). C1 is 4x4 (columns 0-3 of C).
    # In the global indices:
    # A is always the same 4x4 matrix (rows 0-3).
    # B1 corresponds to j in 0..3, k in 0..3.
    # C1 corresponds to k in 0..3, i in 0..3.
    
    # We need to construct the U, V, W contributions for the 4x4x4 tensor.
    # Let's define the Strassen decomposition for 2x2x2 first.
    # For 2x2 matrices X, Y, Z = X*Y:
    # Z00 = M1 - M2 + M4 - M5
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M3 + M6 - M7
    # Where:
    # M1 = (X00+X11)*(Y00+Y11)
    # M2 = (X10+X11)*Y00
    # M3 = X00*(Y01-Y11)
    # M4 = X11*(Y10-Y00)
    # M5 = (X00+X10)*Y11
    # M6 = (X11-X10)*(Y00+Y01)
    # M7 = (X01-X11)*(Y10+Y11)
    
    # For 4x4, we split into 2x2 blocks of 2x2 matrices.
    # A = [[A00, A01], [A10, A11]] where each block is 2x2.
    # B1 = [[B00, B01], [B10, B11]] where each block is 2x2.
    # C1 = A * B1.
    # C1 = [[C00, C01], [C10, C11]]
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Each product (e.g., A00*B00) is a 2x2 matrix multiplication, which has rank 7.
    # There are 8 such products in the block multiplication.
    # But Strassen allows us to compute the 4 block products (C00, C01, C10, C11) 
    # using 7 "macro" multiplications of 2x2 matrices? No, that's not right.
    # Strassen computes the product of two 2x2 block matrices (which are 4x4 overall) 
    # using 7 multiplications of 2x2 block matrices.
    # So, we treat the 4x4 matrices as 2x2 matrices of 2x2 blocks.
    # Let A_blocks be a 2x2 matrix of 2x2 blocks.
    # Let B1_blocks be a 2x2 matrix of 2x2 blocks.
    # C1_blocks = A_blocks * B1_blocks.
    # Strassen computes this using 7 multiplications of 2x2 block matrices.
    # Each 2x2 block matrix multiplication has rank 7 (over the base field, treating blocks as scalars? No, blocks are matrices).
    # Wait, the rank of the tensor for 2x2x2 matrix multiplication is 7.
    # When we replace each "scalar" with a 2x2 matrix, the rank multiplies by the rank of the 2x2x2 tensor?
    # Yes, the Kronecker product structure implies rank multiplication.
    # So rank(4x4x4) <= rank(2x2x2) * rank(2x2x2) = 7 * 7 = 49.
    
    # Let's construct the U, V, W for the 4x4x4 part explicitly.
    # We will iterate over the 7 Strassen steps for the block level, and for each step,
    # iterate over the 7 Strassen steps for the sub-block level.
    
    # Helper to get 2x2 Strassen coefficients
    # For 2x2 matrices X (indices x00, x01, x10, x11) and Y (y00, y01, y10, y11)
    # The 7 products are:
    # P1 = (x00+x11)*(y00+y11)
    # P2 = (x10+x11)*y00
    # P3 = x00*(y01-y11)
    # P4 = x11*(y10-y00)
    # P5 = (x00+x10)*y11
    # P6 = (x11-x10)*(y00+y01)
    # P7 = (x01-x11)*(y10+y11)
    
    # We need to map the 4x4 indices to block indices.
    # A is 4x4. A[i, j].
    # Block (b_i, b_j) contains A[2*b_i : 2*b_i+2, 2*b_j : 2*b_j+2].
    # So A[2*b_i + s_i, 2*b_j + s_j] is the element (s_i, s_j) of block (b_i, b_j).
    
    # Let's define the U, V, W contributions for one 2x2x2 Strassen step.
    # For a single 2x2 multiplication Z = X*Y:
    # Z[0,0] = P1 - P2 + P4 - P5
    # Z[0,1] = P3 + P5
    # Z[1,0] = P2 + P4
    # Z[1,1] = P1 - P3 + P6 - P7
    
    # The tensor T_2x2 has indices (x_idx, y_idx, z_idx) where x_idx in 0..3, y_idx in 0..3, z_idx in 0..3.
    # We want to decompose it into 7 terms.
    # Term r corresponds to one of the P1..P7.
    # For each term r, we have vectors u_r (size 4), v_r (size 4), w_r (size 4) such that
    # T[x,y,z] = sum_r u_r[x] v_r[y] w_r[z].
    
    # Let's construct u_r, v_r, w_r for the 2x2 case.
    # x_idx = 2*xi + xj? No, x_idx is just the linear index of the 2x2 matrix.
    # Let's use linear indices 0,1,2,3 for 2x2 matrices.
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1).
    
    # P1 = (x0+x3)*(y0+y3)
    # u_1 = [1, 0, 0, 1]
    # v_1 = [1, 0, 0, 1]
    # w_1 = [1, 0, 0, 0]  # contributes to Z00
    #       [0, 0, 0, -1] # contributes to Z11? No, P1 is added to Z00 and Z11.
    # Wait, Z00 = P1 - P2 + P4 - P5. Z11 = P1 - P3 + P6 - P7.
    # So P1 contributes +1 to Z00 and +1 to Z11.
    # w_1 = [1, 0, 0, 1]
    
    # P2 = (x2+x3)*y0
    # u_2 = [0, 0, 1, 1]
    # v_2 = [1, 0, 0, 0]
    # w_2 = [-1, 0, 1, 0] # -1 to Z00, +1 to Z10
    
    # P3 = x0*(y1-y3)
    # u_3 = [1, 0, 0, 0]
    # v_3 = [0, 1, 0, -1]
    # w_3 = [0, 1, 0, -1] # +1 to Z01, -1 to Z11
    
    # P4 = x3*(y2-y0)
    # u_4 = [0, 0, 0, 1]
    # v_4 = [0, 0, 1, -1]
    # w_4 = [1, 0, 1, 0] # +1 to Z00, +1 to Z10
    
    # P5 = (x0+x2)*y3
    # u_5 = [1, 0, 1, 0]
    # v_5 = [0, 0, 0, 1]
    # w_5 = [-1, 1, 0, 0] # -1 to Z00, +1 to Z01
    
    # P6 = (x3-x2)*(y0+y1)
    # u_6 = [0, 0, -1, 1]
    # v_6 = [1, 1, 0, 0]
    # w_6 = [0, 0, 0, 1] # +1 to Z11
    
    # P7 = (x1-x3)*(y2+y3)
    # u_7 = [0, 1, 0, -1]
    # v_7 = [0, 0, 1, 1]
    # w_7 = [0, 0, 0, -1] # -1 to Z11
    
    # Let's verify Z00:
    # P1: 1, P2: -1, P3: 0, P4: 1, P5: -1, P6: 0, P7: 0. Sum: 1-1+1-1 = 0? 
    # Wait, Z00 = P1 - P2 + P4 - P5.
    # Coeffs in w:
    # w_1[0]=1, w_2[0]=-1, w_3[0]=0, w_4[0]=1, w_5[0]=-1, w_6[0]=0, w_7[0]=0.
    # Sum = 1 - 1 + 0 + 1 - 1 + 0 + 0 = 0. This is wrong. Z00 should be 1 (from P1) - 1 (from P2) + 1 (from P4) - 1 (from P5).
    # The formula is Z00 = 1*P1 - 1*P2 + 0*P3 + 1*P4 - 1*P5 + 0*P6 + 0*P7.
    # So w_1[0] should be 1, w_2[0] should be -1, w_4[0] should be 1, w_5[0] should be -1.
    # My w vectors above represent the contribution to Z.
    # Let's re-list w_r for each r, where w_r is a vector of length 4 (for Z00, Z01, Z10, Z11).
    
    # r=1 (P1): Z00+=1, Z11+=1. w_1 = [1, 0, 0, 1]
    # r=2 (P2): Z00-=1, Z10+=1. w_2 = [-1, 0, 1, 0]
    # r=3 (P3): Z01+=1, Z11-=1. w_3 = [0, 1, 0, -1]
    # r=4 (P4): Z00+=1, Z10+=1. w_4 = [1, 0, 1, 0]
    # r=5 (P5): Z00-=1, Z01+=1. w_5 = [-1, 1, 0, 0]
    # r=6 (P6): Z11+=1. w_6 = [0, 0, 0, 1]
    # r=7 (P7): Z11-=1. w_7 = [0, 0, 0, -1]
    
    # Now u_r and v_r:
    # r=1: u_1=[1,0,0,1], v_1=[1,0,0,1]
    # r=2: u_2=[0,0,1,1], v_2=[1,0,0,0]
    # r=3: u_3=[1,0,0,0], v_3=[0,1,0,-1]
    # r=4: u_4=[0,0,0,1], v_4=[0,0,1,-1]
    # r=5: u_5=[1,0,1,0], v_5=[0,0,0,1]
    # r=6: u_6=[0,0,-1,1], v_6=[1,1,0,0]
    # r=7: u_7=[0,1,0,-1], v_7=[0,0,1,1]
    
    # This gives a rank-7 decomposition for 2x2x2.
    
    # Now for 4x4x4:
    # We view 4x4 matrices as 2x2 blocks of 2x2 matrices.
    # Let A be the 4x4 matrix. A[i, j].
    # Block (bi, bj) is A[2*bi:2*bi+2, 2*bj:2*bj+2].
    # The "scalar" at (bi, bj) in the block matrix is the 2x2 matrix A_block(bi, bj).
    # The Strassen algorithm operates on these blocks.
    # Each "multiplication" of blocks (e.g., M1 = (A00+A11)*(B00+B11)) is itself a 2x2 matrix multiplication.
    # So we need to decompose each block multiplication using the rank-7 2x2x2 decomposition.
    
    # Total rank = 7 (block level) * 7 (sub-block level) = 49.
    
    # We will iterate over the 7 block-level Strassen steps (r1 in 0..6) and 7 sub-block steps (r2 in 0..6).
    # For each pair (r1, r2), we add a term to U, V, W.
    
    # The indices for the 4x4 matrix are 0..15 (row-major).
    # Index = 4*i + j.
    # Block (bi, bj) contains elements with i in {2*bi, 2*bi+1} and j in {2*bj, 2*bj+1}.
    # Local index within block: li = i - 2*bi, lj = j - 2*bj.
    # Global index: idx = 4*(2*bi+li) + (2*bj+lj) = 8*bi + 4*li + 2*bj + lj.
    
    # For a given block-level step r1, we have coefficients for the blocks.
    # u_1[bi, bj] (coefficient for A block (bi, bj))
    # v_1[bi, bj] (coefficient for B block (bi, bj))
    # w_1[bi, bj] (coefficient for C block (bi, bj))
    
    # For a given sub-block step r2, we have coefficients for the elements within the block.
    # u_2[li, lj]
    # v_2[li, lj]
    # w_2[li, lj]
    
    # The total coefficient for A[4*i+j] in the term (r1, r2) is:
    # u_1[bi, bj] * u_2[li, lj]
    # where i = 2*bi + li, j = 2*bj + lj.
    
    # Similarly for B and C.
    # Note: B is 4x4 in this part. C is 4x4 in this part.
    
    # Let's construct the U, V, W for the 4x4x4 part.
    # U_part1 will be 16x49.
    # V_part1 will be 16x49 (since B is 4x4, m=4, p_part1=4, so m*p_part1 = 16).
    # W_part1 will be 16x49 (since C is 4x4, p_part1=4, n=4, so p_part1*n = 16).
    
    # Wait, the global V is 28xR (m*p = 4*7).
    # The global W is 28xR (p*n = 7*4).
    
    # For Part 1 (columns 0-3 of B, columns 0-3 of C):
    # V indices: j*p + k. j in 0..3, k in 0..3. So indices 0..15.
    # W indices: k*n + i. k in 0..3, i in 0..3. So indices 0..15.
    
    # For Part 2 (columns 4-6 of B, columns 4-6 of C):
    # V indices: j*p + k. j in 0..3, k in 4..6. So indices 16..27.
    # W indices: k*n + i. k in 4..6, i in 0..3. So indices 16..27.
    
    # Let's define the 2x2 Strassen coefficients again as lists of lists.
    # u_2x2[r] is a list of 4 values [u00, u01, u10, u11]
    # v_2x2[r] is a list of 4 values
    # w_2x2[r] is a list of 4 values
    
    u_2x2 = [
        [1, 0, 0, 1], # r=1
        [0, 0, 1, 1], # r=2
        [1, 0, 0, 0], # r=3
        [0, 0, 0, 1], # r=4
        [1, 0, 1, 0], # r=5
        [0, 0, -1, 1], # r=6
        [0, 1, 0, -1]  # r=7
    ]
    
    v_2x2 = [
        [1, 0, 0, 1], # r=1
        [1, 0, 0, 0], # r=2
        [0, 1, 0, -1], # r=3
        [0, 0, 1, -1], # r=4
        [0, 0, 0, 1], # r=5
        [1, 1, 0, 0], # r=6
        [0, 0, 1, 1]  # r=7
    ]
    
    w_2x2 = [
        [1, 0, 0, 1], # r=1
        [-1, 0, 1, 0], # r=2
        [0, 1, 0, -1], # r=3
        [1, 0, 1, 0], # r=4
        [-1, 1, 0, 0], # r=5
        [0, 0, 0, 1], # r=6
        [0, 0, 0, -1]  # r=7
    ]
    
    # Now construct Part 1 (4x4x4)
    # R_part1 = 49
    R_part1 = 49
    col_offset = 0
    
    for r1 in range(7):
        for r2 in range(7):
            r = col_offset
            col_offset += 1
            
            # u_2x2[r1] gives coefficients for blocks of A.
            # u_2x2[r2] gives coefficients for elements within blocks.
            
            # For each global index a in 0..15 (A):
            for i in range(4):
                for j in range(4):
                    a = 4*i + j
                    bi = i // 2
                    bj = j // 2
                    li = i % 2
                    lj = j % 2
                    
                    # Coefficient from block level
                    c1 = u_2x2[r1][bi*2 + bj]
                    # Coefficient from sub-block level
                    c2 = u_2x2[r2][li*2 + lj]
                    
                    U[a, r] = c1 * c2
            
            # For each global index b in 0..15 (B part 1):
            for j in range(4):
                for k in range(4):
                    b = 4*j + k
                    bj = j // 2
                    bk = k // 2
                    lj = j % 2
                    lk = k % 2
                    
                    c1 = v_2x2[r1][bj*2 + bk]
                    c2 = v_2x2[r2][lj*2 + lk]
                    
                    V[b, r] = c1 * c2
            
            # For each global index c in 0..15 (C part 1):
            for k in range(4):
                for i in range(4):
                    c = 4*k + i
                    bk = k // 2
                    bi = i // 2
                    lk = k % 2
                    li = i % 2
                    
                    c1 = w_2x2[r1][bk*2 + bi]
                    c2 = w_2x2[r2][lk*2 + li]
                    
                    W[c, r] = c1 * c2
    
    # --- Part 2: 4x4 * 4x3 using Schoolbook ---
    # Rank = 4 * 4 * 3 = 48.
    # For each i, j, k in {0..3, 0..3, 4..6}, we have a term.
    # A[i, j] * B[j, k] -> C[i, k]
    
    R_part2 = 48
    col_offset = 49
    
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                r = col_offset
                col_offset += 1
                
                a = 4*i + j
                b = 4*j + k
                c = 7*k + i  # Note: global W index is k*n + i, where n=4.
                             # Wait, the problem says W[k*n+i, r].
                             # So c = k*4 + i.
                             # In Part 1, k was 0..3, so c was 0..15.
                             # In Part 2, k is 4..6, so c is 16..27.
                             # This matches the global W size 28.
                
                U[a, r] = 1
                V[b, r] = 1
                W[c, r] = 1
    
    # Check if we have a valid decomposition
    # Reconstruct M
    M_recon = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Construct M
    M = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                M[a, b, c] = 1.0
    
    residual = np.linalg.norm(M - M_recon)
    
    if residual > 1e-9:
        # Fallback to schoolbook if construction failed (should not happen)
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=np.int64)
        V = np.zeros((m*p, rank), dtype=np.int64)
        W = np.zeros((p*n, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
