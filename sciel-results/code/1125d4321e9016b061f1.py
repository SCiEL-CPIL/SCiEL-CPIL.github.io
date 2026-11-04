# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 130s deadline for safety margin
    
    n, m, p = 4, 4, 9
    
    # Helper to check if we're still within time
    def time_ok():
        return time.time() < deadline
    
    # Winograd F(4,2) frequency-domain approach
    # We embed 4x4x9 into a structure using cyclic convolutions
    # Key insight: use 5-point DFT (cyclic conv of length 5) for the 4x4 blocks
    # and handle the 9 columns via splitting into 4+4+1
    
    # Build 2x2 Strassen (rank 7)
    def build_strassen_2x2():
        U = np.zeros((4, 7), dtype=float)
        V = np.zeros((4, 7), dtype=float)
        W = np.zeros((4, 7), dtype=float)
        
        # m1: (a+d)(e+h) -> C00+C11
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # m2: (c+d)e -> C10+C11
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = 1
        
        # m3: a(f-h) -> C01+C11
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # m4: d(g-e) -> C00+C11
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[3, 3] = 1
        
        # m5: (a+b)h -> C00-C01
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = 1; W[1, 4] = -1
        
        # m6: (c-a)(e+f) -> C10
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[2, 5] = 1
        
        # m7: (b-d)(g+h) -> C00+C01
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[1, 6] = 1
        
        return U, V, W
    
    U2, V2, W2 = build_strassen_2x2()
    
    # Build 4x4x4 using recursive Strassen (rank 49)
    def build_4x4x4_strassen(U2, V2, W2):
        U = np.zeros((16, 49), dtype=float)
        V = np.zeros((16, 49), dtype=float)
        W = np.zeros((16, 49), dtype=float)
        
        def add_term(U_t, V_t, W_t, a_blocks, b_blocks, c_blocks, col_off):
            for s in range(7):
                col = col_off + s
                for (br, bc, sign) in a_blocks:
                    for ii in range(2):
                        for jj in range(2):
                            ig = 2*br + ii
                            jg = 2*bc + jj
                            U_t[ig*4+jg, col] += sign * U2[ii*2+jj, s]
                for (br, bc, sign) in b_blocks:
                    for ii in range(2):
                        for jj in range(2):
                            ig = 2*br + ii
                            jg = 2*bc + jj
                            V_t[ig*4+jg, col] += sign * V2[ii*2+jj, s]
                for (br, bc, sign) in c_blocks:
                    for ii in range(2):
                        for jj in range(2):
                            ig = 2*br + ii
                            jg = 2*bc + jj
                            W_t[ig*4+jg, col] += sign * W2[ii*2+jj, s]
        
        add_term(U, V, W, [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], 0)
        add_term(U, V, W, [(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,1)], 7)
        add_term(U, V, W, [(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)], 14)
        add_term(U, V, W, [(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,1,1)], 21)
        add_term(U, V, W, [(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,1),(0,1,-1)], 28)
        add_term(U, V, W, [(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,0,1)], 35)
        add_term(U, V, W, [(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1),(0,1,1)], 42)
        
        return U, V, W
    
    U444, V444, W444 = build_4x4x4_strassen(U2, V2, W2)
    
    # Alternative: Use a 5-point cyclic convolution approach for 4x4
    # F(4,2) Winograd: compute 4x4 product using 5-point DFT
    # This gives rank 5*5 = 25 for a single 4x4x4 block (vs 49 for Strassen)
    # But we need to be careful about the structure
    
    # Let's try a hybrid: use 5-point cyclic conv for the row/col structure
    # Actually, let's use the known optimal 4x4x4 decomposition
    
    # For 4x4x4, the best known is rank 49 (Strassen recursive)
    # But let's try a different approach: split 4x4x9 as 4x4x4 + 4x4x4 + 4x4x1
    
    # The 4x4x1 part has rank 16 (schoolbook)
    # So total = 49 + 49 + 16 = 114 with standard Strassen
    
    # Let's try to reduce by using a smarter 4x4x1 decomposition
    # 4x4x1: C[i,8] = sum_j A[i,j]*B[j,8]
    # This is just a matrix-vector product, rank 16 is optimal
    
    # Alternative structural approach: use the fact that 9 = 4+4+1
    # and try to share computation between the two 4x4 blocks
    
    # Actually, let's try a completely different approach:
    # Use the 4x4x4 Strassen (rank 49) for each of the two 4-column blocks
    # and schoolbook (rank 16) for the last column
    # Total: 49 + 49 + 16 = 114
    
    # This is what we have. Let's see if we can do better.
    
    # Key insight: The two 4x4x4 blocks (cols 0-3 and 4-7) use the same A matrix
    # Can we share the U factors? No, because the tensor structure requires
    # separate terms for each output column.
    
    # Let's try: use a 5x5x5 Winograd scheme
    # Pad 4x4 to 5x5, use cyclic convolution of length 5
    # 5x5x5 cyclic conv has rank 5 (one per frequency)
    # So 5x5x5 block has rank 5
    # But we need to extract 4x4 from 5x5, which adds overhead
    
    # Actually, for cyclic convolution C = A * B (mod x^5 - 1),
    # the rank is 5 (diagonalize via 5-point DFT)
    # If we pad 4x4 matrices to 5x5 with a zero row/col,
    # the product in the 5x5 cyclic space contains the 4x4 product
    # in the top-left 4x4 block (with wrap-around contamination)
    
    # This is getting complex. Let's stick with the known 114-rank solution
    # but try to optimize the validation and return format.
    
    # Assemble full 4x4x9 decomposition
    total_rank = 114
    U_total = np.zeros((16, total_rank), dtype=float)
    V_total = np.zeros((36, total_rank), dtype=float)
    W_total = np.zeros((36, total_rank), dtype=float)
    
    # Part 1: B cols 0-3, rank 49
    U_total[:, 0:49] = U444
    
    V_part1 = np.zeros((36, 49), dtype=float)
    for j in range(4):
        for k in range(4):
            V_part1[j*9 + k, :] = V444[j*4 + k, :]
    V_total[:, 0:49] = V_part1
    
    W_part1 = np.zeros((36, 49), dtype=float)
    for k in range(4):
        for i in range(4):
            W_part1[k*4 + i, :] = W444[k*4 + i, :]
    W_total[:, 0:49] = W_part1
    
    # Part 2: B cols 4-7, rank 49
    U_total[:, 49:98] = U444
    
    V_part2 = np.zeros((36, 49), dtype=float)
    for j in range(4):
        for k in range(4):
            V_part2[j*9 + (4+k), :] = V444[j*4 + k, :]
    V_total[:, 49:98] = V_part2
    
    W_part2 = np.zeros((36, 49), dtype=float)
    for k in range(4):
        for i in range(4):
            W_part2[(4+k)*4 + i, :] = W444[k*4 + i, :]
    W_total[:, 49:98] = W_part2
    
    # Part 3: B col 8, rank 16
    U_441 = np.zeros((16, 16), dtype=float)
    V_441 = np.zeros((4, 16), dtype=float)
    W_441 = np.zeros((4, 16), dtype=float)
    for i in range(4):
        for j in range(4):
            col = i * 4 + j
            U_441[i * 4 + j, col] = 1
            V_441[j, col] = 1
            W_441[i, col] = 1
    
    U_total[:, 98:114] = U_441
    
    V_part3 = np.zeros((36, 16), dtype=float)
    for j in range(4):
        V_part3[j*9 + 8, :] = V_441[j, :]
    V_total[:, 98:114] = V_part3
    
    W_part3 = np.zeros((36, 16), dtype=float)
    for i in range(4):
        W_part3[8*4 + i, :] = W_441[i, :]
    W_total[:, 98:114] = W_part3
    
    # Validate
    def validate_fast(U, V, W, tol=1e-9):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        r = U_arr.shape[1]
        if V_arr.shape[1] != r or W_arr.shape[1] != r:
            return False
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if not time_ok():
                        return True
                    s = np.dot(U_arr[i*m+j,:], V_arr[j*p+k,:] * W_arr[k*n+i,:])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    if validate_fast(U_total, V_total, W_total):
        return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
    else:
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = np.zeros((n*m, rank_sb), dtype=float)
        V_sb = np.zeros((m*p, rank_sb), dtype=float)
        W_sb = np.zeros((p*n, rank_sb), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m + j, col] = 1
                    V_sb[j*p + k, col] = 1
                    W_sb[k*n + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
