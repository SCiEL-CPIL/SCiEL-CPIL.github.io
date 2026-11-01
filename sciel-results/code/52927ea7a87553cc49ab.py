import time
import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    start_time = time.time()
    deadline = 140  # 150s limit, keep 10s safety margin
    
    n, m, p = DIMENSIONS
    # n=2, m=4, p=4
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (16, rank)
    # W shape: (p*n, rank) = (8, rank)
    
    # Target Tensor T
    # T[u, v, w] = 1 if u corresponds to A[i, j], v corresponds to B[j, k], w corresponds to C[i, k]
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    
    size_u = n * m
    size_v = m * p
    size_w = p * n
    T = np.zeros((size_u, size_v, size_w))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    def verify(U, V, W):
        # U: (8, R), V: (16, R), W: (8, R)
        T_calc = np.einsum('ur, vr, wr -> uvw', U, V, W)
        return np.allclose(T_calc, T, atol=1e-6)

    # Fallback: Rank 32 Schoolbook
    U32 = np.zeros((8, 32))
    V32 = np.zeros((16, 32))
    W32 = np.zeros((8, 32))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U32[u_idx, col] = 1
                V32[v_idx, col] = 1
                W32[w_idx, col] = 1
                col += 1
                
    best_U, best_V, best_W = U32, V32, W32
    best_rank = 32

    # Strategy: Construct Rank 20 using two Rank 10 decompositions for 2x2x4.
    # The 2x4x4 multiplication can be split into two 2x2x4 multiplications:
    # 1. A_top (2x2) * B_top (2x4) -> C_part1 (2x4)
    # 2. A_bot (2x2) * B_bot (2x4) -> C_part2 (2x4)
    # C = C_part1 + C_part2
    
    # We need a Rank 10 decomposition for 2x2x4.
    # Let A be 2x2, B be 2x4, C be 2x4.
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01, b02, b03], [b10, b11, b12, b13]]
    # C = A B
    
    # We can view B as two 2x2 matrices B1 and B2.
    # C = A B1 + A B2.
    # This is not quite right for rank addition because A is shared.
    # However, R(2,2,4) = 10 is a known result.
    # Let's derive or find the Rank 10 coefficients for 2x2x4.
    
    # Alternative: Use the known Rank 7 Strassen for 2x2x2.
    # R(2,2,4) <= R(2,2,2) + R(2,2,2) = 14.
    # Can we do better?
    # Yes, R(2,2,4) = 10.
    
    # Let's try to construct Rank 10 for 2x2x4 manually or via a known formula.
    # Actually, there is a simpler structure.
    # If we can't find Rank 10, Rank 28 (7+7+7+7 for four 2x2x2 blocks) is 28.
    # But we can do better than 28.
    # R(2,4,4) <= 2 * R(2,2,4).
    
    # Let's use a known Rank 10 decomposition for 2x2x4.
    # Reference: "Tensor rank of 2x2x4 matrix multiplication is 10"
    # The 10 bilinear forms are:
    # 1. (a00+a01+a10) * (b00+b10) -> c00, c10 terms? No, need to be careful.
    
    # Let's use a different approach: 
    # Use Rank 7 Strassen for the first 2x2 block of B, and Rank 7 for the second?
    # That gives 14 for 2x2x4.
    # Total Rank 28 for 2x4x4.
    
    # Is Rank 20 possible?
    # If R(2,2,4)=10, then R(2,4,4) <= 20.
    
    # Let's implement the Rank 10 decomposition for 2x2x4.
    # A = [[a, b], [c, d]]
    # B = [[e, f, g, h], [i, j, k, l]]
    # C = A B = [[ae+bi, af+bj, ag+bk, ah+bl], [ce+di, cf+dj, cg+dk, ch+dl]]
    
    # We need 10 terms.
    # Let's try to find them.
    # This is equivalent to multiplying a 2x2 matrix by two 2x2 matrices B1=[e,f; i,j] and B2=[g,h; k,l].
    # C1 = A B1, C2 = A B2.
    # Rank(C1) = 7, Rank(C2) = 7. Total 14.
    # Can we share some terms?
    # Strassen for 2x2x2:
    # M1 = (a+d)(e+i)
    # M2 = (c+d)(e)
    # M3 = (a-b)(f+j)
    # M4 = (a+d)(j-i)
    # M5 = (a+c)(e-i)
    # M6 = (b+d)(f-j)
    # M7 = (a+c)(j-k) -- Wait, standard Strassen is for 2x2*2x2.
    
    # Standard Strassen for 2x2:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+g)
    # M2 = (c+d)(e)
    # M3 = (a-b)(f+h)
    # M4 = (a+d)(h-g)
    # M5 = (a+c)(e-g)
    # M6 = (b+d)(f-h)
    # M7 = (a+c)(h-f)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M6
    # C21 = M2+M5
    # C22 = M1-M3+M2-M6
    
    # For 2x2x4, B has 8 elements.
    # Let B = [[e,f,g,h],[i,j,k,l]]
    # We can split into B1 = [[e,f],[i,j]] and B2 = [[g,h],[k,l]]
    # If we use Strassen for B1 and Strassen for B2, we get 14 terms.
    # M1_1, ..., M7_1 for B1
    # M1_2, ..., M7_2 for B2
    # Total 14.
    
    # Can we reduce 14 to 10?
    # Yes, R(2,2,4)=10.
    # The reduction comes from the fact that the "A" matrix is the same.
    # However, deriving the specific 10 coefficients is complex and prone to error if done ad-hoc.
    
    # Given the time constraint and the risk of incorrect manual derivation, 
    # and the fact that Rank 28 is a safe, easily verifiable improvement over 32,
    # I will implement Rank 28 using two independent Strassen decompositions for the top and bottom halves of B.
    # Wait, the split was A_top * B_top + A_bot * B_bot?
    # No, A is 2x4. A_top is A[:, 0:2] (2x2). A_bot is A[:, 2:4] (2x2).
    # B is 4x4. B_top is B[0:2, :] (2x4). B_bot is B[2:4, :] (2x4).
    # C = A_top B_top + A_bot B_bot.
    # A_top and A_bot are disjoint sets of variables in A.
    # B_top and B_bot are disjoint sets of variables in B.
    # So the tensors are independent.
    # Rank(2,4,4) <= Rank(2,2,4) + Rank(2,2,4).
    # If I use Rank 7 Strassen for each 2x2x2 sub-block of B_top and B_bot:
    # B_top = [B11, B12] (2x2, 2x2).
    # A_top B_top = A_top B11 + A_top B12.
    # This requires 2 * Rank(2,2,2) = 14 for the top half.
    # Similarly 14 for the bottom half.
    # Total 28.
    
    # So Rank 28 is achievable by simply applying Strassen to each 2x2x2 sub-multiplication.
    # There are 4 such sub-multiplications:
    # 1. A_top * B[0:2, 0:2]
    # 2. A_top * B[0:2, 2:4]
    # 3. A_bot * B[2:4, 0:2]
    # 4. A_bot * B[2:4, 2:4]
    
    # Let's construct Rank 28.
    
    U28 = np.zeros((8, 28))
    V28 = np.zeros((16, 28))
    W28 = np.zeros((8, 28))
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[p,q],[r,s]]
    # M1 = (a+d)(e+g)
    # M2 = (c+d)(e)
    # M3 = (a-b)(f+h)
    # M4 = (a+d)(h-g)
    # M5 = (a+c)(e-g)
    # M6 = (b+d)(f-h)
    # M7 = (a+c)(h-f)
    # p = M1+M4-M5+M7
    # q = M3+M6
    # r = M2+M5
    # s = M1-M3+M2-M6
    
    # We define a helper to add a Strassen block.
    def add_strassen_block(A_rows, B_cols_start, B_cols_end, C_cols_start, C_cols_end):
        # A_rows: tuple of 2 row indices in A (0..1)
        # B_cols_start, B_cols_end: row indices in B (0..3)
        # C_cols_start, C_cols_end: col indices in C (0..3)
        # Note: B is 4x4. We are multiplying A (2x2) by B (2x2 subblock).
        # The result is added to C (2x2 subblock).
        
        nonlocal col
        
        # Indices for A submatrix
        a00 = A_rows[0] * 4 + B_cols_start # Wait, A is 2x4.
        # A is 2x4. Rows 0,1. Cols 0..3.
        # If we split A into A_top (cols 0,1) and A_bot (cols 2,3).
        # Then A_top is 2x2.
        # A_top[0,0] is A[0,0] -> u_idx = 0*4+0 = 0
        # A_top[0,1] is A[0,1] -> u_idx = 0*4+1 = 1
        # A_top[1,0] is A[1,0] -> u_idx = 1*4+0 = 4
        # A_top[1,1] is A[1,1] -> u_idx = 1*4+1 = 5
        
        # Let's generalize.
        # A submatrix rows: 0, 1.
        # A submatrix cols: c0, c1.
        # B submatrix rows: r0, r1.
        # B submatrix cols: k0, k1.
        # C submatrix rows: 0, 1.
        # C submatrix cols: k0, k1.
        
        # A indices:
        # a = A[0, c0] -> u = 0*4 + c0
        # b = A[0, c1] -> u = 0*4 + c1
        # c = A[1, c0] -> u = 1*4 + c0
        # d = A[1, c1] -> u = 1*4 + c1
        
        # B indices:
        # e = B[r0, k0] -> v = r0*4 + k0
        # f = B[r0, k1] -> v = r0*4 + k1
        # g = B[r1, k0] -> v = r1*4 + k0
        # h = B[r1, k1] -> v = r1*4 + k1
        
        # C indices:
        # p = C[0, k0] -> w = k0*2 + 0
        # q = C[0, k1] -> w = k1*2 + 0
        # r = C[1, k0] -> w = k0*2 + 1
        # s = C[1, k1] -> w = k1*2 + 1
        
        u_a = 0*4 + c0
        u_b = 0*4 + c1
        u_c = 1*4 + c0
        u_d = 1*4 + c1
        
        v_e = r0*4 + k0
        v_f = r0*4 + k1
        v_g = r1*4 + k0
        v_h = r1*4 + k1
        
        w_p = k0*2 + 0
        w_q = k1*2 + 0
        w_r = k0*2 + 1
        w_s = k1*2 + 1
        
        # M1 = (a+d)(e+g)
        U28[u_a, col] += 1
        U28[u_d, col] += 1
        V28[v_e, col] += 1
        V28[v_g, col] += 1
        W28[w_p, col] += 1
        W28[w_q, col] -= 1 # Wait, C12 = M3+M6. M1 contributes to C11 and C22.
        # p = M1+M4-M5+M7
        # q = M3+M6
        # r = M2+M5
        # s = M1-M3+M2-M6
        
        # M1 contributes to p and s.
        W28[w_p, col] += 1
        W28[w_s, col] += 1
        # Reset W for M1, I did it twice.
        W28[w_p, col] = 1
        W28[w_s, col] = 1
        col += 1
        
        # M2 = (c+d)(e)
        U28[u_c, col] += 1
        U28[u_d, col] += 1
        V28[v_e, col] += 1
        W28[w_r, col] += 1
        W28[w_s, col] -= 1
        col += 1
        
        # M3 = (a-b)(f+h)
        U28[u_a, col] += 1
        U28[u_b, col] -= 1
        V28[v_f, col] += 1
        V28[v_h, col] += 1
        W28[w_q, col] += 1
        W28[w_s, col] -= 1
        col += 1
        
        # M4 = (a+d)(h-g)
        U28[u_a, col] += 1
        U28[u_d, col] += 1
        V28[v_h, col] += 1
        V28[v_g, col] -= 1
        W28[w_p, col] += 1
        col += 1
        
        # M5 = (a+c)(e-g)
        U28[u_a, col] += 1
        U28[u_c, col] += 1
        V28[v_e, col] += 1
        V28[v_g, col] -= 1
        W28[w_p, col] -= 1
        W28[w_r, col] += 1
        col += 1
        
        # M6 = (b+d)(f-h)
        U28[u_b, col] += 1
        U28[u_d, col] += 1
        V28[v_f, col] += 1
        V28[v_h, col] -= 1
        W28[w_q, col] += 1
        W28[w_s, col] -= 1
        col += 1
        
        # M7 = (a+c)(h-f)
        U28[u_a, col] += 1
        U28[u_c, col] += 1
        V28[v_h, col] += 1
        V28[v_f, col] -= 1
        W28[w_p, col] += 1
        col += 1
        
    # We need to call this 4 times.
    # 1. A_top (cols 0,1) * B_top-left (rows 0,1, cols 0,1) -> C (cols 0,1)
    # A_top cols: 0, 1.
    # B rows: 0, 1. Cols: 0, 1.
    # C cols: 0, 1.
    add_strassen_block((0,1), 0, 1, 0, 1) # Args: A_rows (unused in current impl, hardcoded 0,1), B_rows 0,1, B_cols 0,1, C_cols 0,1
    
    # Wait, my helper signature was:
    # add_strassen_block(A_rows, B_cols_start, B_cols_end, C_cols_start, C_cols_end)
    # But inside I used r0, r1, k0, k1, c0, c1.
    # I need to pass the correct indices.
    
    # Let's rewrite the helper to take explicit indices.
    
    # Reset
    U28 = np.zeros((8, 28))
    V28 = np.zeros((16, 28))
    W28 = np.zeros((8, 28))
    col = 0
    
    def add_strassen(c0, c1, r0, r1, k0, k1):
        nonlocal col
        
        u_a = 0*4 + c0
        u_b = 0*4 + c1
        u_c = 1*4 + c0
        u_d = 1*4 + c1
        
        v_e = r0*4 + k0
        v_f = r0*4 + k1
        v_g = r1*4 + k0
        v_h = r1*4 + k1
        
        w_p = k0*2 + 0
        w_q = k1*2 + 0
        w_r = k0*2 + 1
        w_s = k1*2 + 1
        
        # M1 = (a+d)(e+g) -> p, s
        U28[u_a, col] = 1; U28[u_d, col] = 1
        V28[v_e, col] = 1; V28[v_g, col] = 1
        W28[w_p, col] = 1; W28[w_s, col] = 1
        col += 1
        
        # M2 = (c+d)(e) -> r, -s
        U28[u_c, col] = 1; U28[u_d, col] = 1
        V28[v_e, col] = 1
        W28[w_r, col] = 1; W28[w_s, col] = -1
        col += 1
        
        # M3 = (a-b)(f+h) -> q, -s
        U28[u_a, col] = 1; U28[u_b, col] = -1
        V28[v_f, col] = 1; V28[v_h, col] = 1
        W28[w_q, col] = 1; W28[w_s, col] = -1
        col += 1
        
        # M4 = (a+d)(h-g) -> p
        U28[u_a, col] = 1; U28[u_d, col] = 1
        V28[v_h, col] = 1; V28[v_g, col] = -1
        W28[w_p, col] = 1
        col += 1
        
        # M5 = (a+c)(e-g) -> -p, r
        U28[u_a, col] = 1; U28[u_c, col] = 1
        V28[v_e, col] = 1; V28[v_g, col] = -1
        W28[w_p, col] = -1; W28[w_r, col] = 1
        col += 1
        
        # M6 = (b+d)(f-h) -> q, -s
        U28[u_b, col] = 1; U28[u_d, col] = 1
        V28[v_f, col] = 1; V28[v_h, col] = -1
        W28[w_q, col] = 1; W28[w_s, col] = -1
        col += 1
        
        # M7 = (a+c)(h-f) -> p
        U28[u_a, col] = 1; U28[u_c, col] = 1
        V28[v_h, col] = 1; V28[v_f, col] = -1
        W28[w_p, col] = 1
        col += 1
        
    # Block 1: A cols 0,1. B rows 0,1. B cols 0,1. C cols 0,1.
    add_strassen(0, 1, 0, 1, 0, 1)
    
    # Block 2: A cols 0,1. B rows 0,1. B cols 2,3. C cols 2,3.
    add_strassen(0, 1, 0, 1, 2, 3)
    
    # Block 3: A cols 2,3. B rows 2,3. B cols 0,1. C cols 0,1.
    add_strassen(2, 3, 2, 3, 0, 1)
    
    # Block 4: A cols 2,3. B rows 2,3. B cols 2,3. C cols 2,3.
    add_strassen(2, 3, 2, 3, 2, 3)
    
    # Verify Rank 28
    if verify(U28, V28, W28):
        best_U, best_V, best_W = U28, V28, W28
        best_rank = 28
    else:
        # If verification fails, stick to 32
        pass

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
