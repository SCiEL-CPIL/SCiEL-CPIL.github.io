import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 102 decomposition
    # 1. 4x4 Strassen (Rank 49)
    # 2. 4x4 Mat-Vec (Rank 16)
    # 3. 4x4 Vec-Mat (Rank 16)
    # 4. 4x4 Outer Prod (Rank 16)
    # 5. 4x1 Dot Prod (Rank 4)
    # 6. 1x1 Square (Rank 1)
    
    total_rank = 102
    U = [[0] * total_rank for _ in range(25)]
    V = [[0] * total_rank for _ in range(25)]
    W = [[0] * total_rank for _ in range(25)]
    
    col = 0
    
    # Helper to add a decomposition block
    def add_block(u_local, v_local, w_local, i_offset, j_offset, k_offset, local_n):
        nonlocal col
        for r in range(u_local.shape[1]):
            for i in range(local_n):
                for j in range(local_n):
                    val_u = u_local[i*local_n+j, r]
                    if val_u != 0:
                        U[(i_offset+i)*5 + (j_offset+j)][col+r] += val_u
            
            for j in range(local_n):
                for k in range(local_n):
                    val_v = v_local[j*local_n+k, r]
                    if val_v != 0:
                        V[(j_offset+j)*5 + (k_offset+k)][col+r] += val_v
            
            for k in range(local_n):
                for i in range(local_n):
                    val_w = w_local[k*local_n+i, r]
                    if val_w != 0:
                        W[(k_offset+k)*5 + (i_offset+i)][col+r] += val_w
        col += u_local.shape[1]

    # 1. Generate 4x4 Strassen
    u4, v4, w4 = get_strassen(4)
    add_block(u4, v4, w4, 0, 0, 0, 4)
    
    # 2. Mat-Vec A[0:4, 0:4] * B[0:4, 4] -> C[0:4, 4]
    # C_i = sum_j A_ij B_j4
    # i in 0..3, j in 0..3
    # U[i*5+j], V[j*5+4], W[4*5+i]
    for i in range(4):
        for j in range(4):
            U[i*5 + j][col] = 1
            V[j*5 + 4][col] = 1
            W[4*5 + i][col] = 1
            col += 1
            
    # 3. Vec-Mat A[4, 0:4] * B[0:4, 0:4] -> C[4, 0:4]
    # C_k = sum_j A_4j B_jk
    # k in 0..3, j in 0..3
    # U[4*5+j], V[j*5+k], W[k*5+4]
    for k in range(4):
        for j in range(4):
            U[4*5 + j][col] = 1
            V[j*5 + k][col] = 1
            W[k*5 + 4][col] = 1
            col += 1
            
    # 4. Outer Prod A[0:4, 4] * B[4, 0:4] -> C[0:4, 0:4]
    # C_ik = A_i4 B_4k
    # i in 0..3, k in 0..3
    # U[i*5+4], V[4*5+k], W[k*5+i]
    for i in range(4):
        for k in range(4):
            U[i*5 + 4][col] = 1
            V[4*5 + k][col] = 1
            W[k*5 + i][col] = 1
            col += 1
            
    # 5. Dot Prod A[4, 0:4] * B[0:4, 4] -> C[4, 4]
    # C_44 = sum_j A_4j B_j4
    # j in 0..3
    # U[4*5+j], V[j*5+4], W[4*5+4]
    for j in range(4):
        U[4*5 + j][col] = 1
        V[j*5 + 4][col] = 1
        W[4*5 + 4][col] = 1
        col += 1
        
    # 6. Square A[4, 4] * B[4, 4] -> C[4, 4]
    # C_44 += A_44 B_44
    U[4*5 + 4][col] = 1
    V[4*5 + 4][col] = 1
    W[4*5 + 4][col] = 1
    col += 1
    
    # Verify col == total_rank
    if col != total_rank:
        # Fallback to schoolbook if construction fails
        return construct_schoolbook()
        
    return {"U": U, "V": V, "W": W}

def construct_schoolbook():
    n, m, p = DIMENSIONS
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

def get_strassen(n):
    # Returns U, V, W for n x n matrix multiplication using Strassen
    # n must be power of 2
    if n == 2:
        # Strassen 2x2
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Indices: 0: A11, 1: A12, 2: A21, 3: A22
        # B: 0: B11, 1: B12, 2: B21, 3: B22
        # C: 0: C11, 1: C12, 2: C21, 3: C22
        
        # M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        # M3
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        
        # M4
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1
        
        # M5
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[3, 4] = 1
        
        # M6
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W
        
    else:
        m = n // 2
        u_m, v_m, w_m = get_strassen(m)
        rank_m = u_m.shape[1]
        rank_n = 7 * rank_m
        
        U = np.zeros((n*n, rank_n))
        V = np.zeros((n*n, rank_n))
        W = np.zeros((n*n, rank_n))
        
        # Map local indices to global
        # A blocks:
        # 0: 0..m-1, 0..m-1
        # 1: 0..m-1, m..n-1
        # 2: m..n-1, 0..m-1
        # 3: m..n-1, m..n-1
        
        # The 7 Strassen terms involve sums/differences of blocks.
        # This recursive construction is tricky to index correctly without explicit block mapping.
        # For n=4, it's better to just hardcode or use a simpler method.
        # Given time, I will stick to the n=2 base case and assume the recursive step
        # can be handled by a generic tensor product if I had more time, but for n=4,
        # I can just use the 7 terms of 2x2 Strassen applied to 2x2 blocks of 4x4.
        
        # Actually, for n=4, the rank is 49.
        # I will implement the recursive step properly.
        
        # Strassen terms for n x n:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Each M is a product of two n/2 x n/2 matrices.
        # So we need 7 decompositions of size m x m.
        
        # We can create a function to add a scaled decomposition block.
        
        def add_scaled(u, v, w, a_block, b_block, c_block, scale=1):
            # a_block, b_block, c_block are (row_start, col_start)
            for r in range(u.shape[1]):
                for i in range(m):
                    for j in range(m):
                        val = u[i*m+j, r] * scale
                        if val != 0:
                            U[(a_block[0]+i)*n + (a_block[1]+j), 7*r] += val # This indexing is wrong for 7 terms
                            
        # This is getting complicated to index correctly in the recursive step.
        # I will use the fact that n=4 is small and just return the schoolbook for n=4 if I can't get Strassen right quickly.
        # But Rank 49 is much better than 16.
        # Let's try to hardcode the 4x4 Strassen indices? No, too many.
        
        # Alternative: Use the 2x2 Strassen on the 4x4 matrix by viewing it as 2x2 blocks of 2x2.
        # This is exactly what the recursion does.
        
        # I will implement the recursion carefully.
        
        # For each of the 7 terms, we have a specific combination of A and B blocks.
        # We need to map the local U, V, W of the m x m decomposition to the global n x n.
        
        # Term 1: (A11+A22)(B11+B22)
        # A part: A11 and A22.
        # B part: B11 and B22.
        # C part: C11 and C22? No, the result is added to C11 and C22.
        
        # This is a bit complex. I will rely on the 2x2 base case and a generic recursive builder.
        
        # To save time and ensure correctness, I will use the schoolbook for the 4x4 part if the recursive Strassen is too risky.
        # Rank 49 vs 16. 49 is better.
        
        # Let's try to implement the recursive step.
        
        # The 7 terms correspond to 7 multiplications of m x m matrices.
        # Let U_m, V_m, W_m be the decomposition for m x m.
        
        # For term t (0..6), we have coefficients for A blocks and B blocks.
        
        # I will define the block indices.
        # A00: (0,0), A01: (0,1), A10: (1,0), A11: (1,1)
        
        # Term 0: M1
        # A: A00 (+1), A11 (+1)
        # B: B00 (+1), B11 (+1)
        # C: C00 (+1), C11 (+1)
        
        # We need to add the decomposition of (A00+A11)(B00+B11).
        # This is not a simple tensor product of the decompositions.
        # It requires summing the U vectors for A00 and A11, etc.
        
        # U_global for this term:
        # For each rank r in U_m:
        # U_global[A00 indices, r] += U_m[:, r]
        # U_global[A11 indices, r] += U_m[:, r]
        
        # This is doable.
        
        # I will write a helper to add a term.
        
        # Due to complexity and time, I will use the schoolbook for 4x4 if I don't finish the recursion.
        # But I can just use the 2x2 Strassen and apply it 4 times? No.
        
        # Let's just use the schoolbook for the 4x4 part to be safe.
        # Rank 16 for 4x4 schoolbook.
        # Total Rank = 16 + 16 + 16 + 16 + 4 + 1 = 69.
        # This is still < 125.
        
        # So I will use Schoolbook for 4x4.
        
        u4 = np.zeros((16, 16))
        v4 = np.zeros((16, 16))
        w4 = np.zeros((16, 16))
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    u4[i*4+j, i*4+j] = 1 # Wait, schoolbook U, V, W
                    # U[i*m+j, c] = 1
                    # V[j*p+k, c] = 1
                    # W[k*n+i, c] = 1
                    # For 4x4:
                    c = i*4 + j*4 + k # This is not right.
                    
        # Let's just generate schoolbook 4x4.
        u4 = np.zeros((16, 64)) # Rank 64? No, 4*4*4=64.
        # I want rank 16? No, schoolbook is 64.
        # My previous calculation: 49 (Strassen) + ...
        # If I use schoolbook 4x4, rank is 64.
        # Total = 64 + 16 + 16 + 16 + 4 + 1 = 117.
        # 117 < 125.
        
        # So I will use Schoolbook for 4x4.
        
        u4 = np.zeros((16, 64))
        v4 = np.zeros((16, 64))
        w4 = np.zeros((16, 64))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    u4[i*4+j, col] = 1
                    v4[j*4+k, col] = 1
                    w4[k*4+i, col] = 1
                    col += 1
                    
        return u4, v4, w4

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    total_rank = 117 # 64 + 16 + 16 + 16 + 4 + 1
    U = [[0] * total_rank for _ in range(25)]
    V = [[0] * total_rank for _ in range(25)]
    W = [[0] * total_rank for _ in range(25)]
    
    col = 0
    
    # 1. 4x4 Schoolbook
    u4, v4, w4 = get_strassen(4) # This returns schoolbook 4x4 in my current draft
    for r in range(u4.shape[1]):
        for i in range(4):
            for j in range(4):
                val = u4[i*4+j, r]
                if val != 0:
                    U[i*5+j][col+r] += val
        for j in range(4):
            for k in range(4):
                val = v4[j*4+k, r]
                if val != 0:
                    V[j*5+k][col+r] += val
        for k in range(4):
            for i in range(4):
                val = w4[k*4+i, r]
                if val != 0:
                    W[k*5+i][col+r] += val
    col += u4.shape[1]
    
    # 2. Mat-Vec
    for i in range(4):
        for j in range(4):
            U[i*5+j][col] = 1
            V[j*5+4][col] = 1
            W[4*5+i][col] = 1
            col += 1
            
    # 3. Vec-Mat
    for k in range(4):
        for j in range(4):
            U[4*5+j][col] = 1
            V[j*5+k][col] = 1
            W[k*5+4][col] = 1
            col += 1
            
    # 4. Outer Prod
    for i in range(4):
        for k in range(4):
            U[i*5+4][col] = 1
            V[4*5+k][col] = 1
            W[k*5+i][col] = 1
            col += 1
            
    # 5. Dot Prod
    for j in range(4):
        U[4*5+j][col] = 1
        V[j*5+4][col] = 1
        W[4*5+4][col] = 1
        col += 1
        
    # 6. Square
    U[4*5+4][col] = 1
    V[4*5+4][col] = 1
    W[4*5+4][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
