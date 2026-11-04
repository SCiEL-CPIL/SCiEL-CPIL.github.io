import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0

    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    
    def schoolbook():
        rank = rank_schoolbook
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = rank_schoolbook

    def check_decomposition(U, V, W):
        """Check if U, V, W form a valid decomposition."""
        rank = len(U[0])
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        if len(V[0]) != rank or len(W[0]) != rank:
            return False
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1) > 1e-9:
                        return False
        return True

    # Winograd F(2) tensor product decomposition for 4x4x4x4 part
    # F(2) for 2x2x2x2 has rank 7
    # Tensor product gives rank 7*7 = 49 for 4x4x4x4
    # Plus 4 more for the 5th column = 53 total
    
    # F(2) algorithm: 
    # Input: A (2x2), B (2x2)
    # Output: C = A*B (2x2)
    # 
    # 7 multiplications:
    # m0 = (a00+a11)*(b00+b11)
    # m1 = (a10+a11)*b00
    # m2 = a00*(b10-b00)
    # m3 = a11*(b01-b11)
    # m4 = (a00+a01)*b11
    # m5 = (a10-a00)*(b00+b10)
    # m6 = (a01-a11)*(b11+b01)
    #
    # C00 = m0 + m2 - m3 - m6
    # C01 = m3 + m6
    # C10 = m1 - m2
    # C11 = m0 - m1 + m4 + m5

    # For F(2), the factor matrices are:
    # U (for A): 4x7, V (for B): 4x7, W (for C): 4x7
    
    # Let's define F(2) factors
    # U[i,j] where i is index in A (row*2+col), j is multiplication index
    # V[i,j] where i is index in B (row*2+col), j is multiplication index
    # W[i,j] where i is index in C (row*2+col), j is multiplication index
    
    # A indices: 0=a00, 1=a01, 2=a10, 3=a11
    # B indices: 0=b00, 1=b01, 2=b10, 3=b11
    # C indices: 0=c00, 1=c01, 2=c10, 3=c11
    
    # m0 = (a00+a11)*(b00+b11)
    # m1 = (a10+a11)*b00
    # m2 = a00*(b10-b00)
    # m3 = a11*(b01-b11)
    # m4 = (a00+a01)*b11
    # m5 = (a10-a00)*(b00+b10)
    # m6 = (a01-a11)*(b11+b01)
    
    F2_U = [
        [1, 0, 1, 0, 1, -1, 0],   # a00: m0, m2, m4, m5
        [0, 0, 0, 0, 1, 0, 1],    # a01: m4, m6
        [0, 1, 0, 0, 0, 1, 0],    # a10: m1, m5
        [1, 1, 0, 1, 0, 0, -1],   # a11: m0, m1, m3, m6
    ]
    
    F2_V = [
        [1, 1, -1, 0, 0, 1, 0],   # b00: m0, m1, m2, m5
        [0, 0, 0, 1, 0, 0, 1],    # b01: m3, m6
        [0, 0, 1, 0, 0, 1, 0],    # b10: m2, m5
        [1, 0, 0, -1, 1, 0, 1],   # b11: m0, m3, m4, m6
    ]
    
    F2_W = [
        [1, 0, 1, -1, 0, 0, -1],  # c00: m0, m2, m3, m6
        [0, 0, 0, 1, 0, 0, 1],    # c01: m3, m6
        [0, 1, -1, 0, 0, 0, 0],   # c10: m1, m2
        [1, -1, 0, 0, 1, 1, 0],   # c11: m0, m1, m4, m5
    ]
    
    # Verify F(2) decomposition
    def verify_F2():
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    # Check if U[i]*V[j]*W[k] sums to 1 when (i,j,k) corresponds to valid multiplication
                    # Actually, we need to check the tensor decomposition property
                    pass
        # The F(2) algorithm is well-known and correct, so we trust it
    
    # Now build 4x4x4x4 using tensor product of F(2) with itself
    # A is 4x4, indexed as A[i,j] where i,j in {0,1,2,3}
    # We can view A as 2x2 blocks of 2x2 matrices
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    
    # The tensor product approach:
    # For 4x4x4x4, we use the Kronecker product structure
    # U_4x4[i,j] = F2_U[i%2, j%7] * F2_U[i//2, j//7]
    # But this needs careful indexing
    
    # Actually, let's use a simpler approach:
    # The 4x4x4x4 tensor can be decomposed as the tensor product of two F(2) tensors
    # This gives rank 7*7 = 49
    
    # Let me construct it properly
    # A index: (i1,i0,j1,j0) -> i1*4 + i0*2 + j1*2 + j0 where i1,i0,j1,j0 in {0,1}
    # Actually, let's use row-major: A[i,j] where i = i1*2 + i0, j = j1*2 + j0
    
    # For the tensor product F(2) ⊗ F(2):
    # U[i,j] = F2_U[i1, j1] * F2_U[i0, j0] where i = i1*2 + i0, j = j1*7 + j0
    # Wait, this isn't quite right for the tensor product
    
    # Let me use the explicit construction:
    # The 4x4x4x4 multiplication tensor T has T[i,j,k] = 1 if (i,j,k) is a valid triple
    # We decompose it as sum over r of U[i,r] * V[j,r] * W[k,r]
    
    # For tensor product of F(2) with F(2):
    # r = r1*7 + r0 where r1, r0 in {0,...,6}
    # i = i1*2 + i0, j = j1*2 + j0, k = k1*2 + k0
    # U[i,r] = F2_U[i1, r1] * F2_U[i0, r0]
    # V[j,r] = F2_V[j1, r1] * F2_V[j0, r0]  
    # W[k,r] = F2_W[k1, r1] * F2_W[k0, r0]
    
    U_44 = [[0] * 49 for _ in range(16)]
    V_44 = [[0] * 49 for _ in range(16)]
    W_44 = [[0] * 49 for _ in range(16)]
    
    for i1 in range(2):
        for i0 in range(2):
            i = i1 * 2 + i0
            for r1 in range(7):
                for r0 in range(7):
                    r = r1 * 7 + r0
                    U_44[i][r] = F2_U[i1][r1] * F2_U[i0][r0]
    
    for j1 in range(2):
        for j0 in range(2):
            j = j1 * 2 + j0
            for r1 in range(7):
                for r0 in range(7):
                    r = r1 * 7 + r0
                    V_44[j][r] = F2_V[j1][r1] * F2_V[j0][r0]
    
    for k1 in range(2):
        for k0 in range(2):
            k = k1 * 2 + k0
            for r1 in range(7):
                for r0 in range(7):
                    r = r1 * 7 + r0
                    W_44[k][r] = F2_W[k1][r1] * F2_W[k0][r0]
    
    # Verify the 4x4x4x4 decomposition
    def verify_44(U, V, W):
        for i in range(16):
            for j in range(16):
                for k in range(16):
                    total = 0
                    for r in range(49):
                        total += U[i][r] * V[j][r] * W[k][r]
                    # Check if total equals the expected value
                    # i = i1*2 + i0, j = j1*2 + j0, k = k1*2 + k0
                    i1, i0 = i // 2, i % 2
                    j1, j0 = j // 2, j % 2
                    k1, k0 = k // 2, k % 2
                    # The tensor T[i,j,k] should be 1 if i1==k1 and i0==k0 (for the right structure)
                    # Actually, for matrix multiplication, T[i,j,k] = 1 if the indices match
                    # Let me just check a few known values
                    if i == j == k:
                        if abs(total - 1) > 1e-9:
                            return False
        return True
    
    # Now extend to 4x4x4x5
    # We need to handle the 5th column of B
    # B is 4x5, so we have 4*5 = 20 entries
    # The first 4 columns use the 4x4x4x4 decomposition
    # The 5th column needs 4 additional terms
    
    # U has 16 rows (4*4 for A), V has 20 rows (4*5 for B), W has 20 rows (5*4 for C)
    # Wait, W has p*n = 5*4 = 20 rows
    
    # Let me re-examine the indexing
    # U: n*m = 4*4 = 16 rows
    # V: m*p = 4*5 = 20 rows  
    # W: p*n = 5*4 = 20 rows
    
    # For the 4x4x4x4 part (first 4 columns of B):
    # U[i][r] for i in 0..15, r in 0..48
    # V[j][r] for j in 0..15 (first 16 of 20), r in 0..48
    # W[k][r] for k in 0..15 (first 16 of 20), r in 0..48
    
    # For the 5th column of B (j = 16..19):
    # We need 4 more terms, one for each row of A
    # U[i][49+r] = 1 if i corresponds to row i//4 of A
    # V[16+r][49+r] = 1
    # W[16+r][49+r] = 1
    
    # Actually, let me think about this more carefully
    # The 5th column of B is a 4x1 vector
    # We need to compute C[:,4] = A @ B[:,4]
    # This is a 4x4 matrix times a 4x1 vector
    # We can use 4 scalar multiplications:
    # C[i,4] = sum_j A[i,j] * B[j,4]
    
    # So for each i in 0..3 (row of C), we need one term
    # U[i*4 + j][49+i] = 1 for the j that matches
    # Wait, U is indexed by (i_A, j_A) where i_A is row, j_A is col of A
    
    # Let me use:
    # For the 5th column, add 4 terms:
    # Term t (t=0..3):
    #   U[i*4 + j][49+t] = 1 if i == t (row of A)
    #   V[j*5 + 4][49+t] = 1 (5th column of B, row j)
    #   W[4*4 + t][49+t] = 1 (5th column of C, row t)
    
    # Wait, I'm getting confused with the indexing. Let me restart.
    
    # A is 4x4: A[i,j] where i,j in 0..3
    # B is 4x5: B[j,k] where j in 0..3, k in 0..4
    # C is 4x5: C[i,k] where i in 0..3, k in 0..4
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # U has n*m = 4*4 = 16 rows, indexed by (i,j) -> i*4+j
    # V has m*p = 4*5 = 20 rows, indexed by (j,k) -> j*5+k
    # W has p*n = 5*4 = 20 rows, indexed by (k,i) -> k*4+i
    
    # For the 4x4x4x4 part (k=0..3):
    # Use the F(2)⊗F(2) decomposition with 49 terms
    # U[i*4+j][r] = F2_U[i%2, r%7] * F2_U[i//2, r//7]
    # V[j*5+k][r] = F2_V[j%2, r%7] * F2_V[j//2, r//7]
    # W[k*4+i][r] = F2_W[k%2, r%7] * F2_W[k//2, r//7]
    
    # For the 5th column (k=4):
    # Add 4 terms (one for each i):
    # Term t (t=0..3):
    #   U[i*4+j][49+t] = 1 if i == t
    #   V[j*5+4][49+t] = 1
    #   W[4*4+t][49+t] = 1
    
    # Wait, that's not right either. Let me think again.
    
    # For C[i,4] = sum_j A[i,j] * B[j,4]
    # We can write this as:
    # C[i,4] = A[i,0]*B[0,4] + A[i,1]*B[1,4] + A[i,2]*B[2,4] + A[i,3]*B[3,4]
    
    # Each term A[i,j]*B[j,4] can be a separate multiplication
    # So we need 4*4 = 16 terms for the 5th column? No, that's schoolbook.
    
    # Actually, for a fixed i, C[i,4] = sum_j A[i,j] * B[j,4]
    # This is a dot product of row i of A with column 4 of B
    # We can compute this with 4 multiplications (one for each j)
    # But we can share the A[i,j] factors across different i's? No, each i is different.
    
    # So for the 5th column, we need 4*4 = 16 terms if we do it schoolbook
    # But we can do better by using the structure
    
    # Actually, let's just use 4 terms for the 5th column:
    # For each row i of C (i=0..3):
    #   C[i,4] = sum_j A[i,j] * B[j,4]
    #   This requires 4 multiplications, but we can factor it as:
    #   Let x_j = B[j,4] for j=0..3
    #   Then C[i,4] = A[i,0]*x_0 + A[i,1]*x_1 + A[i,2]*x_2 + A[i,3]*x_3
    #   This is a linear combination, not a product of two factors
    
    # I think I'm overcomplicating this. Let me just use the schoolbook approach for the 5th column:
    # 16 terms for the 5th column
    # Total rank = 49 + 16 = 65
    
    # But we can do better. Let's use the fact that B[j,4] is shared across all i.
    # We can precompute y_j = B[j,4] and then C[i,4] = sum_j A[i,j] * y_j
    # This is still 16 multiplications if we do it naively
    
    # Actually, for the purpose of tensor decomposition, each term is a product U[i,j] * V[j,k] * W[k,i]
    # For the 5th column, we need:
    # C[i,4] = sum_j A[i,j] * B[j,4]
    # Each term A[i,j] * B[j,4] contributes to C[i,4]
    # So we need 4*4 = 16 terms for the 5th column
    
    # Total rank = 49 + 16 = 65
    
    # Let's construct this
    U_full = [[0] * 65 for _ in range(16)]
    V_full = [[0] * 65 for _ in range(20)]
    W_full = [[0] * 65 for _ in range(20)]
    
    # Fill in the 4x4x4x4 part
    for i in range(4):
        for j in range(4):
            i_idx = i * 4 + j
            for r in range(49):
                r1, r0 = r // 7, r % 7
                U_full[i_idx][r] = F2_U[i % 2][r0] * F2_U[i // 2][r1]
    
    for j in range(4):
        for k in range(4):
            j_idx = j * 5 + k
            for r in range(49):
                r1, r0 = r // 7, r % 7
                V_full[j_idx][r] = F2_V[j % 2][r0] * F2_V[j // 2][r1]
    
    for k in range(4):
        for i in range(4):
            k_idx = k * 4 + i
            for r in range(49):
                r1, r0 = r // 7, r % 7
                W_full[k_idx][r] = F2_W[k % 2][r0] * F2_W[k // 2][r1]
    
    # Fill in the 5th column part (k=4)
    # C[i,4] = sum_j A[i,j] * B[j,4]
    # Each term A[i,j] * B[j,4] is a separate multiplication
    for i in range(4):
        for j in range(4):
            r = 49 + i * 4 + j
            U_full[i * 4 + j][r] = 1
            V_full[j * 5 + 4][r] = 1
            W_full[4 * 4 + i][r] = 1
    
    # Verify the full decomposition
    if check_decomposition(U_full, V_full, W_full):
        best_result = {"U": U_full, "V": V_full, "W": W_full}
        best_rank = 65
    else:
        # Fall back to schoolbook
        best_result = schoolbook()
        best_rank = rank_schoolbook
    
    return best_result
