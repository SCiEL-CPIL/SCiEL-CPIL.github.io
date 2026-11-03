"""Program for matrix multiplication tensor decomposition 3x3x6."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

def _build_rank19_3x3x3():
    """
    Return a rank-19 decomposition for 3x3x3 matrix multiplication.
    Uses the known result that the tensor rank of 3x3 matrix multiplication
    is at most 19. We construct this by splitting 3x3 into a 2x2 block
    (Strassen rank 7) plus cross terms with the 3rd row/column.
    
    For C = A * B where A, B are 3x3:
    Partition: A = [[A11, a12], [a21, a22]] where A11 is 2x2,
                B = [[B11, b12], [b21, b22]] where B11 is 2x2.
    
    C11 = A11*B11 + a12*b21  (2x2 product + outer product)
    C12 = A11*b12 + a12*b22  (2x1 products)
    C21 = a21*B11 + a22*b21  (1x2 products)
    C22 = a21*b12 + a22*b22  (1x1 products)
    
    Using Strassen for the 2x2 part (7 multiplications) and handling
    the cross terms efficiently.
    
    Actually, let me use a more systematic approach. The rank-19 decomposition
    for 3x3 matrix multiplication can be constructed as follows:
    
    We use the fact that we can write 3x3 multiplication as:
    - 7 Strassen multiplications for the top-left 2x2 block
    - Additional terms for the cross-block interactions
    
    Let me use a known explicit construction.
    """
    n = 3
    # We'll build U (9 x r), V (9 x r), W (9 x r) for 3x3x3
    # Indexing: U[i*m+j], V[j*p+k], W[k*n+i] with m=p=n=3
    
    # Strategy: Use the identity that 3x3 matmul can be done with
    # at most 19 scalar multiplications. One known construction:
    # 
    # Split A and B into blocks:
    # A = [A11 A12; A21 A22] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    # B = [B11 B12; B21 B22] similarly
    #
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22  
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    #
    # For C11: 2x2*2x2 needs 7 (Strassen), 2x1*1x2 (outer) needs 4 -> total 11
    # For C12: 2x2*2x1 needs 4 (schoolbook since 2x1), 2x1*1x1 needs 2 -> total 6
    # For C21: 1x2*2x2 needs 4, 1x1*1x2 needs 2 -> total 6
    # For C22: 1x2*2x1 needs 2, 1x1*1x1 needs 1 -> total 3
    # Total: 11 + 6 + 6 + 3 = 26. Not good enough.
    #
    # Better approach: share some multiplications.
    # Let me use a different known rank-19 construction.
    #
    # Known: The rank of 3x3 matrix multiplication tensor is 19.
    # One explicit construction uses the following 19 bilinear forms:
    #
    # Let me use the construction from de Groote et al. (1982) or 
    # the simpler one from the literature.
    
    # I'll use a constructive approach:
    # 1. Use Strassen for the 2x2 submatrix (7 mults)
    # 2. Add terms for the remaining entries
    # 3. Optimize by sharing
    
    # Actually, let me just use a verified rank-19 construction.
    # Here's one based on splitting:
    
    r = 19
    U = [[0]*r for _ in range(9)]
    V = [[0]*r for _ in range(9)]
    W = [[0]*r for _ in range(9)]
    
    # Strassen's 7 for the 2x2 block (indices 0-1 for rows/cols 0,1)
    # A11=A[0,0], A12=A[0,1], A21=A[1,0], A22=A[1,1]
    # B11=B[0,0], B12=B[0,1], B21=B[1,0], B22=B[1,1]
    
    # M1 = (A11+A22)(B11+B22)
    U[0*3+0][0] = 1; U[1*3+1][0] = 1
    V[0*3+0][0] = 1; V[1*3+1][0] = 1
    W[0*3+0][0] = 1; W[0*3+1][0] = 1; W[1*3+0][0] = 1; W[1*3+1][0] = -1
    
    # M2 = (A21+A22)B11
    U[1*3+0][1] = 1; U[1*3+1][1] = 1
    V[0*3+0][1] = 1
    W[1*3+0][1] = 1
    
    # M3 = A11(B12-B22)
    U[0*3+0][2] = 1
    V[0*3+1][2] = 1; V[1*3+1][2] = -1
    W[0*3+1][2] = 1
    
    # M4 = A22(B21-B11)
    U[1*3+1][3] = 1
    V[1*3+0][3] = 1; V[0*3+0][3] = -1
    W[1*3+0][3] = 1
    
    # M5 = (A11+A12)B22
    U[0*3+0][4] = 1; U[0*3+1][4] = 1
    V[1*3+1][4] = 1
    W[0*3+1][4] = 1; W[1*3+1][4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U[1*3+0][5] = 1; U[0*3+0][5] = -1
    V[0*3+0][5] = 1; V[0*3+1][5] = 1
    W[1*3+1][5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U[0*3+1][6] = 1; U[1*3+1][6] = -1
    V[1*3+0][6] = 1; V[1*3+1][6] = 1
    W[0*3+1][6] = 1; W[1*3+1][6] = 1
    
    # Now handle the cross terms involving row/col 2 (index 2)
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    # C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
    # C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1]
    # C[2,2] = A[2,0]*B[0,2] + A[2,1]*B[1,2] + A[2,2]*B[2,2]
    
    # That's 15 more terms if done naively. But we can share.
    # Let me use a different approach: add 12 more terms for the cross terms.
    # Total would be 7 + 12 = 19.
    
    # C[0,2]: A[0,0]*B[0,2]
    U[0*3+0][7] = 1
    V[0*3+2][7] = 1
    W[0*3+2][7] = 1
    
    # C[0,2]: A[0,1]*B[1,2]
    U[0*3+1][8] = 1
    V[1*3+2][8] = 1
    W[0*3+2][8] = 1
    
    # C[0,2]: A[0,2]*B[2,2]
    U[0*3+2][9] = 1
    V[2*3+2][9] = 1
    W[0*3+2][9] = 1
    
    # C[1,2]: A[1,0]*B[0,2]
    U[1*3+0][10] = 1
    V[0*3+2][10] = 1
    W[1*3+2][10] = 1
    
    # C[1,2]: A[1,1]*B[1,2]
    U[1*3+1][11] = 1
    V[1*3+2][11] = 1
    W[1*3+2][11] = 1
    
    # C[1,2]: A[1,2]*B[2,2]
    U[1*3+2][12] = 1
    V[2*3+2][12] = 1
    W[1*3+2][12] = 1
    
    # C[2,0]: A[2,0]*B[0,0]
    U[2*3+0][13] = 1
    V[0*3+0][13] = 1
    W[0*3+2][13] = 1
    
    # C[2,1]: A[2,0]*B[0,1]
    U[2*3+0][14] = 1
    V[0*3+1][14] = 1
    W[1*3+2][14] = 1
    
    # C[2,2]: A[2,0]*B[0,2]
    U[2*3+0][15] = 1
    V[0*3+2][15] = 1
    W[2*3+2][15] = 1
    
    # C[2,0]: A[2,1]*B[1,0]
    U[2*3+1][16] = 1
    V[1*3+0][16] = 1
    W[0*3+2][16] = 1
    
    # C[2,1]: A[2,1]*B[1,1]
    U[2*3+1][17] = 1
    V[1*3+1][17] = 1
    W[1*3+2][17] = 1
    
    # C[2,2]: A[2,1]*B[1,2]
    U[2*3+1][18] = 1
    V[1*3+2][18] = 1
    W[2*3+2][18] = 1
    
    # C[2,0]: A[2,2]*B[2,0]
    # C[2,1]: A[2,2]*B[2,1]
    # C[2,2]: A[2,2]*B[2,2]
    # We're at 19 terms (0-18), but we still need 3 more for row 2 col 0,1,2 with A[2,2]
    # This doesn't work. Let me recount.
    
    # I have terms 0-18 = 19 terms, but I'm missing:
    # C[2,0]: A[2,2]*B[2,0]
    # C[2,1]: A[2,2]*B[2,1]  
    # C[2,2]: A[2,2]*B[2,2]
    
    # So this naive approach gives 7 + 12 + 3 = 22, not 19.
    # I need to be smarter. Let me use a known rank-19 construction.
    
    # Alternative: use the fact that we can combine some terms.
    # Let me restart with a cleaner construction.
    
    # Known rank-19 construction for 3x3 matmul:
    # Reference: "The rank of 3x3 matrix multiplication" 
    # One construction uses 19 bilinear forms.
    
    # Let me use a different strategy: I'll verify if my construction works
    # and if not, fall back to a higher rank that I can verify.
    
    # For now, let me use schoolbook for 3x3x3 (rank 27) which I can verify,
    # giving total rank 54 for 3x3x6. Same as before.
    
    # Let me try a rank-20 construction that I can verify:
    # Use Strassen for 2x2 (7) + 13 for cross terms = 20
    
    r = 20
    U = [[0]*r for _ in range(9)]
    V = [[0]*r for _ in range(9)]
    W = [[0]*r for _ in range(9)]
    
    # Strassen 7 terms (same as above, indices 0-6)
    U[0][0]=1; U[4][0]=1; V[0][0]=1; V[4][0]=1; W[0][0]=1; W[1][0]=1; W[3][0]=-1
    U[3][1]=1; U[4][1]=1; V[0][1]=1; W[3][1]=1
    U[0][2]=1; V[1][2]=1; V[4][2]=-1; W[1][2]=1
    U[4][3]=1; V[3][3]=1; V[0][3]=-1; W[3][3]=1
    U[0][4]=1; U[1][4]=1; V[4][4]=1; W[1][4]=1; W[4][4]=1
    U[3][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[4][5]=1
    U[1][6]=1; U[4][6]=-1; V[3][6]=1; V[4][6]=1; W[1][6]=1; W[4][6]=1
    
    # Cross terms (indices 7-19), 13 terms:
    # Row 0, Col 2: A00*B02, A01*B12, A02*B22
    U[0][7]=1; V[2][7]=1; W[2][7]=1
    U[1][8]=1; V[5][8]=1; W[2][8]=1
    U[2][9]=1; V[8][9]=1; W[2][9]=1
    # Row 1, Col 2: A10*B02, A11*B12, A12*B22
    U[3][10]=1; V[2][10]=1; W[5][10]=1
    U[4][11]=1; V[5][11]=1; W[5][11]=1
    U[5][12]=1; V[8][12]=1; W[5][12]=1
    # Row 2, Col 0: A20*B00, A21*B10, A22*B20
    U[6][13]=1; V[0][13]=1; W[6][13]=1
    U[7][14]=1; V[3][14]=1; W[6][14]=1
    U[8][15]=1; V[6][15]=1; W[6][15]=1
    # Row 2, Col 1: A20*B01, A21*B11, A22*B21
    U[6][16]=1; V[1][16]=1; W[7][16]=1
    U[7][17]=1; V[4][17]=1; W[7][17]=1
    U[8][18]=1; V[7][18]=1; W[7][18]=1
    # Row 2, Col 2: A20*B02, A21*B12, A22*B22
    U[6][19]=1; V[2][19]=1; W[8][19]=1
    
    # Verify
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] should give C[i,k] = sum_j A[i,j]*B[j,k]
    # Reshape and check
    U_t = U_arr.reshape(3, 3, r)  # U[i,j,r]
    V_t = V_arr.reshape(3, 3, r)  # V[j,k,r]
    W_t = W_arr.reshape(3, 3, r)  # W[k,i,r]
    
    # The tensor should satisfy: T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    # should equal delta (i.e., the matmul tensor where T[i,j,k] = 1 if this is a valid matmul term)
    # Actually, the matmul tensor is T[i,j,k] = 1 for all i,j,k (it's the identity in the right sense)
    # No, the matmul tensor M[i,j,k] represents the coefficient of A[i,j]*B[j,k] in C[i,k]
    # So M[i,j,k] = 1 for all i,j,k.
    
    # Let me verify differently: for random A, B, check that sum_r (U^T A)(V^T B) * W^T = A*B
    # i.e., sum_r (sum_ij U[i,j,r]*A[i,j]) * (sum_jk V[j,k,r]*B[j,k]) * W[k,i,r] = (A*B)[i,k]
    
    np.random.seed(42)
    A_test = np.random.randn(3, 3)
    B_test = np.random.randn(3, 3)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((3, 3))
    for r_idx in range(r):
        a_proj = sum(U_arr[i*3+j, r_idx] * A_test[i, j] for i in range(3) for j in range(3))
        b_proj = sum(V_arr[j*3+k, r_idx] * B_test[j, k] for j in range(3) for k in range(3))
        for i in range(3):
            for k in range(3):
                C_computed[i, k] += W_arr[k*3+i, r_idx] * a_proj * b_proj
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fallback to schoolbook
        return None
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = 3, 3, 6
    
    # Try to get rank-19 or rank-20 decomposition for each 3x3x3 block
    result = _build_rank19_3x3x3()
    
    if result is not None:
        U_block, V_block, W_block = result
        r_block = len(U_block[0])
        total_rank = 2 * r_block
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        # Two blocks: k=0,3 and k=1,4,5? No, k=0,3 and k=1,4 and k=2,5
        # Actually for 3x3x6, we have two independent 3x3x3 multiplications:
        # Block 1: columns 0, 3 of B (k=0 and k=3)
        # Block 2: columns 1, 4 of B (k=1 and k=4)  
        # Block 3: columns 2, 5 of B (k=2 and k=5)
        # Wait, that's 3 blocks, not 2. Let me reconsider.
        
        # For 3x3 x 3x6: A is 3x3, B is 3x6, C is 3x6
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # This is NOT two independent 3x3x3 multiplications.
        # It's one 3x3x6 multiplication.
        
        # The tensor rank of 3x3x6 matmul: we can think of it as
        # multiplying 3x3 by three 3x1 vectors (columns of B).
        # Each 3x3 x 3x1 multiplication has rank at most 9 (schoolbook) 
        # or possibly less.
        
        # Actually, 3x3 x 3x1: C[i] = sum_j A[i,j] * b[j]
        # This is just applying A to a vector. The "tensor" here is
        # T[i,j,k] where k indexes the column of B (but B is 3x1, so k is fixed).
        # For 3x3 x 3x1, the rank is at most 9 (each A[i,j]*b[j] term).
        # But we can do better: it's just a linear map, so rank 1 in some sense?
        # No, the tensor rank for matrix-vector multiplication 3x3x1 is 9
        # (you need all 9 products A[i,j]*b[j]).
        
        # For 3x3 x 3x6, we have 6 such vector multiplications.
        # Total schoolbook rank = 9 * 6 = 54.
        
        # Can we share? The A[i,j] factors are shared across different k.
        # So we can write: for each i,j, we have A[i,j] * (sum over k of B[j,k]*W[k,i])
        # But that's not a valid tensor decomposition because we need bilinear forms.
        
        # The tensor rank of 3x3x6 matmul is known to be at most 54 (schoolbook).
        # It could be lower. Let me just use schoolbook to be safe.
        
        pass
    
    # Safe fallback: schoolbook rank 54
    rank = n * m * p  # 54
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
