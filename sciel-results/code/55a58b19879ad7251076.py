"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=6
    
    # For multiplying a 3x3 matrix A by a 3x6 matrix B to get a 3x6 matrix C,
    # we can use a bilinear algorithm. The schoolbook method requires 3*3*6 = 54 multiplications.
    # However, we can potentially do better using Strassen-like ideas or other decompositions.
    
    # One approach: treat this as two 3x3 matrix multiplications (since B is 3x6, we can split into two 3x3 blocks)
    # B = [B1 | B2] where B1 and B2 are 3x3
    # C = A*B = [A*B1 | A*B2]
    # Each 3x3 * 3x3 multiplication can be done with rank 19 (known result for 3x3 matrix multiplication)
    # But we need to share some computations.
    
    # Actually, let's think about this differently. The standard rank for 3x3 matrix multiplication is 19.
    # For 3x3 * 3x6, we can view it as computing 6 column vectors of the form A * b where b is a 3x1 vector.
    
    # A simpler approach: use the fact that we can decompose the problem.
    # Let's try to find a rank-18 decomposition or similar.
    
    # For now, let's use a known decomposition. The rank of the 3x3x6 matrix multiplication tensor is at most 18.
    # Here's one construction:
    
    # We'll use a combination of Strassen's algorithm adapted for rectangular matrices.
    
    # Actually, let me construct a valid bilinear algorithm with rank 18.
    
    # The key insight: C[i][k] = sum_j A[i][j] * B[j][k] for i in 0..2, k in 0..5
    
    # Let's define 18 bilinear terms. We'll use a systematic approach.
    
    rank = 18
    U = [[0] * rank for _ in range(n * m)]  # 9 x 18
    V = [[0] * rank for _ in range(m * p)]  # 18 x 18
    W = [[0] * rank for _ in range(p * n)]  # 18 x 18
    
    # Define the 18 terms. Each term t has U[i,j] coefficient, V[j,k] coefficient, and W[k,i] coefficient.
    # C[i,k] = sum_t U[i,j_t] * V[j_t,k] * W[k,i] ... wait, that's not right.
    
    # The tensor decomposition: T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    # So C[i,k] = sum_j A[i,j] * B[j,k] = sum_t (sum_j U[i*m+j,t] * A[i,j]) * (sum_j V[j*p+k,t] * B[j,k]) * W[k*n+i,t]
    
    # Let me define the 18 terms explicitly.
    
    # Term 0: A[0,0]*B[0,0] -> contributes to C[0,0]
    U[0*3+0][0] = 1; V[0*6+0][0] = 1; W[0*3+0][0] = 1
    # Term 1: A[0,0]*B[0,1] -> contributes to C[0,1]
    U[0*3+0][1] = 1; V[0*6+1][1] = 1; W[1*3+0][1] = 1
    # Term 2: A[0,0]*B[0,2] -> contributes to C[0,2]
    U[0*3+0][2] = 1; V[0*6+2][2] = 1; W[2*3+0][2] = 1
    # Term 3: A[0,0]*B[0,3] -> contributes to C[0,3]
    U[0*3+0][3] = 1; V[0*6+3][3] = 1; W[3*3+0][3] = 1
    # Term 4: A[0,0]*B[0,4] -> contributes to C[0,4]
    U[0*3+0][4] = 1; V[0*6+4][4] = 1; W[4*3+0][4] = 1
    # Term 5: A[0,0]*B[0,5] -> contributes to C[0,5]
    U[0*3+0][5] = 1; V[0*6+5][5] = 1; W[5*3+0][5] = 1
    # Term 6: A[0,1]*B[1,0] -> contributes to C[0,0]
    U[0*3+1][6] = 1; V[1*6+0][6] = 1; W[0*3+0][6] = 1
    # Term 7: A[0,1]*B[1,1] -> contributes to C[0,1]
    U[0*3+1][7] = 1; V[1*6+1][7] = 1; W[1*3+0][7] = 1
    # Term 8: A[0,1]*B[1,2] -> contributes to C[0,2]
    U[0*3+1][8] = 1; V[1*6+2][8] = 1; W[2*3+0][8] = 1
    # Term 9: A[0,1]*B[1,3] -> contributes to C[0,3]
    U[0*3+1][9] = 1; V[1*6+3][9] = 1; W[3*3+0][9] = 1
    # Term 10: A[0,1]*B[1,4] -> contributes to C[0,4]
    U[0*3+1][10] = 1; V[1*6+4][10] = 1; W[4*3+0][10] = 1
    # Term 11: A[0,1]*B[1,5] -> contributes to C[0,5]
    U[0*3+1][11] = 1; V[1*6+5][11] = 1; W[5*3+0][11] = 1
    # Term 12: A[0,2]*B[2,0] -> contributes to C[0,0]
    U[0*3+2][12] = 1; V[2*6+0][12] = 1; W[0*3+0][12] = 1
    # Term 13: A[0,2]*B[2,1] -> contributes to C[0,1]
    U[0*3+2][13] = 1; V[2*6+1][13] = 1; W[1*3+0][13] = 1
    # Term 14: A[0,2]*B[2,2] -> contributes to C[0,2]
    U[0*3+2][14] = 1; V[2*6+2][14] = 1; W[2*3+0][14] = 1
    # Term 15: A[0,2]*B[2,3] -> contributes to C[0,3]
    U[0*3+2][15] = 1; V[2*6+3][15] = 1; W[3*3+0][15] = 1
    # Term 16: A[0,2]*B[2,4] -> contributes to C[0,4]
    U[0*3+2][16] = 1; V[2*6+4][16] = 1; W[4*3+0][16] = 1
    # Term 17: A[0,2]*B[2,5] -> contributes to C[0,5]
    U[0*3+2][17] = 1; V[2*6+5][17] = 1; W[5*3+0][17] = 1
    
    # Wait, this only covers row 0 of C. I need to cover all rows.
    # This approach won't work with rank 18 if I'm doing it naively.
    
    # Let me reconsider. The schoolbook method requires 54 terms. To get rank 18, I need significant sharing.
    
    # Let's use a different approach. I'll use the known result that the rank of the 3x3 matrix multiplication tensor is 19.
    # For 3x3 * 3x6, we can think of it as 6 separate 3x1 * 3x6 column computations, but that's not quite right.
    
    # Actually, C is 3x6, so C[i,k] = sum_j A[i,j] * B[j,k].
    # This is equivalent to computing 6 column vectors C[:,k] = A * B[:,k].
    
    # One valid decomposition with rank 18:
    # We can use the fact that we can share computations across the 6 output columns.
    
    # Let me define a proper rank-18 algorithm.
    
    # Reset and try again with a systematic construction.
    
    rank = 18
    U = [[0] * rank for _ in range(n * m)]  # 9 x 18
    V = [[0] * rank for _ in range(m * p)]  # 18 x 18
    W = [[0] * rank for _ in range(p * n)]  # 18 x 18
    
    # Define terms such that for each (i,k) in C, C[i,k] = sum_t U[i*3+j, t] * V[j*6+k, t] * W[k*3+i, t] * A[i,j] * B[j,k]
    # Actually: C[i,k] = sum_t (sum_j U[i*3+j, t] * A[i,j]) * (sum_j V[j*6+k, t] * B[j,k]) * W[k*3+i, t]
    
    # Let me define 18 terms. I'll use a pattern based on Strassen-like ideas.
    
    # For simplicity and correctness, let me just use the schoolbook method with rank 54,
    # but that's not an improvement. Let me try to find a better decomposition.
    
    # Actually, let me verify: is rank 18 achievable? The problem says to find rank-minimal.
    # The known lower bound for 3x3 matrix multiplication is 19. For 3x3 * 3x6, it might be lower.
    
    # Let me just provide a valid decomposition. I'll use rank 54 (schoolbook) to be safe,
    # but that's the same as the current program. Let me try to do better.
    
    # Here's a rank-18 construction:
    
    # We'll define 18 bilinear forms. The key is to share computations.
    
    # Let me use a different indexing. Define the 18 terms as follows:
    
    # For j in 0..2 (the summation index), and for groups of output columns:
    
    # Term 3*j + 0: A[0,j] * (B[j,0] + B[j,1]) -> contributes to C[0,0] and C[0,1]
    # Term 3*j + 1: A[1,j] * (B[j,0] + B[j,1]) -> contributes to C[1,0] and C[1,1]
    # Term 3*j + 2: A[2,j] * (B[j,0] + B[j,1]) -> contributes to C[2,0] and C[2,1]
    
    # But this doesn't directly give us individual entries.
    
    # Let me just use a valid schoolbook decomposition with rank 54 for now, 
    # but structured properly.
    
    rank = n * m * p  # 54
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
# EVOLVE-BLOCK-END
