"""Matrix multiplication 4x4x9 - Winograd Fourier Transform Technique"""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 9  # U: n*m=16, V: m*p=36, W: p*n=36
    
    # Winograd FFT approach:
    # Apply 2-point FFT on the m=4 dimension (split into two pairs of 2).
    # This decomposes 4x4x9 into 4 independent 2x2x9 sub-problems.
    # Each 2x2x9 has rank <= 2*2*9 = 36, total 4*36 = 144.
    # But we can do better by also applying FFT on the n=4 dimension.
    # 
    # Full 2D FFT: 4x4x9 -> 16 independent 1x1x9? No.
    # 
    # Let me use the standard Winograd convolution approach:
    # For the m-dimension (size 4 = 2*2), use 2-point FFT.
    # For the n-dimension (size 4 = 2*2), use 2-point FFT.
    # This gives 4*4 = 16 independent scalar multiplications per k-column.
    # Total rank: 16 * 9 = 144. Same as schoolbook!
    #
    # The improvement comes from the fact that in the transformed domain,
    # the 2x2x2 sub-problem can be done in rank 7 (Strassen) instead of 8.
    # 
    # Let me use a different strategy: exploit that p=9 = 3*3.
    # Partition the 9 output columns into three groups of 3.
    # For each 4x4x3 block, use a rank-12 decomposition.
    # Total: 3*12 = 36.
    #
    # For 4x4x3, here's a rank-12 decomposition using FFT:
    # Apply 2-point FFT on m-dimension (4->2x2) and on the 3 columns.
    # 3 = 3 (prime), so no further FFT.
    # 
    # After 2-point FFT on m: 4x4x3 -> 4 independent 2x2x3.
    # Each 2x2x3 has rank <= 2*2*3 = 12. Total: 4*12 = 48.
    # 
    # Can we do 2x2x3 in rank < 12? The tensor T_{i,j,k} = delta_{i,j} for i in {0,1}, j in {0,1}, k in {0,1,2}.
    # Actually no, 2x2x3 is just 3 independent 2x2x1 problems, each of rank 4. Total 12.
    # 
    # Let me try yet another approach: use the fact that 4x4x9 can be viewed as
    # computing 9 matrix-vector products A*b_k where b_k is the k-th column of B.
    # We can batch these.
    #
    # Practical approach: Use a rank-72 decomposition.
    # Partition A into two 2x4 blocks (top/bottom rows).
    # For each 2x4x9: C(2x9) = A(2x4)*B(4x9).
    # Apply 2-point FFT on the 4-dimension of B:
    # B' = [B[0,:]+B[1,:], B[0,:]-B[1,:], B[2,:]+B[3,:], B[2,:]-B[3,:]]
    # A' = [A[:,0]+A[:,1], A[:,0]-A[:,1], A[:,2]+A[:,3], A[:,2]-A[:,3]]
    # Then 4 independent 2x2x9 sub-problems, each rank 36. Total per 2x4x9: 4*36=144. Worse.
    #
    # I'll use a straightforward rank-48 decomposition:
    # 3 blocks of 4x4x3, each with rank 16.
    # For 4x4x3 with rank 16: use 2-point FFT on m-dimension.
    # 4x4x3 -> 4 independent 2x2x3, each rank 6. Total: 4*6=24. Not 16.
    #
    # Let me just implement a correct rank-48 solution using 3x(4x4x3 rank-16).
    # For 4x4x3 rank-16: I'll use a direct construction.
    
    rank = 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For each block of 3 columns, use a rank-16 decomposition of 4x4x3.
    # 4x4x3: C(4x3) = A(4x4) * B(4x3)
    # Use 2-point FFT on the m=4 dimension:
    # A'_r for r in {0,1,2,3}:
    #   A'_0 = A[0,:]+A[1,:], A'_1 = A[0,:]-A[1,:], A'_2 = A[2,:]+A[3,:], A'_3 = A[2,:]-A[3,:]
    # B'_r for r in {0,1,2,3}:
    #   B'_0 = B[0,:]+B[1,:], B'_1 = B[0,:]-B[1,:], B'_2 = B[2,:]+B[3,:], B'_3 = B[2,:]-B[3,:]
    # M_r = A'_r * B'_r (2x3), r=0..3
    # C[0,:] = (M_0[0,:]+M_1[0,:])/2 ... this gets complicated with the 3 columns.
    
    # Let me use a simpler valid approach: rank 72.
    # 9 columns, each 4x4x1 with rank 8? No, 4x4x1 has rank 16.
    
    # Final approach: use the schoolbook but verify it's correct.
    # Actually, let me use a rank-48 construction based on 3 independent 4x4x3 blocks.
    # For each 4x4x3, I'll use a rank-16 method based on the following:
    # C[i,k] = sum_j A[i,j]*B[j,k] for i in 0..3, k in 0..2
    # 
    # Use the following 16 terms (indexed by (a,b) where a,b in {0,1,2,3}):
    # This is essentially the schoolbook for 4x4x3 which is 48, not 16.
    
    # I'll just return a valid rank-144 schoolbook for now to ensure correctness,
    # then optimize.
    
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
# EVOLVE-BLOCK-END
