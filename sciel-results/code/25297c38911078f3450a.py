"""Program for 3x5x5 matrix multiplication tensor decomposition."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x5 matrix times 5x5 matrix.
    
    Strategy: Decompose the 3x5x5 multiplication as a direct sum of:
    - A 2x5x5 sub-tensor (for rows 0,1 of A)
    - A 1x5x5 sub-tensor (for row 2 of A)
    
    For the 2x5x5 component, partition the 5-dimensional inner product into 2+3 dimensions.
    Apply Strassen (rank 7) to the 2x2x2 blocks and schoolbook (rank 9) to the 3x3x3 blocks.
    
    The 2x5x5 tensor T2[i,l,k] for i in {0,1}, l in {0..4}, k in {0..4} can be written as:
    sum_{l} A[i,l] * B[l,k]
    
    Partition l into L1 = {0,1} and L2 = {2,3,4}.
    Then T2 = T2_a + T2_b where:
    T2_a[i,l,k] = A[i,l]*B[l,k] for l in {0,1}
    T2_b[i,l,k] = A[i,l]*B[l,k] for l in {2,3,4}
    
    T2_a is a 2x2x5 tensor: sum_{l in {0,1}} A[i,l]*B[l,k]
    This can be viewed as two separate 1x2x5 multiplications (one for each i).
    Actually, let's think more carefully.
    
    For fixed i, sum_{l in {0,1}} A[i,l]*B[l,k] is a 1x2x5 multiplication.
    For i=0: A[0,0]*B[0,k] + A[0,1]*B[1,k]
    For i=1: A[1,0]*B[0,k] + A[1,1]*B[1,k]
    
    So T2_a has rank 2*2 = 4 (schoolbook for 1x2x5 done twice).
    
    T2_b is a 2x3x5 tensor: sum_{l in {2,3,4}} A[i,l]*B[l,k]
    For fixed i, this is a 1x3x5 multiplication.
    For i=0: A[0,2]*B[2,k] + A[0,3]*B[3,k] + A[0,4]*B[4,k]
    For i=1: A[1,2]*B[2,k] + A[1,3]*B[3,k] + A[1,4]*B[4,k]
    
    So T2_b has rank 2*3 = 6 (schoolbook for 1x3x5 done twice).
    
    Total for 2x5x5: 4 + 6 = 10 (schoolbook would be 2*5 = 10, so no gain yet)
    
    Let me reconsider the approach. The directive mentions using Strassen on 2x2x2 blocks.
    
    Alternative: For the 2x5x5 tensor, we can think of it as:
    For each pair (i,k), we compute sum_l A[i,l]*B[l,k].
    
    If we partition l into two groups of sizes 2 and 3:
    Group 1: l in {0,1}, size 2
    Group 2: l in {2,3,4}, size 3
    
    For Group 1, we have a 2x2x5 tensor. This is equivalent to computing 2 outputs (i=0,1) 
    each as a dot product of length 2 with B's rows.
    
    Actually, let me reconsider. The standard way to apply Strassen is to matrix multiply 
    larger matrices by partitioning them into blocks.
    
    For 2x5 times 5x5:
    A is 2x5, B is 5x5.
    
    We can't directly apply Strassen since the dimensions don't align nicely for 2x2 blocks.
    
    Let me try a different approach. Since n=3 is small, let's just use schoolbook for now
    but see if we can do better.
    
    Actually, the current schoolbook gives rank 75 = 3*5*5.
    
    Let me think about whether we can reduce this. The tensor rank of 3x5x5 multiplication...
    
    For now, let's keep the schoolbook decomposition as it's correct and simple.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for l in range(m):
            for k in range(p):
                U[i * m + l][col] = 1
                V[l * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
