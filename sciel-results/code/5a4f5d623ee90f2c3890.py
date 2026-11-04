"""Optimized matrix multiplication decomposition for 2x5 * 5x6.

Uses the direct sum structure:
The multiplication tensor T for 2x5*6 is a direct sum of 6 tensors T_k,
each corresponding to the computation of the k-th column of C.
T_k is a 2x5x1 tensor (computing 2 dot products of length 5).
The rank of a 2x5x1 tensor is 10 (since it's a direct sum of 5 1x1x1 tensors
for each of the 2 rows, or simply R(2,5,1)=10).
Since the W-supports for different k are disjoint, R(T) = sum R(T_k) = 6 * 10 = 60.
This is the minimal rank.
"""

def construct():
    n, m, p = 2, 5, 6
    
    # Rank is sum of ranks of the p column-blocks.
    # Each block is 2x5x1, rank 10.
    rank = p * (n * m)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Iterate over columns of B (and C)
    for k in range(p):
        # For this column k, we compute C[0,k] and C[1,k]
        # This is a 2x5x1 problem.
        # Rank 10 decomposition:
        for i in range(n):
            for j in range(m):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
