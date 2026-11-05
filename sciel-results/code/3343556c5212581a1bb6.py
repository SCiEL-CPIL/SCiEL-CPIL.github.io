"""Tensor decomposition for 3x3 * 3x6 matrix multiplication."""
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    # For multiplying a 3x3 matrix by a 3x6 matrix,
    # the output C is 3x6 where C[i,k] = sum_j A[i,j]*B[j,k]
    # This decomposes into 6 independent 3x3 matrix multiplications (one for each column k)
    # Each 3x3 multiplication has rank 1 in the trivial sense but we need the full structure.
    # Actually, the tensor for (3,3,6) matrix mult has rank at most 3*3*6 = 54 (schoolbook).
    # But we can do better: the 6 columns of the right matrix are independent,
    # so we need 6 copies of the 3x3 tensor, each of rank at most 9.
    # Total rank = 6 * 9 = 54 for schoolbook.
    #
    # However, the problem is asking for the minimum rank. For (3,3,3) the min rank is 19 or 20.
    # For (3,3,6), since the p=6 dimension just replicates the structure,
    # the rank should be 6 * (rank of 3x3 mult) if there's no sharing.
    # But actually, the tensor T_{i,j,k} = A_{i,j} * B_{j,k} has the structure that for each k,
    # it's a 3x3 matrix multiplication tensor. Since the B indices (j,k) with k fixed give independent
    # problems, the rank is at most 6 * R(3,3,3) where R is the rank of 3x3 mult.
    # The best known rank for 3x3 is 19 (Strassen-like) or 20.
    # But we need integer/half-integer entries.
    #
    # Let me just output the schoolbook decomposition of rank 54, which is safe.
    # The rank 54 decomposition uses U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    # for each (i,j,k) triple.
    
    n, m, p = 3, 3, 6
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
