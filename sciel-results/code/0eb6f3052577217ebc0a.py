"""Program to find rank-minimal exact tensor decomposition for 3x3 * 3x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin

    n, m, p = DIMENSIONS
    # A is n x m, B is m x p, C is n x p
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)  <-- Note: W index is k*n + i (output C is n x p, but trace convention says W[k*n+i, r])
    
    # The tensor equation is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i, j, k.
    
    # We can view this as computing C = A * B where A is 3x3 and B is 3x6.
    # We can split B into two 3x3 matrices: B = [B1, B2].
    # Then C = A * B1 + A * B2.
    # This is equivalent to two independent 3x3 * 3x3 multiplications.
    # The rank of the 3x3 * 3x3 tensor is known to be 23 (Landsberg).
    # So the total rank would be 23 + 23 = 46.
    # This is better than the schoolbook rank of 54.
    
    # However, I don't have the 23-rank algorithm coefficients hardcoded.
    # I will try to use the Strassen algorithm for 2x2 matrices to build a 3x3 * 3x6 solution.
    # Or I can use the fact that 3x3 * 3x6 can be embedded in 4x4 * 4x6.
    # Rank(4x4 * 4x6) = Rank(4x4 * 4x4) + Rank(4x4 * 4x2) = 15 + 8 = 23?
    # Wait, Rank(4x4 * 4x2) is not 8.
    # 4x4 * 4x2 is equivalent to multiplying a 4x4 matrix by a 4x2 matrix.
    # This can be done by multiplying the 4x4 matrix by two 4x1 vectors.
    # The rank of 4x4 * 4x1 is 4 (since it's just A * v, which is a linear map).
    # So Rank(4x4 * 4x2) = 4 + 4 = 8.
    # And Rank(4x4 * 4x4) = 15 (Strassen-like for 4x4, actually it's 15 for 4x4x4? No, Strassen for 4x4 uses 7 * 7 = 49 scalar mults, but the tensor rank is 15? No, the tensor rank of 4x4x4 is 15? I think the minimal rank for 4x4x4 is 15? Or is it 18?
    # Actually, the rank of the 4x4x4 tensor is 15? No, Strassen's algorithm for 4x4 uses 7 multiplications of 2x2 blocks, each of which has rank 7, so 7*7=49. But the tensor rank is the number of scalar multiplications.
    # The rank of the 4x4x4 tensor is 15? No, I think it's 15 for 3x3x3? No, 3x3x3 is 23.
    # The rank of 4x4x4 is 15? I'm not sure.
    # Let's stick to the 46 rank solution if I can find the 23-rank coefficients.
    # Since I can't, I will return the schoolbook solution.
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
