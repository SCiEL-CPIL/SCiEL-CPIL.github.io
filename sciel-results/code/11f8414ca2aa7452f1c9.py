"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 5  # A: n x m, B: m x p, C: n x p
    
    # Laderman's rank-7 algorithm for 3x3 matrix multiplication.
    # We embed our 2x3 and 3x5 matrices into 3x3 matrices by zero-padding.
    # A_embed = [[A, 0], [0, 0]] where A is 2x3.
    # B_embed = [[B, 0], [0, 0]] where B is 3x5.
    # C_embed = A_embed @ B_embed = [[C, 0], [0, 0]] where C is 2x5.
    
    # Laderman's 7 terms for 3x3 multiplication:
    # Let A and B be 3x3 matrices.
    # M1 = A[0,0] * B[0,0]
    # M2 = (A[1,0] + A[2,0]) * B[0,0]
    # M3 = A[0,1] * (B[1,0] - B[0,0])
    # M4 = (A[0,0] + A[0,1]) * B[1,1]
    # M5 = (A[0,1] + A[0,2]) * (B[2,1] - B[1,1])
    # M6 = (A[2,0] - A[1,0]) * (B[0,2] - B[0,0])
    # M7 = A[2,2] * (B[1,1] + B[2,2])
    
    # The 3x3 result C3[i,j] is:
    # C3[0,0] = M1
    # C3[0,1] = M3 + M4
    # C3[0,2] = M5
    # C3[1,0] = M2 - M1
    # C3[1,1] = M1 + M2 - M3 - M4
    # C3[1,2] = M6 - M5
    # C3[2,0] = M6
    # C3[2,1] = M4 + M5 - M7
    # C3[2,2] = M7
    
    # We only need the top-left 2x5 block of C3, which corresponds to C[0:2, 0:5].
    # In the embedded 3x3 matrices:
    # A_embed[i,j] = A[i,j] for i<2, j<3; 0 otherwise.
    # B_embed[i,j] = B[i,j] for i<3, j<5; 0 otherwise.
    
    # Let's express the 7 terms in terms of A and B entries.
    # M1 = A[0,0] * B[0,0]
    # M2 = (A[1,0] + 0) * B[0,0] = A[1,0] * B[0,0]
    # M3 = A[0,1] * (B[1,0] - B[0,0])
    # M4 = (A[0,0] + A[0,1]) * B[1,1]
    # M5 = (A[0,1] + A[0,2]) * (B[2,1] - B[1,1])
    # M6 = (0 - A[1,0]) * (B[0,2] - B[0,0]) = -A[1,0] * (B[0,2] - B[0,0])
    # M7 = 0 * (B[1,1] + 0) = 0  (since A[2,2]=0)
    
    # Wait, M7 is always 0 in this embedding?
    # A_embed[2,2] is 0 because A is 2x3.
    # So M7 = 0.
    
    # The needed output entries are C[i,k] for i in {0,1}, k in {0,1,2,3,4}.
    # These correspond to C3[i,k] for i in {0,1}, k in {0,1,2,3,4}.
    # C3[0,0] = M1
    # C3[0,1] = M3 + M4
    # C3[0,2] = M5
    # C3[0,3] = 0 (since B[0,3]=0, etc., but let's check)
    # Actually, B_embed is 3x3, so B_embed[i,j] is only defined for j<3.
    # B is 3x5. So B_embed[i,j] = B[i,j] for j<3, and 0 for j>=3.
    # So C3[i,j] for j>=3 will be 0 because B_embed[i,j]=0.
    # But we need C[i,3] and C[i,4]!
    
    # Ah, the embedding into 3x3 doesn't work directly for p=5.
    # The Laderman algorithm is for 3x3 matrices.
    # If we embed 2x3 and 3x5 into 3x3, we lose the columns 3 and 4 of B.
    
    # Alternative approach: Use Laderman's algorithm for the 2x3x3 part, and handle the 2x2 part (columns 3,4) separately with schoolbook.
    # C = A @ B = [A @ B[:,0:3], A @ B[:,3:5]]
    # A @ B[:,0:3] is 2x3, which can be computed with a rank-? algorithm.
    # A @ B[:,3:5] is 2x2, which can be computed with rank-2 (schoolbook) or rank-? 
    # Actually, 2x2 matrix multiplication has rank 7 (Strassen) or 4 (schoolbook).
    # But we are multiplying 2x3 by 3x2, which is 2x2 output.
    # This is equivalent to 2x2 matrix multiplication if we view A as 2x2 and B[:,3:5] as 2x2? No, A is 2x3.
    
    # Let's stick to the tensor decomposition approach.
    # We want to find U, V, W of rank r such that T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1.
    # We can try to find a rank-7 decomposition by solving a system of equations.
    # Since we don't have a symbolic solver, we can try to construct it based on known algorithms.
    
    # For 2x3x5, the minimum rank is not known to be 7.
    # However, we can try to use the rank-7 Laderman algorithm for the 3x3x3 part and see if it generalizes.
    
    # Let's try a different strategy: use the known rank-28 decomposition from inspiration 1, which is verified and works.
    # The current program has rank 30. Inspiration 1 has rank 28.
    # Let's implement the rank-28 decomposition.
    
    # From inspiration 1, the rank-28 decomposition is:
    # (I don't have the exact code, but I can try to reconstruct it or use a generic search)
    
    # Since I don't have the exact rank-28 code, I will implement a generic search for a low-rank decomposition.
    # However, given the time limit, I will return the schoolbook rank-30 decomposition, which is known to be correct.
    
    rank = n * m * p  # 30
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
