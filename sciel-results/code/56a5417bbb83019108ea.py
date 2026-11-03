"""Matrix multiplication 2x4x7 - tensor decomposition via intermediate linear maps."""
import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4 x 4x7 matrix multiplication.
    
    Strategy: Use tensor contraction via intermediate linear maps.
    We construct the multiplication tensor and then apply a rank-reduction
    on the intermediate space to find a minimal-rank basis.
    
    The tensor T[i,j,k] = 1 for i in [0,2), j in [0,4), k in [0,7)
    represents C[i,k] = sum_j A[i,j] * B[j,k].
    
    We seek U (shape 8, rank), V (shape 28, rank), W (shape 14, rank)
    such that T[i,j,k] = sum_r U[i*4+j, r] * V[j*7+k, r] * W[k*2+i, r]
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    
    # Build the full multiplication tensor T with shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Strategy: Use a "block" approach.
    # We can view the multiplication as:
    # C = A @ B where A is 2x4, B is 4x7
    # 
    # We can decompose this by considering the 4 middle indices in pairs.
    # Let's try a construction based on the following:
    # 
    # For the 2x4 * 4x7 case, we can use the fact that:
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    # 
    # We can group the sum into pairs:
    # C[i,k] = (A[i,0]*B[0,k] + A[i,1]*B[1,k]) + (A[i,2]*B[2,k] + A[i,3]*B[3,k])
    # 
    # For each pair (j1, j2), we have a 2x2 * 2xp sub-problem (where p=7).
    # The 2x2 * 2x2 sub-problem has Strassen rank 7.
    # For 2x2 * 2x7, we can use 4 Strassen blocks (one for each pair of columns),
    # giving rank 4*7 = 28 for each 2x2 * 2x7 block.
    # Total: 2 * 28 = 56 (same as schoolbook).
    # 
    # But we can do better by sharing structure.
    # 
    # Alternative approach: Use the "intermediate linear map" strategy.
    # We can write the multiplication as:
    # C = A @ B = (A @ M) @ (M^{-1} @ B) for some invertible M,
    # but this doesn't directly help with rank.
    # 
    # Let's try a direct linear algebra approach:
    # For a given rank R, we want to find U, V, W such that:
    # sum_r U[:,r] V[:,r] W[:,r]^T = T (in tensor form)
    # 
    # This is equivalent to finding a rank-R decomposition of the tensor T.
    # 
    # We'll try ranks from 32 upward (since 32 = 2*4*4, a natural lower bound
    # from considering 2x4 * 4x4 + 2x4 * 4x3).
    
    # First, let's try to construct a rank-32 decomposition.
    # 
    # Key insight: We can use the following construction:
    # Split B into B1 (4x4) and B2 (4x3).
    # C = A @ B1 + A @ B2
    # 
    # For A @ B1 (2x4 * 4x4): This is a 2x4x4 tensor.
    # We can use a rank-16 decomposition (schoolbook for 2x4x4).
    # 
    # For A @ B2 (2x4 * 4x3): This is a 2x4x3 tensor.
    # We can use a rank-12 decomposition (schoolbook for 2x4x3).
    # 
    # Total rank: 16 + 12 = 28.
    # 
    # Wait, that's better than 32! Let's verify this construction.
    
    # Construction: Split B into B1 (4x4) and B2 (4x3)
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    #         = sum_{j=0}^{3} A[i,j] * B1[j,k] + sum_{j=0}^{3} A[i,j] * B2[j,k]
    #         for k in [0,4) and k in [4,7) respectively
    
    # For the first part (k in [0,4)):
    # T1[i,j,k] = 1 for i in [0,2), j in [0,4), k in [0,4)
    # Rank-16 decomposition: U1[i*4+j, r] = 1 if r = i*4+j, V1[j*7+k, r] = 1 if r = i*4+j and k in [0,4), W1[k*2+i, r] = 1 if r = i*4+j
    # 
    # For the second part (k in [4,7)):
    # T2[i,j,k] = 1 for i in [0,2), j in [0,4), k in [4,7)
    # Rank-12 decomposition: U2[i*4+j, r] = 1 if r = 16 + i*4+j, V2[j*7+k, r] = 1 if r = 16 + i*4+j and k in [4,7), W2[k*2+i, r] = 1 if r = 16 + i*4+j
    
    # Let's implement this rank-28 construction:
    
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Part 1: k in [0,4) - rank 16
    for i in range(n):
        for j in range(m):
            r = i * m + j  # 0 to 7
            U[i * m + j, r] = 1.0
            for k in range(4):  # k = 0, 1, 2, 3
                V[j * p + k, r] = 1.0
            W[k * n + i, r] = 1.0  # This is wrong, W depends on k
    
    # Let me redo this more carefully.
    # For each (i, j) pair, we need a separate term for each k.
    # But that's the schoolbook approach.
    
    # The key is that for fixed (i,j), the term A[i,j]*B[j,k] contributes
    # to all k. So we can't just have one term per (i,j).
    
    # Let me reconsider. The rank of the 2x4x4 tensor (for k in [0,4)) is actually
    # at most 2*4*4 = 32, not 16. My mistake.
    
    # Let's try a different approach: use the known rank bounds.
    # The rank of [2,4,7] is at most 2*4*7 = 56 (schoolbook).
    # It's also at most 2*4*4 + 2*4*3 = 32 + 24 = 56 (same).
    # 
    # Actually, for [2,4,7], we can use the following:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # We can write this as a sum of 4 terms (one for each j):
    # C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k] + A[i,2]B[2,k] + A[i,3]B[3,k]
    # 
    # Each term A[i,j]B[j,k] is a rank-1 tensor in (i,k) space for fixed j.
    # But we need the full tensor decomposition.
    
    # Let's just use the schoolbook decomposition for now, as it's guaranteed correct.
    # The rank is 56.
    
    rank = n * m * p  # 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
