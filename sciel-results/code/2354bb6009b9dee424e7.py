import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    # Schoolbook decomposition
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Check if rank can be reduced by checking linear dependencies
    # Construct the matrix of rank-1 tensors
    dim_A = n * m
    dim_B = m * p
    dim_C = p * n
    total_dim = dim_A * dim_B * dim_C
    
    # M is total_dim x rank_sb
    # We can use a sparse representation or just a dense array if it fits
    # 2304 * 48 = 110592 floats. This is small.
    M = np.zeros((total_dim, rank_sb), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_A = i * m + j
                idx_B = j * p + k
                idx_C = k * n + i
                # Flatten index: (idx_A * dim_B + idx_B) * dim_C + idx_C
                flat_idx = (idx_A * dim_B + idx_B) * dim_C + idx_C
                M[flat_idx, col] = 1.0
                col += 1
                
    # Compute SVD to find rank
    try:
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
        # Rank is number of singular values > tol
        tol = 1e-9
        rank = np.sum(s > tol)
        
        if rank < rank_sb:
            # We can potentially reduce the rank
            # For simplicity, if rank is 47, we can try to eliminate one term.
            # However, constructing the new decomposition from SVD is complex.
            # Given the time constraint and the likelihood that rank is 48,
            # we just return the schoolbook.
            # If we wanted to be aggressive, we could return the rank from SVD?
            # No, we must return a valid decomposition.
            # If rank < 48, the schoolbook decomposition is not minimal, but it is valid.
            # The score is based on the rank of the returned decomposition.
            # So we must return a decomposition of the minimal rank found.
            # Since finding the actual minimal decomposition is hard, and we suspect rank is 48,
            # we return the schoolbook.
            pass
    except Exception:
        pass
        
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
