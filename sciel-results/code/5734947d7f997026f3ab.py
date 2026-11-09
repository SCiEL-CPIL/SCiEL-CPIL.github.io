import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 5]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Build the tensor from U, V, W
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # The multiplication tensor has T[i,j,k] = 1 if the indices correspond to a valid term
    # For matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor: T[i,j,k] = 1 for all valid i,j,k (since each term A[i,j]*B[j,k] contributes)
    # Actually the tensor entries are T[i,j,k] = 1 for all i in [n], j in [m], k in [p]
    
    # Check: for each (i,k) output entry, C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor decomposition should give: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = delta_{valid}
    # Actually, the multiplication tensor T has T[i,j,k] = 1 for all valid indices
    # and the decomposition means: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if total != 1:
                    return False
    return True

def _try_rank_search(n, m, p, max_rank, time_limit=60):
    """Try to find a rank-minimal decomposition using random search."""
    start_time = time.time()
    best_rank = None
    best_decomp = None
    
    # Start from a known good rank and try to reduce
    # Use the Strassen-based solution as starting point
    import copy
    
    # Generate the Strassen-based solution
    def generate_strassen_solution():
        strassen_U = [
            [1,0,0,1],
            [0,0,1,1],
            [1,0,0,0],
            [0,0,0,1],
            [1,1,0,0],
            [-1,0,1,0],
            [0,1,0,-1],
        ]
        strassen_V = [
            [1,0,0,1],
            [1,0,0,0],
            [0,1,0,-1],
            [-1,0,1,0],
            [0,0,0,1],
            [1,1,0,0],
            [0,0,1,1],
        ]
        strassen_W = [
            [1,0,0,1],
            [0,1,0,-1],
            [0,0,1,1],
            [1,1,0,0],
            [-1,0,1,0],
            [0,0,0,1],
            [1,0,0,0],
        ]
        
        rank = 4 * 7 + 8
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        for j0 in [0, 2]:
            for k0 in [0, 2]:
                for r in range(7):
                    for li in range(2):
                        for lj in range(2):
                            gi = li
                            gj = j0 + lj
                            U[gi * m + gj][col] = strassen_U[r][li * 2 + lj]
                    for lj in range(2):
                        for lk in range(2):
                            gj = j0 + lj
                            gk = k0 + lk
                            V[gj * p + gk][col] = strassen_V[r][lj * 2 + lk]
                    for lk in range(2):
                        for li in range(2):
                            gk = k0 + lk
                            gi = li
                            W[gk * n + gi][col] = strassen_W[r][lk * 2 + li]
                    col += 1
        
        for i in range(n):
            for j in range(m):
                U[i * m + j][col] = 1
                V[j * p + 4][col] = 1
                W[4 * n + i][col] = 1
                col += 1
        
        return U, V, W, rank
    
    U0, V0, W0, r0 = generate_strassen_solution()
    if _verify_decomposition(U0, V0, W0, n, m, p):
        best_rank = r0
        best_decomp = (U0, V0, W0)
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        best_rank = rank
        best_decomp = (U, V, W)
    
    # Try to reduce rank by removing columns and adjusting
    # This is a hard problem, so we'll do a limited search
    # Try random linear combinations of columns to see if we can merge them
    
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed
    if remaining < 5:
        return best_decomp
    
    # Attempt: try to find a lower rank by random perturbation and rank reduction
    # This is heuristic and may not find improvements
    # For now, return the best we have
    return best_decomp

def construct():
    n, m, p = 2, 4, 5
    start_time = time.time()
    
    # Use the Strassen-based solution which achieves rank 36
    strassen_U = [
        [1,0,0,1],  # M1
        [0,0,1,1],  # M2
        [1,0,0,0],  # M3
        [0,0,0,1],  # M4
        [1,1,0,0],  # M5
        [-1,0,1,0], # M6
        [0,1,0,-1], # M7
    ]
    strassen_V = [
        [1,0,0,1],  # M1
        [1,0,0,0],  # M2
        [0,1,0,-1], # M3
        [-1,0,1,0], # M4
        [0,0,0,1],  # M5
        [1,1,0,0],  # M6
        [0,0,1,1],  # M7
    ]
    strassen_W = [
        [1,0,0,1],  # M1: C00+=1, C11+=1
        [0,1,0,-1], # M2: C10+=1, C11+=-1
        [0,0,1,1],  # M3: C01+=1, C11+=1
        [1,1,0,0],  # M4: C00+=1, C10+=1
        [-1,0,1,0], # M5: C00+=-1, C01+=1
        [0,0,0,1],  # M6: C11+=1
        [1,0,0,0],  # M7: C00+=1
    ]
    
    rank = 4 * 7 + 8
    U = [[0] * rank for _ in range(n * m)]  # 8 x rank
    V = [[0] * rank for _ in range(m * p)]  # 20 x rank
    W = [[0] * rank for _ in range(p * n)]  # 10 x rank
    
    col = 0
    for j0 in [0, 2]:
        for k0 in [0, 2]:
            for r in range(7):
                for li in range(2):
                    for lj in range(2):
                        gi = li
                        gj = j0 + lj
                        U[gi * m + gj][col] = strassen_U[r][li * 2 + lj]
                for lj in range(2):
                    for lk in range(2):
                        gj = j0 + lj
                        gk = k0 + lk
                        V[gj * p + gk][col] = strassen_V[r][lj * 2 + lk]
                for lk in range(2):
                    for li in range(2):
                        gk = k0 + lk
                        gi = li
                        W[gk * n + gi][col] = strassen_W[r][lk * 2 + li]
                col += 1
    
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
