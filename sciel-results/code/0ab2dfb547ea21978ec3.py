"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

import time
import random
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Helper to validate a decomposition
    def is_valid_decomposition(U, V, W, rank):
        U_mat = np.array(U)
        V_mat = np.array(V)
        W_mat = np.array(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_mat[i*m+j, :]
                    v_val = V_mat[j*p+k, :]
                    w_val = W_mat[k*n+i, :]
                    total = np.sum(u_val * v_val * w_val)
                    if not np.isclose(total, 1.0):
                        return False
        return True

    # 1. Establish Baseline (Schoolbook)
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
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb
    
    start_time = time.time()
    time_limit = 110.0
    
    # Strategy: Try to find a decomposition with rank < 60 using a structured approach.
    # For 2x5x6, we can think of it as two 1x5x6 multiplications.
    # A key insight: if we can share some "computation" between the two rows of A,
    # we might reduce the rank.
    
    # Let's try a systematic search with a different approach:
    # Start from the schoolbook and try to merge columns that have similar patterns.
    
    # Alternative: Try to find a rank-58 or rank-59 decomposition by searching
    # for a valid assignment of values in {-1, 0, 1} with a smarter heuristic.
    
    # For each target rank r from rank_sb-1 down to a minimum, try random searches.
    # But random search over {-1,0,1}^r for r~59 is hopeless.
    
    # Better approach: Use the fact that the problem is small enough that we can
    # try to find a sparse solution with a greedy or local search method.
    
    # Let's try: Start with schoolbook, then try to "merge" pairs of columns.
    # If two columns c1 and c2 have U, V, W entries such that we can find
    # a single column that replaces both, we reduce rank by 1.
    
    # Actually, a cleaner approach: try to find a rank r decomposition by
    # solving a system of equations. For each (i,j,k), we need:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a system of n*m*p = 60 equations in r variables per matrix.
    # For r = 59, we have 59*10 + 59*30 + 59*12 = 590 + 1770 + 708 = 3068 unknowns
    # but only 60 equations. So there's a lot of freedom.
    
    # Let's try a random search with a bias toward sparse solutions.
    
    found_better = False
    min_search_rank = max(1, rank_sb - 5)  # Only search a small window
    
    for r in range(rank_sb - 1, min_search_rank - 1, -1):
        if time.time() - start_time > time_limit:
            break
            
        attempts = 50
        for attempt in range(attempts):
            if time.time() - start_time > time_limit:
                break
            
            # Generate random matrices with values in {-1, 0, 1}
            # but with a higher probability of 0 to encourage sparsity
            def gen_sparse_matrix(rows, cols):
                mat = [[0] * cols for _ in range(rows)]
                for i in range(rows):
                    for j in range(cols):
                        rand_val = random.random()
                        if rand_val < 0.3:
                            mat[i][j] = random.choice([-1, 1])
                        elif rand_val < 0.5:
                            mat[i][j] = random.choice([-1, 1])
                        # else stays 0
                return mat
            
            U_rand = gen_sparse_matrix(n * m, r)
            V_rand = gen_sparse_matrix(m * p, r)
            W_rand = gen_sparse_matrix(p * n, r)
            
            if is_valid_decomposition(U_rand, V_rand, W_rand, r):
                best_U, best_V, best_W, best_rank = U_rand, V_rand, W_rand, r
                found_better = True
                break
        
        if found_better:
            break
        if time.time() - start_time > time_limit:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
