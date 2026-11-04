"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Rank 42 decomposition based on Strassen's algorithm applied to 2x2 blocks
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a Strassen 2x2x2 decomposition
    def add_strassen_block(a_idx, b_idx, w_idx, start_col):
        U[a_idx[0]][start_col + 0] += 1
        U[a_idx[3]][start_col + 0] += 1
        U[a_idx[2]][start_col + 1] += 1
        U[a_idx[3]][start_col + 1] += 1
        U[a_idx[0]][start_col + 2] += 1
        U[a_idx[3]][start_col + 3] += 1
        U[a_idx[0]][start_col + 4] += 1
        U[a_idx[1]][start_col + 4] += 1
        U[a_idx[2]][start_col + 5] += 1
        U[a_idx[0]][start_col + 5] -= 1
        U[a_idx[1]][start_col + 6] += 1
        U[a_idx[3]][start_col + 6] -= 1
        
        V[b_idx[0]][start_col + 0] += 1
        V[b_idx[3]][start_col + 0] += 1
        V[b_idx[0]][start_col + 1] += 1
        V[b_idx[1]][start_col + 2] += 1
        V[b_idx[3]][start_col + 2] -= 1
        V[b_idx[2]][start_col + 3] += 1
        V[b_idx[0]][start_col + 3] -= 1
        V[b_idx[3]][start_col + 4] += 1
        V[b_idx[0]][start_col + 5] += 1
        V[b_idx[1]][start_col + 5] += 1
        V[b_idx[2]][start_col + 6] += 1
        V[b_idx[3]][start_col + 6] += 1
        
        W[w_idx[0]][start_col + 0] += 1
        W[w_idx[0]][start_col + 3] += 1
        W[w_idx[0]][start_col + 4] -= 1
        W[w_idx[0]][start_col + 6] += 1
        
        W[w_idx[1]][start_col + 2] += 1
        W[w_idx[1]][start_col + 4] += 1
        
        W[w_idx[2]][start_col + 1] += 1
        W[w_idx[2]][start_col + 3] += 1
        
        W[w_idx[3]][start_col + 0] += 1
        W[w_idx[3]][start_col + 1] -= 1
        W[w_idx[3]][start_col + 2] += 1
        W[w_idx[3]][start_col + 5] += 1

    a_L = [0, 1, 4, 5]
    a_R = [2, 3, 6, 7]
    
    b_1_T = [0, 1, 6, 7]
    b_1_B = [12, 13, 18, 19]
    w_1 = [0, 2, 1, 3]
    
    b_2_T = [2, 3, 8, 9]
    b_2_B = [14, 15, 20, 21]
    w_2 = [4, 6, 5, 7]
    
    b_3_T = [4, 5, 10, 11]
    b_3_B = [16, 17, 22, 23]
    w_3 = [8, 10, 9, 11]
    
    add_strassen_block(a_L, b_1_T, w_1, 0)
    add_strassen_block(a_R, b_1_B, w_1, 7)
    
    add_strassen_block(a_L, b_2_T, w_2, 14)
    add_strassen_block(a_R, b_2_B, w_2, 21)
    
    add_strassen_block(a_L, b_3_T, w_3, 28)
    add_strassen_block(a_R, b_3_B, w_3, 35)
    
    # Verify the decomposition
    def verify(U_mat, V_mat, W_mat, n, m, p):
        rank = len(U_mat[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U_mat[i * m + j][r] * V_mat[j * p + k][r] * W_mat[k * n + i][r]
                    if total != 1:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook
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
    
    # Precompute the tensor values for fast verification
    # tensor[i,j,k] should be 1 if j==k else 0 (but actually it's always 1 for valid indices)
    # The decomposition must satisfy: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k
    
    def verify_fast(U_mat, V_mat, W_mat, n, m, p, remove_set=None):
        """Fast verification using precomputed structure."""
        rank = len(U_mat[0])
        if remove_set is None:
            remove_set = set()
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        if r in remove_set:
                            continue
                        total += U_mat[i * m + j][r] * V_mat[j * p + k][r] * W_mat[k * n + i][r]
                    if total != 1:
                        return False
        return True
    
    # Try removing single factors
    best_result = None
    for r in range(rank):
        if time.time() > deadline:
            break
        if verify_fast(U, V, W, n, m, p, {r}):
            # Remove factor r
            indices_to_keep = [i for i in range(rank) if i != r]
            U_new = [[row[i] for i in indices_to_keep] for row in U]
            V_new = [[row[i] for i in indices_to_keep] for row in V]
            W_new = [[row[i] for i in indices_to_keep] for row in W]
            return {"U": U_new, "V": V_new, "W": W_new}
    
    # Try removing pairs of factors - more efficient using numpy
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Precompute the contribution of each factor to the tensor
    # contrib[r] is a tensor of shape (n, m, p)
    n_mp = n * m * p
    contrib = np.zeros((rank, n_mp), dtype=float)
    for r in range(rank):
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    contrib[r, idx] = U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                    idx += 1
    
    # The sum of all contributions should be all 1s
    # Check: sum_r contrib[r] = 1 for all entries
    # If we remove factors R, we need sum_{r not in R} contrib[r] = 1
    # i.e., sum_{r in R} contrib[r] = 0
    
    # For single removal: contrib[r] must be 0 for all entries
    # For double removal: contrib[r1] + contrib[r2] must be 0
    
    # Check single removals via contrib
    for r in range(rank):
        if time.time() > deadline:
            break
        if np.allclose(contrib[r], 0, atol=1e-10):
            indices_to_keep = [i for i in range(rank) if i != r]
            U_new = [[row[i] for i in indices_to_keep] for row in U]
            V_new = [[row[i] for i in indices_to_keep] for row in V]
            W_new = [[row[i] for i in indices_to_keep] for row in W]
            return {"U": U_new, "V": V_new, "W": W_new}
    
    # Check double removals: contrib[r1] + contrib[r2] = 0
    if time.time() < deadline - 20:
        for r1 in range(rank):
            if time.time() > deadline - 5:
                break
            for r2 in range(r1 + 1, rank):
                if np.allclose(contrib[r1] + contrib[r2], 0, atol=1e-10):
                    indices_to_keep = [i for i in range(rank) if i not in (r1, r2)]
                    U_new = [[row[i] for i in indices_to_keep] for row in U]
                    V_new = [[row[i] for i in indices_to_keep] for row in V]
                    W_new = [[row[i] for i in indices_to_keep] for row in W]
                    return {"U": U_new, "V": V_new, "W": W_new}
    
    # Try triple removals
    if time.time() < deadline - 10:
        for r1 in range(rank):
            if time.time() > deadline - 5:
                break
            for r2 in range(r1 + 1, rank):
                if time.time() > deadline - 5:
                    break
                for r3 in range(r2 + 1, rank):
                    if np.allclose(contrib[r1] + contrib[r2] + contrib[r3], 0, atol=1e-10):
                        indices_to_keep = [i for i in range(rank) if i not in (r1, r2, r3)]
                        U_new = [[row[i] for i in indices_to_keep] for row in U]
                        V_new = [[row[i] for i in indices_to_keep] for row in V]
                        W_new = [[row[i] for i in indices_to_keep] for row in W]
                        return {"U": U_new, "V": V_new, "W": W_new}
    
    # If no simple removal works, try to find a better decomposition
    # by looking at the structure more carefully
    
    # The key insight: since n=2, the W matrix has only 2 rows per k.
    # We can try to find a rank-40 decomposition by exploiting this.
    
    # Let's try a different approach: use the fact that the problem is
    # 2x4x6, and try to find a decomposition of rank 40 or less.
    
    # One approach: use the "laser method" or "Coppersmith-Winograd" style
    # but that's complex. Let's try a more direct approach.
    
    # Actually, let's try to find if there's a rank-41 decomposition by
    # checking if any factor can be expressed as a combination of others
    # with appropriate scaling.
    
    # For now, return the rank-42 decomposition
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
