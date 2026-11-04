"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s kill
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Build the rank-42 Strassen decomposition
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    def add_strassen_block(a_idx, b_idx, w_idx, start_col):
        c = start_col
        # M1 = (a00+a11)(b00+b11)
        U[a_idx[0]][c] += 1; U[a_idx[3]][c] += 1
        V[b_idx[0]][c] += 1; V[b_idx[3]][c] += 1
        W[w_idx[0]][c] += 1; W[w_idx[3]][c] += 1
        # M2 = (a10+a11)b00
        U[a_idx[2]][c+1] += 1; U[a_idx[3]][c+1] += 1
        V[b_idx[0]][c+1] += 1
        W[w_idx[2]][c+1] += 1; W[w_idx[3]][c+1] -= 1
        # M3 = a00(b01-b11)
        U[a_idx[0]][c+2] += 1
        V[b_idx[1]][c+2] += 1; V[b_idx[3]][c+2] -= 1
        W[w_idx[1]][c+2] += 1; W[w_idx[3]][c+2] += 1
        # M4 = a11(b10-b00)
        U[a_idx[3]][c+3] += 1
        V[b_idx[2]][c+3] += 1; V[b_idx[0]][c+3] -= 1
        W[w_idx[0]][c+3] += 1; W[w_idx[2]][c+3] += 1
        # M5 = (a00+a01)b11
        U[a_idx[0]][c+4] += 1; U[a_idx[1]][c+4] += 1
        V[b_idx[3]][c+4] += 1
        W[w_idx[0]][c+4] -= 1; W[w_idx[1]][c+4] += 1
        # M6 = (a10-a00)(b00+b01)
        U[a_idx[2]][c+5] += 1; U[a_idx[0]][c+5] -= 1
        V[b_idx[0]][c+5] += 1; V[b_idx[1]][c+5] += 1
        W[w_idx[3]][c+5] += 1
        # M7 = (a01-a11)(b10+b11)
        U[a_idx[1]][c+6] += 1; U[a_idx[3]][c+6] -= 1
        V[b_idx[2]][c+6] += 1; V[b_idx[3]][c+6] += 1
        W[w_idx[0]][c+6] += 1

    # A is 2x4: indices i*m+j
    a_L = [0, 1, 4, 5]
    a_R = [2, 3, 6, 7]
    
    # B is 4x6: indices j*p+k
    # Block 1: B cols 0,1
    b_1_T = [0, 1, 6, 7]
    b_1_B = [12, 13, 18, 19]
    w_1 = [0, 2, 1, 3]
    
    add_strassen_block(a_L, b_1_T, w_1, 0)
    add_strassen_block(a_R, b_1_B, w_1, 7)
    
    # Block 2: B cols 2,3
    b_2_T = [2, 3, 8, 9]
    b_2_B = [14, 15, 20, 21]
    w_2 = [4, 6, 5, 7]
    
    add_strassen_block(a_L, b_2_T, w_2, 14)
    add_strassen_block(a_R, b_2_B, w_2, 21)
    
    # Block 3: B cols 4,5
    b_3_T = [4, 5, 10, 11]
    b_3_B = [16, 17, 22, 23]
    w_3 = [8, 10, 9, 11]
    
    add_strassen_block(a_L, b_3_T, w_3, 28)
    add_strassen_block(a_R, b_3_B, w_3, 35)
    
    # Verify the decomposition
    def verify_decomp(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if total != 1:
                        return False
        return True
    
    if not verify_decomp(U, V, W, rank):
        # Fallback: return schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i*m+j][r] = 1
                    V[j*p+k][r] = 1
                    W[k*n+i][r] = 1
        return {"U": U, "V": V, "W": W}
    
    # Convert to numpy for linear algebra
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Try to reduce rank by finding linear dependencies
    # Check if the set of rank-1 tensors {u_r ⊗ v_r ⊗ w_r} is linearly independent
    rank1_tensors = []
    for r in range(rank):
        t = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    t[i*m+j, j*p+k, k*n+i] = U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
        rank1_tensors.append(t.flatten())
    
    rank1_array = np.array(rank1_tensors)
    
    # Check rank of the matrix whose rows are the rank-1 tensors
    matrix_rank = np.linalg.matrix_rank(rank1_array, tol=1e-10)
    
    if matrix_rank < rank:
        # There are linear dependencies. Try to find and remove redundant factors.
        # For each factor, check if it can be expressed as a linear combination of others.
        # This is a complex problem, so we try a heuristic: remove factors one at a time
        # and check if the decomposition still holds.
        
        def try_remove_factors(U_list, V_list, W_list, factors_to_remove, n, m, p):
            """Try removing the specified factors and see if the decomposition still works."""
            rank_curr = len(U_list[0])
            indices_to_keep = [i for i in range(rank_curr) if i not in factors_to_remove]
            new_rank = len(indices_to_keep)
            
            if new_rank <= 0:
                return None
            
            U_new = [row[i] for row in U_list for i in indices_to_keep]
            # Reshape back
            U_new = [[U_list[i][idx] for idx in indices_to_keep] for i in range(len(U_list))]
            V_new = [[V_list[i][idx] for idx in indices_to_keep] for i in range(len(V_list))]
            W_new = [[W_list[i][idx] for idx in indices_to_keep] for i in range(len(W_list))]
            
            # Verify
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = 0
                        for r in range(new_rank):
                            total += U_new[i*m+j][r] * V_new[j*p+k][r] * W_new[k*n+i][r]
                        if total != 1:
                            return None
            return {"U": U_new, "V": V_new, "W": W_new}
        
        # Try removing each single factor
        best_result = None
        for r in range(rank):
            if time.time() > deadline - 10:
                break
            result = try_remove_factors(U, V, W, [r], n, m, p)
            if result is not None:
                best_result = result
                break
        
        if best_result is not None:
            return best_result
        
        # Try removing pairs of factors (only if we have time)
        if time.time() < deadline - 30:
            for r1 in range(rank):
                for r2 in range(r1 + 1, rank):
                    if time.time() > deadline - 10:
                        break
                    result = try_remove_factors(U, V, W, [r1, r2], n, m, p)
                    if result is not None:
                        return result
        
        # If no simple removal works, return the original rank-42 decomposition
        return {"U": U, "V": W, "W": W} if False else {"U": U, "V": V, "W": W}
    
    # Return the rank-42 decomposition
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
