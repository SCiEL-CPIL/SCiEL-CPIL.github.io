"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin
    
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

    a_L = [0, 1, 4, 5]
    a_R = [2, 3, 6, 7]
    
    b_1_T = [0, 1, 6, 7]
    b_1_B = [12, 13, 18, 19]
    w_1 = [0, 2, 1, 3]
    
    add_strassen_block(a_L, b_1_T, w_1, 0)
    add_strassen_block(a_R, b_1_B, w_1, 7)
    
    b_2_T = [2, 3, 8, 9]
    b_2_B = [14, 15, 20, 21]
    w_2 = [4, 6, 5, 7]
    
    add_strassen_block(a_L, b_2_T, w_2, 14)
    add_strassen_block(a_R, b_2_B, w_2, 21)
    
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
    
    # Convert to float arrays
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build the tensor matrix
    flat_terms = []
    for r in range(rank):
        term = np.outer(np.outer(U_np[:, r], V_np[:, r]), W_np[:, r])
        flat_terms.append(term.flatten())
    M = np.array(flat_terms).T  # shape (2304, 42)
    
    # Best result so far
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank
    
    def to_list(arr):
        result = []
        for row in arr:
            result.append([])
            for val in row:
                frac = round(val * 2) / 2
                if abs(frac - round(frac)) < 1e-10:
                    frac = int(round(frac))
                result[-1].append(frac)
        return result
    
    def verify_float(final_U, final_V, final_W, new_rank, tol=1e-7):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(new_rank):
                        total += final_U[i*m+j, r] * final_V[j*p+k, r] * final_W[k*n+i, r]
                    if abs(total - 1.0) > tol:
                        return False
        return True
    
    # Strategy: Full Null Space Elimination
    # Compute SVD of M
    try:
        U_svd, s_svd, Vt_svd = np.linalg.svd(M, full_matrices=False)
    except np.linalg.LinAlgError:
        return best_result
    
    # Determine effective rank
    tol = 1e-9
    eff_rank = int(np.sum(s_svd > tol))
    
    if eff_rank < best_rank:
        # Null space is in the last (rank - eff_rank) rows of Vt_svd
        null_space = Vt_svd[eff_rank:, :]
        null_dim = null_space.shape[0]
        
        # Eliminate columns corresponding to the null space
        # We can eliminate 'null_dim' columns.
        # To maximize stability, we eliminate columns with largest norms in the null space?
        # Or just the last 'null_dim' columns if we reorder?
        # Better: find a set of columns that can be eliminated.
        # If we have null vector v, we can eliminate column r if v[r] != 0.
        # We can eliminate up to null_dim columns.
        
        # Simple approach: eliminate the last 'null_dim' columns.
        # This requires that the null space vectors have non-zero components in these columns.
        # We can permute columns to ensure this.
        
        # For now, let's try to eliminate columns greedily as before, but with the full null space knowledge.
        # The previous greedy method was fine, but let's ensure we use the full null space.
        
        # Actually, if we have the null space, we can just drop 'null_dim' columns and adjust the others.
        # Let's drop the last 'null_dim' columns.
        keep_cols = list(range(best_rank - null_dim))
        
        # Check if the null space vectors have significant weight in the dropped columns.
        # If not, we might need to drop different columns.
        # For simplicity, we assume the standard basis works or we permute.
        
        # Let's try to drop columns that have the largest projection onto the null space.
        # But dropping a column requires updating the others.
        # The update formula is: U_new[:, c] = U[:, c] - (v[c]/v[pivot]) * U[:, pivot]
        # This is what the greedy method did.
        
        # Let's stick to the greedy method but run it with the knowledge that we can eliminate 'null_dim' columns.
        
        new_U = U_np.copy()
        new_V = V_np.copy()
        new_W = W_np.copy()
        eliminated = set()
        
        for _ in range(null_dim):
            if time.time() > deadline - 5:
                break
            
            # Find the best column to eliminate
            best_col = None
            best_score = -1
            best_null_idx = 0
            
            for nv_idx in range(null_dim):
                nv = null_space[nv_idx]
                for c in range(best_rank):
                    if c in eliminated:
                        continue
                    score = abs(nv[c])
                    if score > best_score and score > 1e-8:
                        best_score = score
                        best_col = c
                        best_null_idx = nv_idx
            
            if best_col is None:
                break
            
            nv = null_space[best_null_idx]
            pivot_val = nv[best_col]
            
            for c in range(best_rank):
                if c == best_col or c in eliminated:
                    continue
                coeff = -nv[c] / pivot_val
                if abs(coeff) > 1e-10:
                    new_U[:, c] += coeff * new_U[:, best_col]
                    new_V[:, c] += coeff * new_V[:, best_col]
                    new_W[:, c] += coeff * new_W[:, best_col]
            
            eliminated.add(best_col)
        
        if eliminated:
            keep = [c for c in range(best_rank) if c not in eliminated]
            new_rank = len(keep)
            
            if verify_float(new_U[:, keep], new_V[:, keep], new_W[:, keep], new_rank):
                best_rank = new_rank
                best_result = {
                    "U": to_list(new_U[:, keep]),
                    "V": to_list(new_V[:, keep]),
                    "W": to_list(new_W[:, keep])
                }
    
    return best_result
# EVOLVE-BLOCK-END
