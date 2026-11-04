"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
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
    
    # Convert to float arrays
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build the tensor matrix: column r = flattened (u_r ⊗ v_r ⊗ w_r)
    flat_terms = []
    for r in range(rank):
        term = np.outer(np.outer(U_np[:, r], V_np[:, r]), W_np[:, r])
        flat_terms.append(term.flatten())
    M = np.array(flat_terms).T  # shape (2304, 42)
    
    # Best result so far
    best_rank = rank
    best_U = U_np.copy()
    best_V = V_np.copy()
    best_W = W_np.copy()
    
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
    
    def verify_float(Uf, Vf, Wf, rnk, tol=1e-7):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rnk):
                        total += Uf[i*m+j, r] * Vf[j*p+k, r] * Wf[k*n+i, r]
                    if abs(total - 1.0) > tol:
                        return False
        return True
    
    def eliminate_columns(U_base, V_base, W_base, M_base, max_eliminations, deadline, seed_offset=0):
        """Eliminate redundant columns using SVD null space analysis"""
        new_U = U_base.copy()
        new_V = V_base.copy()
        new_W = W_base.copy()
        eliminated = set()
        
        for iteration in range(max_eliminations):
            if time.time() > deadline:
                break
            
            remaining = [c for c in range(new_U.shape[1]) if c not in eliminated]
            if len(remaining) <= 1:
                break
            
            M_sub = M_base[:, remaining]
            
            try:
                U_sub, s_sub, Vt_sub = np.linalg.svd(M_sub, full_matrices=False)
            except np.linalg.LinAlgError:
                break
            
            tol_sub = 1e-10
            eff_rank_sub = int(np.sum(s_sub > tol_sub))
            
            if eff_rank_sub >= len(remaining):
                break
            
            null_sub = Vt_sub[eff_rank_sub:, :]
            
            # Pick best elimination column
            best_col = None
            best_score = -1
            best_null_idx = 0
            
            for nv_idx in range(null_sub.shape[0]):
                nv = null_sub[nv_idx]
                for local_c, global_c in enumerate(remaining):
                    score = abs(nv[local_c])
                    if score > best_score and score > 1e-8:
                        best_score = score
                        best_col = global_c
                        best_null_idx = nv_idx
            
            if best_col is None:
                break
            
            nv = null_sub[best_null_idx]
            local_best = remaining.index(best_col)
            pivot_val = nv[local_best]
            
            if abs(pivot_val) < 1e-10:
                continue
            
            for local_c, global_c in enumerate(remaining):
                if global_c == best_col:
                    continue
                coeff = -nv[local_c] / pivot_val
                if abs(coeff) > 1e-10:
                    new_U[:, global_c] += coeff * new_U[:, best_col]
                    new_V[:, global_c] += coeff * new_V[:, best_col]
                    new_W[:, global_c] += coeff * new_W[:, best_col]
            
            eliminated.add(best_col)
        
        if not eliminated:
            return None, None, None, 0
        
        keep = [c for c in range(new_U.shape[1]) if c not in eliminated]
        new_rank = len(keep)
        final_U = new_U[:, keep]
        final_V = new_V[:, keep]
        final_W = new_W[:, keep]
        
        return final_U, final_V, final_W, new_rank
    
    # Strategy 1: Multiple SVD-based elimination with different random seeds
    for trial in range(30):
        if time.time() > deadline - 10:
            break
        
        np.random.seed(42 + trial * 137)
        
        # Try different elimination strategies
        max_elim = rank - 38 + (trial % 5)  # Try eliminating 4 to 8 columns
        
        U_res, V_res, W_res, new_rank = eliminate_columns(
            U_np, V_np, W_np, M, max_elim, deadline - 10, seed_offset=trial
        )
        
        if U_res is not None and new_rank < best_rank:
            if verify_float(U_res, V_res, W_res, new_rank, tol=1e-7):
                best_rank = new_rank
                best_U = U_res
                best_V = V_res
                best_W = W_res
    
    # Strategy 2: More aggressive elimination with tighter tolerance
    if time.time() < deadline - 15 and best_rank > 40:
        for trial in range(20):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(1000 + trial * 211)
            
            max_elim = rank - 39 + (trial % 4)
            
            U_res, V_res, W_res, new_rank = eliminate_columns(
                U_np, V_np, W_np, M, max_elim, deadline - 10, seed_offset=trial + 100
            )
            
            if U_res is not None and new_rank < best_rank:
                if verify_float(U_res, V_res, W_res, new_rank, tol=1e-8):
                    best_rank = new_rank
                    best_U = U_res
                    best_V = V_res
                    best_W = W_res
    
    # Final polish: try to eliminate one more column if possible
    if time.time() < deadline - 5 and best_rank > 40:
        U_res, V_res, W_res, new_rank = eliminate_columns(
            best_U, best_V, best_W, M, 1, deadline - 3, seed_offset=999
        )
        
        if U_res is not None and new_rank < best_rank:
            if verify_float(U_res, V_res, W_res, new_rank, tol=1e-9):
                best_rank = new_rank
                best_U = U_res
                best_V = V_res
                best_W = W_res
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
