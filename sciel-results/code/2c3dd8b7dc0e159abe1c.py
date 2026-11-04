"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit
    
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
    
    def verify_fast(Uf, Vf, Wf, rnk, tol=1e-7):
        # Vectorized verification
        U_t = Uf.T.reshape(n, m, rnk) # (n, m, r)
        V_t = Vf.T.reshape(m, p, rnk) # (m, p, r)
        W_t = Wf.T.reshape(p, n, rnk) # (p, n, r)
        
        # Result[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
        result = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        
        return np.allclose(result, 1.0, atol=tol)
    
    def try_round_and_verify(Uf, Vf, Wf, rnk):
        """Round to nearest half-integers and verify. Return (rounded_U, rounded_V, rounded_W, success)."""
        U_r = np.round(Uf * 2) / 2
        V_r = np.round(Vf * 2) / 2
        W_r = np.round(Wf * 2) / 2
        
        if verify_fast(U_r, V_r, W_r, rnk, tol=1e-9):
            return U_r, V_r, W_r, True
        return None, None, None, False
    
    def try_round_int_and_verify(Uf, Vf, Wf, rnk):
        """Try rounding to nearest integers and verify. Return (rounded_U, rounded_V, rounded_W, success)."""
        U_r = np.round(Uf)
        V_r = np.round(Vf)
        W_r = np.round(Wf)
        
        if verify_fast(U_r, V_r, W_r, rnk, tol=1e-9):
            return U_r, V_r, W_r, True
        return None, None, None, False
    
    def try_round_quarter_and_verify(Uf, Vf, Wf, rnk):
        """Try rounding to nearest quarter-integers and verify. Return (rounded_U, rounded_V, rounded_W, success)."""
        U_r = np.round(Uf * 4) / 4
        V_r = np.round(Vf * 4) / 4
        W_r = np.round(Wf * 4) / 4
        
        if verify_fast(U_r, V_r, W_r, rnk, tol=1e-9):
            return U_r, V_r, W_r, True
        return None, None, None, False
    
    def eliminate_columns(U_base, V_base, W_base, max_eliminations, deadline, seed_offset=0):
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
            
            # Rebuild M_sub from current U, V, W
            M_sub_cols = []
            for c in remaining:
                term = np.outer(np.outer(new_U[:, c], new_V[:, c]), new_W[:, c])
                M_sub_cols.append(term.flatten())
            M_sub = np.array(M_sub_cols).T
            
            try:
                U_sub, s_sub, Vt_sub = np.linalg.svd(M_sub, full_matrices=False)
            except np.linalg.LinAlgError:
                break
            
            tol_sub = 1e-10
            eff_rank_sub = int(np.sum(s_sub > tol_sub))
            
            if eff_rank_sub >= len(remaining):
                break
            
            null_sub = Vt_sub[eff_rank_sub:, :]
            
            best_col = None
            best_score = -1
            best_null_idx = 0
            
            for nv_idx in range(null_sub.shape[0]):
                nv = null_sub[nv_idx]
                for local_c, global_c in enumerate(remaining):
                    score = abs(nv[local_c])
                    if score > best_score and score > 1e-6:
                        best_score = score
                        best_col = global_c
                        best_null_idx = nv_idx
            
            if best_col is None:
                break
            
            nv = null_sub[best_null_idx]
            local_best = remaining.index(best_col)
            pivot_val = nv[local_best]
            
            if abs(pivot_val) < 1e-8:
                continue
            
            for local_c, global_c in enumerate(remaining):
                if global_c == best_col:
                    continue
                coeff = -nv[local_c] / pivot_val
                if abs(coeff) > 1e-8:
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
    
    def refine_with_random_perturbation(U_base, V_base, W_base, max_iter, deadline):
        """Apply small random perturbations and check if we can eliminate more columns."""
        U_cur = U_base.copy()
        V_cur = V_base.copy()
        W_cur = W_base.copy()
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            # Small random perturbation
            eps = 0.01
            U_pert = U_cur + eps * np.random.randn(*U_cur.shape)
            V_pert = V_cur + eps * np.random.randn(*V_cur.shape)
            W_pert = W_cur + eps * np.random.randn(*W_cur.shape)
            
            # Try elimination from perturbed state
            U_res, V_res, W_res, new_rank = eliminate_columns(
                U_pert, V_pert, W_pert, U_cur.shape[1] - 34, deadline
            )
            
            if U_res is not None and new_rank < U_cur.shape[1]:
                U_cur, V_cur, W_cur = U_res, V_res, W_res
        
        return U_cur, V_cur, W_cur
    
    # Strategy 1: SVD-based elimination with rounding
    for trial in range(80):
        if time.time() > deadline - 20:
            break
        
        np.random.seed(42 + trial * 137)
        
        max_elim = rank - 33 + (trial % 9)  # Try eliminating 9 to 17 columns
        
        U_res, V_res, W_res, new_rank = eliminate_columns(
            U_np, V_np, W_np, max_elim, deadline - 20, seed_offset=trial
        )
        
        if U_res is not None and new_rank < best_rank:
            # Try rounding to half-integers
            U_r, V_r, W_r, success = try_round_and_verify(U_res, V_res, W_res, new_rank)
            if success:
                best_rank = new_rank
                best_U = U_r
                best_V = V_r
                best_W = W_r
                continue
            
            # Try rounding to integers
            U_r, V_r, W_r, success = try_round_int_and_verify(U_res, V_res, W_res, new_rank)
            if success:
                best_rank = new_rank
                best_U = U_r
                best_V = V_r
                best_W = W_r
                continue
            
            # Try rounding to quarters
            U_r, V_r, W_r, success = try_round_quarter_and_verify(U_res, V_res, W_res, new_rank)
            if success:
                best_rank = new_rank
                best_U = U_r
                best_V = V_r
                best_W = W_r
    
    # Strategy 2: More aggressive elimination with different seeds
    if time.time() < deadline - 30 and best_rank > 35:
        for trial in range(50):
            if time.time() > deadline - 20:
                break
            
            np.random.seed(1000 + trial * 211)
            
            max_elim = rank - 34 + (trial % 8)
            
            U_res, V_res, W_res, new_rank = eliminate_columns(
                U_np, V_np, W_np, max_elim, deadline - 20, seed_offset=trial + 100
            )
            
            if U_res is not None and new_rank < best_rank:
                U_r, V_r, W_r, success = try_round_and_verify(U_res, V_res, W_res, new_rank)
                if success:
                    best_rank = new_rank
                    best_U = U_r
                    best_V = V_r
                    best_W = W_r
                    continue
                
                U_r, V_r, W_r, success = try_round_int_and_verify(U_res, V_res, W_res, new_rank)
                if success:
                    best_rank = new_rank
                    best_U = U_r
                    best_V = V_r
                    best_W = W_r
    
    # Strategy 3: Try to eliminate from the best result found so far
    if time.time() < deadline - 30 and best_rank < rank:
        for trial in range(40):
            if time.time() > deadline - 20:
                break
            
            np.random.seed(5000 + trial * 331)
            
            max_elim = best_rank - 32 + (trial % 6)
            
            U_res, V_res, W_res, new_rank = eliminate_columns(
                best_U, best_V, best_W, max_elim, deadline - 20, seed_offset=trial + 200
            )
            
            if U_res is not None and new_rank < best_rank:
                U_r, V_r, W_r, success = try_round_and_verify(U_res, V_res, W_res, new_rank)
                if success:
                    best_rank = new_rank
                    best_U = U_r
                    best_V = V_r
                    best_W = W_r
                    continue
                
                U_r, V_r, W_r, success = try_round_int_and_verify(U_res, V_res, W_res, new_rank)
                if success:
                    best_rank = new_rank
                    best_U = U_r
                    best_V = V_r
                    best_W = W_r
    
    # Strategy 4: Random perturbation refinement
    if time.time() < deadline - 40 and best_rank > 36:
        np.random.seed(9999)
        U_ref, V_ref, W_ref = refine_with_random_perturbation(
            best_U, best_V, best_W, 20, deadline - 20
        )
        
        if U_ref.shape[1] < best_rank:
            U_r, V_r, W_r, success = try_round_and_verify(U_ref, V_ref, W_ref, U_ref.shape[1])
            if success:
                best_rank = U_ref.shape[1]
                best_U = U_r
                best_V = V_r
                best_W = W_r
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
