import numpy as np
import time
import sys
from fractions import Fraction

_START_TIME = time.time()
_DEADLINE = _START_TIME + 110  # Safety margin before 120s limit

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    def build_current_solution():
        """Build the 93-term solution."""
        total_rank = 93
        
        U = [[0.0] * total_rank for _ in range(n * m)]
        V = [[0.0] * total_rank for _ in range(m * p)]
        W = [[0.0] * total_rank for _ in range(p * n)]
        
        current_rank = 0
        
        # Part 1: 4x4x4 using Strassen (Rank 49)
        strassen_terms = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}),
            ({2: 1, 3: 1}, {0: 1}),
            ({0: 1}, {1: 1, 3: -1}),
            ({3: 1}, {2: 1, 0: -1}),
            ({0: 1, 1: 1}, {3: 1}),
            ({2: 1, 0: -1}, {0: 1, 1: 1}),
            ({1: 1, 3: -1}, {2: 1, 3: 1})
        ]
        
        sub_terms = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1})
        ]
        
        for m_idx in range(7):
            x_def, y_def = strassen_terms[m_idx]
            for t_idx in range(7):
                r = current_rank
                current_rank += 1
                x_sub, y_sub = sub_terms[t_idx]
                
                u_coeffs = [0.0] * (n*m)
                v_coeffs = [0.0] * (m*p)
                
                for (li, lj), c_x in x_sub.items():
                    for b, coeff_block in x_def.items():
                        bi = b // 2
                        bj = b % 2
                        i = 2 * bi + li
                        j = 2 * bj + lj
                        u_idx = i * m + j
                        u_coeffs[u_idx] += c_x * coeff_block
                
                for (li, lj), c_y in y_sub.items():
                    for b, coeff_block in y_def.items():
                        bi = b // 2
                        bj = b % 2
                        i = 2 * bi + li
                        j = 2 * bj + lj
                        v_idx = i * p + j
                        v_coeffs[v_idx] += c_y * coeff_block
                
                for i in range(n*m):
                    U[i][r] = u_coeffs[i]
                for j in range(m*p):
                    V[j][r] = v_coeffs[j]
                
                w_coeffs = [0.0] * (p*n)
                
                cm_coeffs = {
                    (0,0): 0.0, (0,1): 0.0, (1,0): 0.0, (1,1): 0.0
                }
                
                if t_idx == 0:
                    cm_coeffs[(0,0)] += 1.0
                    cm_coeffs[(1,1)] += 1.0
                elif t_idx == 1:
                    cm_coeffs[(1,0)] += 1.0
                    cm_coeffs[(1,1)] += -1.0
                elif t_idx == 2:
                    cm_coeffs[(0,1)] += 1.0
                    cm_coeffs[(1,1)] += 1.0
                elif t_idx == 3:
                    cm_coeffs[(0,0)] += 1.0
                    cm_coeffs[(1,0)] += 1.0
                elif t_idx == 4:
                    cm_coeffs[(0,0)] += -1.0
                    cm_coeffs[(0,1)] += 1.0
                elif t_idx == 5:
                    cm_coeffs[(1,1)] += 1.0
                elif t_idx == 6:
                    cm_coeffs[(0,0)] += 1.0
                
                c_block_coeffs = {}
                if m_idx == 0:
                    c_block_coeffs = {(0,0): 1.0, (1,1): 1.0}
                elif m_idx == 1:
                    c_block_coeffs = {(1,0): 1.0, (1,1): -1.0}
                elif m_idx == 2:
                    c_block_coeffs = {(0,1): 1.0, (1,1): 1.0}
                elif m_idx == 3:
                    c_block_coeffs = {(0,0): 1.0, (1,0): 1.0}
                elif m_idx == 4:
                    c_block_coeffs = {(0,0): -1.0, (0,1): 1.0}
                elif m_idx == 5:
                    c_block_coeffs = {(1,1): 1.0}
                elif m_idx == 6:
                    c_block_coeffs = {(0,0): 1.0}
                
                for (li, lj), cm_c in cm_coeffs.items():
                    if cm_c == 0.0: continue
                    for (br, bc), cb_c in c_block_coeffs.items():
                        if cb_c == 0.0: continue
                        i_final = 2 * br + li
                        k_final = 2 * bc + lj
                        w_idx = k_final * n + i_final
                        w_coeffs[w_idx] += cm_c * cb_c
                
                for i in range(p*n):
                    W[i][r] = w_coeffs[i]
        
        # Part 2: 4x4x3 using 4 * (2x2x3) (Rank 44)
        def add_2x2x3(a_row_block, a_col_block, b_row_block, c_row_block):
            nonlocal current_rank
            
            strassen_2x2x2_terms = [
                ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
                ({(1,0): 1, (1,1): 1}, {(0,0): 1}),
                ({(0,0): 1}, {(0,1): 1, (1,1): -1}),
                ({(1,1): 1}, {(1,0): 1, (0,0): -1}),
                ({(0,0): 1, (0,1): 1}, {(1,1): 1}),
                ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
                ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1})
            ]
            
            for t_idx in range(7):
                r = current_rank
                current_rank += 1
                x_sub, y_sub = strassen_2x2x2_terms[t_idx]
                
                u_coeffs = [0.0] * (n*m)
                v_coeffs = [0.0] * (m*p)
                
                for (li, lj), c_x in x_sub.items():
                    i = 2 * a_row_block + li
                    j = 2 * a_col_block + lj
                    u_idx = i * m + j
                    u_coeffs[u_idx] += c_x
                
                for (li, lj), c_y in y_sub.items():
                    i = 2 * b_row_block + li
                    j = 4 + lj
                    v_idx = i * p + j
                    v_coeffs[v_idx] += c_y
                
                for i in range(n*m):
                    U[i][r] = u_coeffs[i]
                for j in range(m*p):
                    V[j][r] = v_coeffs[j]
                
                w_coeffs = [0.0] * (p*n)
                
                cm_coeffs = {
                    (0,0): 0.0, (0,1): 0.0, (1,0): 0.0, (1,1): 0.0
                }
                
                if t_idx == 0:
                    cm_coeffs[(0,0)] += 1.0
                    cm_coeffs[(1,1)] += 1.0
                elif t_idx == 1:
                    cm_coeffs[(1,0)] += 1.0
                    cm_coeffs[(1,1)] += -1.0
                elif t_idx == 2:
                    cm_coeffs[(0,1)] += 1.0
                    cm_coeffs[(1,1)] += 1.0
                elif t_idx == 3:
                    cm_coeffs[(0,0)] += 1.0
                    cm_coeffs[(1,0)] += 1.0
                elif t_idx == 4:
                    cm_coeffs[(0,0)] += -1.0
                    cm_coeffs[(0,1)] += 1.0
                elif t_idx == 5:
                    cm_coeffs[(1,1)] += 1.0
                elif t_idx == 6:
                    cm_coeffs[(0,0)] += 1.0
                
                for (li, lj), cm_c in cm_coeffs.items():
                    if cm_c == 0.0: continue
                    i_final = 2 * c_row_block + li
                    k_final = 4 + lj
                    w_idx = k_final * n + i_final
                    w_coeffs[w_idx] += cm_c
                
                for i in range(p*n):
                    W[i][r] = w_coeffs[i]
            
            schoolbook_terms = [
                ((0,0), (0,0), 0),
                ((0,1), (1,0), 0),
                ((1,0), (0,0), 1),
                ((1,1), (1,0), 1),
            ]
            
            for a_pos, b_pos, c_row in schoolbook_terms:
                r = current_rank
                current_rank += 1
                
                u_coeffs = [0.0] * (n*m)
                v_coeffs = [0.0] * (m*p)
                
                i_a = 2 * a_row_block + a_pos[0]
                j_a = 2 * a_col_block + a_pos[1]
                u_coeffs[i_a * m + j_a] = 1.0
                
                i_b = 2 * b_row_block + b_pos[0]
                j_b = 6
                v_coeffs[i_b * p + j_b] = 1.0
                
                for i in range(n*m):
                    U[i][r] = u_coeffs[i]
                for j in range(m*p):
                    V[j][r] = v_coeffs[j]
                
                w_coeffs = [0.0] * (p*n)
                i_final = 2 * c_row_block + c_row
                k_final = 6
                w_idx = k_final * n + i_final
                w_coeffs[w_idx] = 1.0
                
                for i in range(p*n):
                    W[i][r] = w_coeffs[i]
        
        add_2x2x3(0, 0, 0, 0)
        add_2x2x3(0, 1, 1, 0)
        add_2x2x3(1, 0, 0, 1)
        add_2x2x3(1, 1, 1, 1)
        
        U = [row[:current_rank] for row in U]
        V = [row[:current_rank] for row in V]
        W = [row[:current_rank] for row in W]
        
        return U, V, W, current_rank
    
    def verify(U, V, W, rank):
        """Verify exact decomposition."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        
        target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    target[a_idx, b_idx, c_idx] = 1.0
        
        recon = np.einsum('ir,jr,kr->ijk', Ua, Va, Wa)
        max_err = np.max(np.abs(recon - target))
        return max_err < 1e-9
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W, rank
    
    def try_merge_terms(U, V, W, rank):
        """Try to merge linearly dependent terms to reduce rank."""
        if time.time() > _DEADLINE - 40:
            return U, V, W, rank
        
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        
        # For each pair of terms, check if one can be eliminated
        # by expressing it as a combination of others
        # This is expensive, so we only try a few strategies
        
        # Strategy 1: Check if any two terms have proportional U vectors
        merged = True
        while merged and time.time() < _DEADLINE - 40:
            merged = False
            new_U = Ua[:, :rank].copy()
            new_V = Va[:, :rank].copy()
            new_W = Wa[:, :rank].copy()
            
            for i in range(rank - 1):
                if time.time() > _DEADLINE - 40:
                    break
                for j in range(i + 1, rank):
                    # Check if U[:,j] is proportional to U[:,i]
                    u_i = new_U[:, i]
                    u_j = new_U[:, j]
                    
                    if np.allclose(u_i, 0, atol=1e-12) or np.allclose(u_j, 0, atol=1e-12):
                        continue
                    
                    # Check proportionality
                    ratios = u_j / u_i
                    mask = np.abs(u_i) > 1e-10
                    if np.all(np.abs(ratios[mask] - ratios[mask][0]) < 1e-8):
                        ratio = ratios[mask][0]
                        # Check if V and W are also proportional
                        v_i = new_V[:, i]
                        v_j = new_V[:, j]
                        w_i = new_W[:, i]
                        w_j = new_W[:, j]
                        
                        mask_v = np.abs(v_i) > 1e-10
                        mask_w = np.abs(w_i) > 1e-10
                        
                        if len(mask_v) > 0 and len(mask_w) > 0:
                            v_ratio = v_j[mask_v] / v_i[mask_v]
                            w_ratio = w_j[mask_w] / w_i[mask_w]
                            
                            if (np.all(np.abs(v_ratio - v_ratio[0]) < 1e-8) and
                                np.all(np.abs(w_ratio - w_ratio[0]) < 1e-8) and
                                abs(v_ratio[0] - ratio) < 1e-8 and
                                abs(w_ratio[0] - ratio) < 1e-8):
                                # Terms are proportional, remove term j
                                new_U = np.delete(new_U, j, axis=1)
                                new_V = np.delete(new_V, j, axis=1)
                                new_W = np.delete(new_W, j, axis=1)
                                rank -= 1
                                merged = True
                                break
                if merged:
                    break
            
            if not merged:
                break
        
        if rank < len(U):
            return new_U.tolist(), new_V.tolist(), new_W.tolist(), rank
        return U, V, W, rank
    
    # Fast alternating solve using matrix operations
    def try_fast_alternating_solve(target_rank, max_iters=50, init_U=None, init_V=None, init_W=None):
        """Fast alternating linear solve using precomputed structure."""
        if time.time() > _DEADLINE - 30:
            return None
        
        if init_U is not None:
            if init_U.shape[1] >= target_rank:
                U = init_U[:, :target_rank].copy()
                V = init_V[:, :target_rank].copy()
                W = init_W[:, :target_rank].copy()
            else:
                U = np.zeros((n*m, target_rank))
                V = np.zeros((m*p, target_rank))
                W = np.zeros((p*n, target_rank))
        else:
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            # Sparse random init
            rng = np.random.default_rng(42)
            for t in range(target_rank):
                nz_u = rng.choice(n*m, size=min(6, n*m), replace=False)
                U[nz_u, t] = rng.choice([-1, 1], size=len(nz_u))
                nz_v = rng.choice(m*p, size=min(5, m*p), replace=False)
                V[nz_v, t] = rng.choice([-1, 1], size=len(nz_v))
                nz_w = rng.choice(p*n, size=min(6, p*n), replace=False)
                W[nz_w, t] = rng.choice([-1, 1], size=len(nz_w))
        
        target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target[i*m+j, j*p+k, k*n+i] = 1.0
        
        best_err = float('inf')
        best_U_sol, best_V_sol, best_W_sol = None, None, None
        
        for iteration in range(max_iters):
            if time.time() > _DEADLINE - 30:
                break
            
            # Solve for V: fix U, W
            # For each (i,k), we need sum_t U[i,t]*W[k,t]*V[j*p+k,t] = 1 for all j
            # This is a linear system in V
            # A_v has shape (n*m*p, m*p*target_rank)
            # But we can exploit structure: for fixed (j,k), the equation involves V[j*p+k, :]
            
            # Build A_v efficiently
            A_v = np.zeros((n * m * p, m * p * target_rank))
            b_v = np.ones(n * m * p)
            
            # Precompute UW products
            UW = U @ W.T  # (n*m, p*n)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Equation: sum_t U[i*m+j, t] * W[k*n+i, t] * V[j*p+k, t] = 1
                        for t in range(target_rank):
                            coeff = U[i*m+j, t] * W[k*n+i, t]
                            if abs(coeff) > 1e-15:
                                var_idx = (j*p+k) * target_rank + t
                                A_v[eq_idx, var_idx] += coeff
                        eq_idx += 1
            
            try:
                V_flat, _, _, _ = np.linalg.lstsq(A_v, b_v, rcond=None)
                V = V_flat.reshape(m * p, target_rank)
            except:
                break
            
            # Solve for W: fix U, V
            A_w = np.zeros((n * m * p, p * n * target_rank))
            b_w = np.ones(n * m * p)
            
            UV = U @ V.T  # (n*m, m*p)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for t in range(target_rank):
                            coeff = U[i*m+j, t] * V[j*p+k, t]
                            if abs(coeff) > 1e-15:
                                var_idx = (k*n+i) * target_rank + t
                                A_w[eq_idx, var_idx] += coeff
                        eq_idx += 1
            
            try:
                W_flat, _, _, _ = np.linalg.lstsq(A_w, b_w, rcond=None)
                W = W_flat.reshape(p * n, target_rank)
            except:
                break
            
            # Solve for U: fix V, W
            A_u = np.zeros((n * m * p, n * m * target_rank))
            b_u = np.ones(n * m * p)
            
            VW = V @ W.T  # (m*p, p*n)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for t in range(target_rank):
                            coeff = V[j*p+k, t] * W[k*n+i, t]
                            if abs(coeff) > 1e-15:
                                var_idx = (i*m+j) * target_rank + t
                                A_u[eq_idx, var_idx] += coeff
                        eq_idx += 1
            
            try:
                U_flat, _, _, _ = np.linalg.lstsq(A_u, b_u, rcond=None)
                U = U_flat.reshape(n * m, target_rank)
            except:
                break
            
            # Check convergence
            recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            err = np.max(np.abs(recon - target))
            
            if err < best_err:
                best_err = err
                best_U_sol = U.copy()
                best_V_sol = V.copy()
                best_W_sol = W.copy()
            
            if err < 1e-9:
                return U.tolist(), V.tolist(), W.tolist(), target_rank
        
        if best_err < 1e-6 and best_U_sol is not None:
            return best_U_sol.tolist(), best_V_sol.tolist(), best_W_sol.tolist(), target_rank
        
        return None
    
    # Build the current 93-term solution first (our baseline)
    U_cur, V_cur, W_cur, rank_cur = build_current_solution()
    
    if not verify(U_cur, V_cur, W_cur, rank_cur):
        # Fallback to schoolbook
        U_sb, V_sb, W_sb, rank_sb = schoolbook()
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    best_U, best_V, best_W, best_rank = U_cur, V_cur, W_cur, rank_cur
    
    # Try merging terms
    U_m, V_m, W_m, rank_m = try_merge_terms(U_cur, V_cur, W_cur, rank_cur)
    if rank_m < best_rank and verify(U_m, V_m, W_m, rank_m):
        best_U, best_V, best_W, best_rank = U_m, V_m, W_m, rank_m
    
    # Try alternating solve starting from the current solution
    U_cur_a = np.array(best_U, dtype=float)
    V_cur_a = np.array(best_V, dtype=float)
    W_cur_a = np.array(best_W, dtype=float)
    
    for target_rank in range(best_rank - 1, max(best_rank - 6, 85), -1):
        if time.time() > _DEADLINE - 35:
            break
        
        result = try_fast_alternating_solve(target_rank, max_iters=40, 
                                             init_U=U_cur_a, init_V=V_cur_a, init_W=W_cur_a)
        if result is not None:
            U_new, V_new, W_new, r_new = result
            if verify(U_new, V_new, W_new, r_new):
                best_U, best_V, best_W, best_rank = U_new, V_new, W_new, r_new
                # Update for next iteration
                U_cur_a = np.array(U_new, dtype=float)
                V_cur_a = np.array(V_new, dtype=float)
                W_cur_a = np.array(W_new, dtype=float)
    
    # Final polish: Round to nearest half-integer and verify
    def round_to_half_int(mat):
        if isinstance(mat, list):
            return [[round(val * 2) / 2 for val in row] for row in mat]
        else:
            return (np.round(np.array(mat) * 2) / 2).tolist()
            
    U_out = round_to_half_int(best_U)
    V_out = round_to_half_int(best_V)
    W_out = round_to_half_int(best_W)
    
    # Verify the rounded solution
    if verify(U_out, V_out, W_out, best_rank):
        return {"U": U_out, "V": V_out, "W": W_out}
    else:
        # Fallback to unrounded if it verifies
        if verify(best_U, best_V, best_W, best_rank):
            return {"U": best_U, "V": best_V, "W": best_W}
        else:
            # Fallback to schoolbook
            U_sb, V_sb, W_sb, rank_sb = schoolbook()
            return {"U": U_sb, "V": V_sb, "W": W_sb}
