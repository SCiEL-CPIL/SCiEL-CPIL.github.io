import numpy as np
import time
import sys

_START_TIME = time.time()
_DEADLINE = _START_TIME + 105  # Safety margin before 120s limit

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
    
    # Build the current 93-term solution first (our baseline)
    U_cur, V_cur, W_cur, rank_cur = build_current_solution()
    
    if not verify(U_cur, V_cur, W_cur, rank_cur):
        # Fallback to schoolbook
        U_sb, V_sb, W_sb, rank_sb = schoolbook()
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    best_U, best_V, best_W, best_rank = U_cur, V_cur, W_cur, rank_cur
    
    # Try to reduce rank by finding linear dependencies in the 93-term decomposition
    # The rank-1 tensors are T_t = U[:,t] ⊗ V[:,t] ⊗ W[:,t]
    # If T_t = sum_s alpha_s * T_s for s != t, we can eliminate term t
    
    def try_reduce_by_dependency(U, V, W, rank):
        """Try to find and eliminate linearly dependent terms."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        
        # Build the tensor matrix: each column t gives a vector in R^(n*m*p)
        # vec(T_t) = U[:,t] ⊗ V[:,t] ⊗ W[:,t]
        # If these vectors are linearly dependent, we can reduce rank
        
        tensor_vecs = np.zeros((n*m*p, rank))
        for t in range(rank):
            tensor_vecs[:, t] = np.kron(np.kron(Ua[:, t], Va[:, t]), Wa[:, t])
        
        # Find the rank of this matrix
        # Use SVD to find null space
        try:
            U_svd, S_vals, Vt = np.linalg.svd(tensor_vecs, full_matrices=False)
        except:
            return U, V, W, rank
        
        # Find how many singular values are above tolerance
        tol = 1e-10
        num_independent = np.sum(S_vals > tol)
        
        if num_independent >= rank:
            return U, V, W, rank  # Already minimal
        
        # Find null space vectors
        null_space = Vt[num_independent:]  # shape (rank - num_independent, rank)
        
        # For each null space vector, we can eliminate one term
        # Keep the first num_independent terms (simplified approach)
        # Actually, let's be more careful: find which terms to keep
        
        # Simple approach: keep first num_independent terms
        # This works if the first num_independent columns are independent
        if num_independent < rank:
            # Verify that keeping first num_independent terms still works
            # (This is only valid if the null space structure allows it)
            # Safer: use QR to find a basis
            pass
        
        # For safety, just return original if reduction is complex
        return U, V, W, rank
    
    # Alternative: try to merge terms by finding pairs that can be combined
    def try_merge_terms(U, V, W, rank):
        """Try to find pairs of terms that can be merged into one."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        
        target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target[i*m+j, j*p+k, k*n+i] = 1.0
        
        best_U, best_V, best_W, best_rank = Ua, Va, Wa, rank
        
        # Try removing each term and seeing if we can compensate
        for remove_t in range(rank):
            if time.time() > _DEADLINE - 30:
                break
            
            # Remove term remove_t
            U_rem = np.delete(Ua, remove_t, axis=1)
            V_rem = np.delete(Va, remove_t, axis=1)
            W_rem = np.delete(Wa, remove_t, axis=1)
            
            recon_rem = np.einsum('ir,jr,kr->ijk', U_rem, V_rem, W_rem)
            residual = target - recon_rem
            
            # Check if residual is a rank-1 tensor
            # Reshape residual to (n*m, m*p, p*n) and check if it's rank 1
            # A tensor is rank 1 if it can be written as a ⊗ b ⊗ c
            # Check by unfolding: if residual[i,j,k] = a[i]*b[j]*c[k]
            # Then for any i1,i2,j1,j2: residual[i1,j1,k]*residual[i2,j2,k] = residual[i1,j2,k]*residual[i2,j1,k]
            
            # Simpler: try to fit a rank-1 tensor to the residual
            # Use the fact that if residual = a⊗b⊗c, then 
            # residual[i,j,k] / residual[i',j,k] = a[i]/a[i'] for all j,k
            # where a[i'] != 0
            
            # Try: pick a reference (i0, j0) and solve
            # residual[i,j,k] = a[i]*b[j]*c[k]
            # Let a[i0] = 1 (scale), then b[j]*c[k] = residual[i0,j,k]
            # And a[i]*b[j]*c[k] = residual[i,j,k]
            # So a[i] = residual[i,j,k] / (b[j]*c[k]) for any j,k with b[j]*c[k] != 0
            
            # Try multiple references
            found = False
            for i0 in range(n*m):
                if found: break
                for j0 in range(m*p):
                    if found: break
                    ref = residual[i0, j0, :]
                    if np.all(np.abs(ref) < 1e-15):
                        continue
                    # b[j0]*c[k] = ref[k] for all k
                    # a[i]*b[j]*c[k] = residual[i,j,k]
                    # a[i]*b[j] = residual[i,j,k] / c[k] for any k with c[k] != 0
                    # a[i]*b[j] = residual[i,j,k] * b[j0] / ref[k] for any k with ref[k] != 0
                    
                    # Find k0 where ref[k0] != 0
                    k0 = None
                    for k in range(p*n):
                        if abs(ref[k]) > 1e-15:
                            k0 = k
                            break
                    if k0 is None:
                        continue
                    
                    # a[i]*b[j] * ref[k0] = residual[i,j,k0] * b[j0]
                    # a[i]*b[j] = residual[i,j,k0] * b[j0] / ref[k0]
                    # But we don't know b[j0] yet.
                    
                    # Alternative: a[i] = residual[i,j0,k0] / (b[j0]*c[k0])
                    # Let's set c[k0] = 1 (scale), then b[j0] = ref[k0]
                    # a[i] = residual[i,j0,k0] / ref[k0]
                    # b[j] = residual[i0,j,k0] / ref[k0]  ... wait this isn't right
                    
                    # Let me use a different approach:
                    # residual[i,j,k] = a[i]*b[j]*c[k]
                    # Fix c[k0] = 1, so b[j]*1 = residual[i0,j,k0]/a[i0]
                    # And a[i]*b[j] = residual[i,j,k0]
                    # So a[i] = residual[i,j,k0] / b[j] = residual[i,j,k0] * a[i0] / residual[i0,j,k0]
                    
                    # This is getting circular. Let me just use SVD on the unfolding.
                    break
                break
            
            # Use SVD approach for rank-1 check
            # Unfold residual along first mode: R_1[i, (j,k)] = residual[i,j,k]
            R1 = residual.reshape(n*m, m*p*p*n)
            # If rank 1, R1 should be rank 1 as a matrix
            try:
                _, S1, _ = np.linalg.svd(R1, full_matrices=False)
                if len(S1) > 0 and S1[0] > 1e-10:
                    if len(S1) > 1 and S1[1] / S1[0] < 1e-10:
                        # Rank 1! Extract the factors
                        # R1 = a * (b*c)^T where b*c is the second factor
                        # a = R1[:, 0] / S1[0] * S1[0] ... 
                        # Actually from SVD: R1 = U * diag(S) * Vt
                        # If rank 1: R1 = S1[0] * u0 * v0^T
                        # So a[i] = S1[0] * u0[i], and b[j]*c[k] = v0[j*p*n + k]
                        # We need to factor v0 into b and c
                        # v0[j*p*n + k] = b[j] * c[k]
                        # This is a rank-1 matrix factorization of the (m*p) x (p*n) matrix
                        V0 = v0_reshape = None
                        # Get v0 from SVD
                        U_svd1, S_svd1, Vt_svd1 = np.linalg.svd(R1, full_matrices=False)
                        v0 = Vt_svd1[0]  # shape (m*p*p*n,)
                        V0 = v0.reshape(m*p, p*n)
                        
                        # Factor V0 = b * c^T (outer product)
                        # Check if V0 is rank 1
                        _, S_V0, _ = np.linalg.svd(V0, full_matrices=False)
                        if len(S_V0) > 0 and S_V0[0] > 1e-10:
                            if len(S_V0) > 1 and S_V0[1] / S_V0[0] < 1e-10:
                                # V0 is rank 1
                                U_svdV, S_svdV, Vt_svdV = np.linalg.svd(V0, full_matrices=False)
                                b_vec = S_svdV[0] * U_svdV[:, 0]  # shape (m*p,)
                                c_vec = Vt_svdV[0]  # shape (p*n,)
                                
                                # Verify: b[j]*c[k] should equal V0[j,k]
                                # And a[i] = S1[0] * U_svd1[i, 0]
                                a_vec = S_svd1[0] * U_svd1[:, 0]
                                
                                # Verify the reconstruction
                                recon_check = np.einsum('i,jk->ijk', a_vec, np.outer(b_vec, c_vec))
                                if np.max(np.abs(recon_check - residual)) < 1e-8:
                                    # Found a rank-1 residual! We can add it back as one term
                                    new_U = np.column_stack([U_rem, a_vec])
                                    new_V = np.column_stack([V_rem, b_vec])
                                    new_W = np.column_stack([W_rem, c_vec])
                                    
                                    # Verify full reconstruction
                                    full_recon = np.einsum('ir,jr,kr->ijk', new_U, new_V, new_W)
                                    if np.max(np.abs(full_recon - target)) < 1e-8:
                                        new_rank = rank - 1
                                        if new_rank < best_rank:
                                            best_U = new_U
                                            best_V = new_V
                                            best_W = new_W
                                            best_rank = new_rank
            except:
                pass
        
        return best_U.tolist(), best_V.tolist(), best_W.tolist(), best_rank
    
    # Try to reduce by finding mergeable terms
    if time.time() < _DEADLINE - 40:
        U_new, V_new, W_new, rank_new = try_merge_terms(
            np.array(U_cur, dtype=float), np.array(V_cur, dtype=float), 
            np.array(W_cur, dtype=float), rank_cur
        )
        if rank_new < best_rank:
            if verify(U_new, V_new, W_new, rank_new):
                best_U, best_V, best_W, best_rank = U_new, V_new, W_new, rank_new
    
    # Try alternating solve with better initialization from the current solution
    def try_alternating_solve(target_rank, max_iters=50, init_U=None, init_V=None, init_W=None):
        """Try alternating linear solves with optional initialization."""
        if time.time() > _DEADLINE - 15:
            return None
        
        if init_U is not None:
            U = init_U[:, :target_rank].copy() if init_U.shape[1] >= target_rank else np.zeros((n*m, target_rank))
            V = init_V[:, :target_rank].copy() if init_V.shape[1] >= target_rank else np.zeros((m*p, target_rank))
            W = init_W[:, :target_rank].copy() if init_W.shape[1] >= target_rank else np.zeros((p*n, target_rank))
        else:
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            for t in range(target_rank):
                num_nonzero_u = min(4, n * m)
                positions = [(t * 3 + s * 7) % (n * m) for s in range(num_nonzero_u)]
                for pos in set(positions):
                    U[pos, t] = 1.0
                num_nonzero_w = min(4, p * n)
                w_positions = [(t * 5 + s * 11) % (p * n) for s in range(num_nonzero_w)]
                for pos in set(w_positions):
                    W[pos, t] = 1.0
                num_nonzero_v = min(3, m * p)
                v_positions = [(t * 7 + s * 13) % (m * p) for s in range(num_nonzero_v)]
                for pos in set(v_positions):
                    V[pos, t] = 1.0
        
        target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target[i*m+j, j*p+k, k*n+i] = 1.0
        
        for iteration in range(max_iters):
            if time.time() > _DEADLINE - 15:
                break
            
            # Fix U and W, solve for V
            A_v = np.zeros((n * m * p, m * p * target_rank))
            b_v = np.ones(n * m * p)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for t in range(target_rank):
                            coeff = U[i * m + j, t] * W[k * n + i, t]
                            if coeff != 0:
                                var_idx = (j * p + k) * target_rank + t
                                A_v[eq_idx, var_idx] += coeff
                        eq_idx += 1
            
            try:
                V_flat, _, _, _ = np.linalg.lstsq(A_v, b_v, rcond=None)
                V = V_flat.reshape(m * p, target_rank)
            except:
                break
            
            # Fix U and V, solve for W
            A_w = np.zeros((n * m * p, p * n * target_rank))
            b_w = np.ones(n * m * p)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for t in range(target_rank):
                            coeff = U[i * m + j, t] * V[j * p + k, t]
                            if coeff != 0:
                                var_idx = (k * n + i) * target_rank + t
                                A_w[eq_idx, var_idx] += coeff
                        eq_idx += 1
            
            try:
                W_flat, _, _, _ = np.linalg.lstsq(A_w, b_w, rcond=None)
                W = W_flat.reshape(p * n, target_rank)
            except:
                break
            
            # Fix V and W, solve for U
            A_u = np.zeros((n * m * p, n * m * target_rank))
            b_u = np.ones(n * m * p)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for t in range(target_rank):
                            coeff = V[j * p + k, t] * W[k * n + i, t]
                            if coeff != 0:
                                var_idx = (i * m + j) * target_rank + t
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
            
            if err < 1e-9:
                return U.tolist(), V.tolist(), W.tolist(), target_rank
        
        return None
    
    # Try alternating solve starting from the 93-term solution, targeting 92
    U_cur_a = np.array(U_cur, dtype=float)
    V_cur_a = np.array(V_cur, dtype=float)
    W_cur_a = np.array(W_cur, dtype=float)
    
    for target_rank in range(best_rank - 1, max(best_rank - 5, 85), -1):
        if time.time() > _DEADLINE - 25:
            break
        
        result = try_alternating_solve(target_rank, max_iters=30, 
                                       init_U=U_cur_a, init_V=V_cur_a, init_W=W_cur_a)
        if result is not None:
            U_new, V_new, W_new, r_new = result
            if verify(U_new, V_new, W_new, r_new):
                best_U, best_V, best_W, best_rank = U_new, V_new, W_new, r_new
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}

