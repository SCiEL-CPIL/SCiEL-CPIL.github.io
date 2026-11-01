"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Build the target tensor T[a, b, c]
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # First, construct the known rank-42 Strassen decomposition as a safe fallback
    def build_strassen_rank42():
        rank = 42
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Strassen 7 products for 2x2 x 2x2
        # M1 = (a00+a11)*(b00+b11) -> c00+=1, c11+=1
        # M2 = (a10+a11)*b00 -> c10+=1
        # M3 = a00*(b01-b11) -> c01+=1
        # M4 = a11*(b10-b00) -> c00+=1, c10+=1
        # M5 = (a00+a01)*b11 -> c00-=1, c01+=1
        # M6 = (a10-a00)*(b00+b01) -> c11+=1
        # M7 = (a01-a11)*(b10+b11) -> c00-=1, c11+=1
        
        # Encode as sparse dicts: key is flat index, value is coeff
        # U indices: i*m+j
        # V indices: j*p+k (but within a pair, k_local is 0 or 1, actual k = 2t+k_local)
        # W indices: k*n+i (actual k = 2t+k_local)
        
        strassen_products = [
            # M1
            {'U': [(0, 1), (3, 1)], 'V': [(0, 1), (3, 1)], 'W': [(0, 1), (3, 1)]},
            # M2
            {'U': [(2, 1), (3, 1)], 'V': [(0, 1)], 'W': [(1, 1)]},
            # M3
            {'U': [(0, 1)], 'V': [(1, 1), (3, -1)], 'W': [(2, 1)]},
            # M4
            {'U': [(3, 1)], 'V': [(2, 1), (0, -1)], 'W': [(0, 1), (1, 1)]},
            # M5
            {'U': [(0, 1), (1, 1)], 'V': [(3, 1)], 'W': [(0, -1), (2, 1)]},
            # M6
            {'U': [(2, 1), (0, -1)], 'V': [(0, 1), (1, 1)], 'W': [(3, 1)]},
            # M7
            {'U': [(1, 1), (3, -1)], 'V': [(2, 1), (3, 1)], 'W': [(0, -1), (3, 1)]},
        ]
        
        for t in range(6):
            for s in range(7):
                col = 7 * t + s
                for idx, coeff in strassen_products[s]['U']:
                    U[idx, col] = coeff
                for idx, coeff in strassen_products[s]['V']:
                    j = idx // 2
                    k_local = idx % 2
                    k = 2 * t + k_local
                    V[j * p + k, col] = coeff
                for idx, coeff in strassen_products[s]['W']:
                    k_local = idx // 2
                    i = idx % 2
                    k = 2 * t + k_local
                    W[k * n + i, col] = coeff
        
        # Verify
        T_rec = np.zeros((n * m, m * p, p * n))
        for col in range(rank):
            T_rec += np.einsum('i,j,k->ijk', U[:, col], V[:, col], W[:, col])
        err = np.linalg.norm(T_rec - T)
        return err, U, V, W
    
    err42, U42, V42, W42 = build_strassen_rank42()
    
    if err42 < 1e-9:
        # We have a valid rank-42 solution. Try to improve via optimization.
        best_rank = 42
        best_U, best_V, best_W = U42, V42, W42
    else:
        # Fallback to schoolbook rank 48
        rank_fb = 48
        U_fb = np.zeros((n * m, rank_fb))
        V_fb = np.zeros((m * p, rank_fb))
        W_fb = np.zeros((p * n, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U_fb[a, col] = 1
                    V_fb[b, col] = 1
                    W_fb[c, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    # Now attempt to find a lower-rank decomposition via structured search.
    # Key insight: the tensor has a special structure. For each output column k,
    # C[:,k] = A @ B[:,k]. The "A" factor is shared across all k.
    # This means we can think of it as: for each rank-1 term r, we have
    # u_r (length 4, the A-part), v_r (length 24, the B-part), w_r (length 24, the C-part).
    # The constraint is that sum_r u_r[i,j] * v_r[j,k] * w_r[k,i] = 1 for all i,j,k.
    
    # Strategy: Try rank 30 with a structured parameterization.
    # Partition 12 columns into 3 groups of 4. For each group, use a 2x2 x 2x4 algorithm.
    # The known best for 2x2 x 2x4 is rank 10 (from literature: Bini, Capovani, Lotti, Romani 1994
    # showed rank 10 for 2x2 x 2x4, improving over 2*7=14 from two Strassens).
    # Actually, let me verify: the rank of [2,2,4] tensor.
    # Lower bound: max(n*m, m*p, n*p) = max(4, 8, 8) = 8.
    # Schoolbook: 2*2*4 = 16.
    # Two Strassens: 2*7 = 14.
    # Known result: rank([2,2,4]) = 10? Let me try to construct it.
    
    # Actually, let me try a different approach: use numerical optimization to find
    # a rank-30 decomposition by starting from the rank-42 Strassen and trying to
    # zero out 12 columns through coordinate descent.
    
    # Alternative: try to directly construct rank 30 using a "3-group" structure.
    # For 3 groups of 4 columns each, if each group needs rank 10, total = 30.
    
    # Let me try to find a rank-10 decomposition for 2x2 x 2x4.
    # The tensor has indices (i, j, k) with i in {0,1}, j in {0,1}, k in {0,1,2,3}.
    # T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j] B[j,k]).
    
    # This is equivalent to computing C = A @ B where A is 2x2 and B is 2x4.
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for i=0,1 and k=0,1,2,3.
    
    # Let's use a numerical approach: set up the equations and solve.
    # We need rank-1 terms (u_r, v_r, w_r) where:
    # u_r has 4 entries (A-part: i,j)
    # v_r has 8 entries (B-part: j,k for k=0..3)
    # w_r has 8 entries (C-part: k,i for k=0..3)
    # And sum_r u_r[i,j] * v_r[j,k] * w_r[k,i] = 1 for all i,j,k.
    
    # That's 2*2*4 = 16 equations. Each term has 4+8+8=20 parameters, but scaling
    # invariance reduces to 19 per term. For rank 10, we have 190 parameters for 16 equations.
    # Very overdetermined in the other direction - should be solvable.
    
    # Let me try a simple structured approach for 2x2 x 2x4:
    # Use the fact that we can write:
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    
    # Define 10 products. Let me try a pattern based on Strassen extended to 4 columns.
    # For 4 columns, we can use:
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,3])
    # M2 = (A[1,0]+A[1,1]) * (B[0,1]+B[0,2])  -- hmm, this is getting complex.
    
    # Let me just do a numerical search for rank 10 on the 2x2 x 2x4 subproblem.
    
    def try_rank10_2x2x4():
        """Try to find rank-10 decomposition for 2x2 x 2x4."""
        n_sub, m_sub, p_sub = 2, 2, 4
        # Target tensor
        T_sub = np.ones((n_sub * m_sub, m_sub * p_sub, p_sub * n_sub))
        # T_sub[a, b, c] where a=i*m+j, b=j*p+k, c=k*n+i
        # For our case, T_sub[i*2+j, j*4+k, k*2+i] = 1
        
        rank_sub = 10
        np.random.seed(42)
        
        # Initialize with random sparse patterns
        best_err = np.inf
        best_terms = None
        
        for attempt in range(50):
            if time.time() > deadline - 30:
                break
            
            # Random initialization with structured sparsity
            U_sub = np.zeros((n_sub * m_sub, rank_sub))
            V_sub = np.zeros((m_sub * p_sub, rank_sub))
            W_sub = np.zeros((p_sub * n_sub, rank_sub))
            
            for r in range(rank_sub):
                # Each term touches a small number of entries
                n_u = np.random.randint(1, 4)
                n_v = np.random.randint(1, 5)
                n_w = np.random.randint(1, 5)
                
                u_idx = np.random.choice(n_sub * m_sub, size=n_u, replace=False)
                v_idx = np.random.choice(m_sub * p_sub, size=n_v, replace=False)
                w_idx = np.random.choice(p_sub * n_sub, size=n_w, replace=False)
                
                U_sub[u_idx, r] = np.random.choice([-1, 1], size=n_u)
                V_sub[v_idx, r] = np.random.choice([-1, 1], size=n_v)
                W_sub[w_idx, r] = np.random.choice([-1, 1], size=n_w)
            
            # Coordinate descent: optimize each entry of U, V, W
            for iteration in range(200):
                if time.time() > deadline - 30:
                    break
                
                # Compute residual
                T_rec_sub = np.zeros((n_sub * m_sub, m_sub * p_sub, p_sub * n_sub))
                for r in range(rank_sub):
                    T_rec_sub += np.einsum('i,j,k->ijk', U_sub[:, r], V_sub[:, r], W_sub[:, r])
                residual = T_rec_sub - T_sub
                
                # Optimize each column of U, V, W
                for r in range(rank_sub):
                    # Optimize U[:, r]
                    # residual = sum_s u_s v_s w_s - T
                    # d/dU[a,r] = sum over (b,c) where V[b,r]*W[c,r] != 0: residual[a,b,c] * V[b,r] * W[c,r]
                    # Set U[a,r] = -sum_{a'} ... / sum ...
                    # Actually, for linear least squares on U[:,r]:
                    # min ||sum_s u_s (v_s ⊗ w_s) - T||_F^2
                    # For fixed other terms, min ||u_r * (v_r ⊗ w_r) - (T - sum_{s≠r} u_s v_s w_s)||_F^2
                    # u_r_opt = <v_r⊗w_r, T - sum_{s≠r}> / <v_r⊗w_r, v_r⊗w_r>
                    
                    vw = np.einsum('j,k->jk', V_sub[:, r], W_sub[:, r])  # shape (m*p, p*n)
                    # But we need to be careful with dimensions
                    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
                    # The outer product U[:,r] ⊗ V[:,r] ⊗ W[:,r] has shape (n*m, m*p, p*n)
                    
                    # For optimizing U[:,r]:
                    # The contribution of term r to T[a,b,c] is U[a,r]*V[b,r]*W[c,r]
                    # residual_without_r[a,b,c] = T[a,b,c] - sum_{s≠r} U[a,s]*V[b,s]*W[c,s]
                    # We want to minimize sum_{a,b,c} (U[a,r]*V[b,r]*W[c,r] - residual_without_r[a,b,c])^2
                    # = sum_a U[a,r]^2 * (sum_{b,c} V[b,r]^2 W[c,r]^2) - 2 * sum_a U[a,r] * (sum_{b,c} V[b,r]*W[c,r]*residual_without_r[a,b,c]) + const
                    # Optimal: U[a,r] = sum_{b,c} V[b,r]*W[c,r]*residual_without_r[a,b,c] / sum_{b,c} V[b,r]^2*W[c,r]^2
                    
                    # But computing residual_without_r is expensive. Use the full residual:
                    # residual[a,b,c] = sum_s U[a,s]*V[b,s]*W[c,s] - T[a,b,c]
                    # residual_without_r[a,b,c] = residual[a,b,c] - U[a,r]*V[b,r]*W[c,r]
                    
                    # Let me compute the full T_rec once
                    T_rec_full = np.zeros((n_sub * m_sub, m_sub * p_sub, p_sub * n_sub))
                    for s in range(rank_sub):
                        T_rec_full += np.einsum('i,j,k->ijk', U_sub[:, s], V_sub[:, s], W_sub[:, s])
                    res_full = T_rec_full - T_sub  # this is the negative of what we want
                    
                    # For U[a,r]:
                    # numerator[a] = sum_{b,c} V[b,r]*W[c,r]*(T[a,b,c] - sum_{s≠r} U[a,s]*V[b,s]*W[c,s])
                    #             = sum_{b,c} V[b,r]*W[c,r]*T[a,b,c] - U[a,r]*sum_{b,c} V[b,r]^2*W[c,r]^2 + sum_{s≠r} U[a,s]*sum_{b,c} V[b,r]*V[b,s]*W[c,r]*W[c,s]
                    # This is getting complex. Let me just do a simple gradient step.
                    
                    # Simple approach: for each nonzero entry of U[:,r], compute gradient and step.
                    for a in np.where(U_sub[:, r] != 0)[0]:
                        # gradient of ||T_rec - T||^2 w.r.t. U[a,r]
                        # = 2 * sum_{b,c} (T_rec[a,b,c] - T[a,b,c]) * V[b,r] * W[c,r]
                        grad = 2.0 * np.sum(res_full[a, :, :] * V_sub[:, r] * W_sub[:, r])
                        # But res_full includes the contribution of U[a,r] itself
                        # Let me recompute without term r for this entry
                        # Actually, let's just do a small step
                        step = 0.1 * grad
                        U_sub[a, r] -= step
                    
                    # Similarly for V and W
                    for b in np.where(V_sub[:, r] != 0)[0]:
                        grad = 2.0 * np.sum(res_full[:, b, :] * U_sub[:, r] * W_sub[:, r])
                        V_sub[b, r] -= 0.1 * grad
                    
                    for c in np.where(W_sub[:, r] != 0)[0]:
                        grad = 2.0 * np.sum(res_full[:, :, c] * U_sub[:, r] * V_sub[:, r])
                        W_sub[c, r] -= 0.1 * grad
                
                # Check convergence
                T_rec_sub = np.zeros((n_sub * m_sub, m_sub * p_sub, p_sub * n_sub))
                for r in range(rank_sub):
                    T_rec_sub += np.einsum('i,j,k->ijk', U_sub[:, r], V_sub[:, r], W_sub[:, r])
                err = np.linalg.norm(T_rec_sub - T_sub)
                
                if err < best_err:
                    best_err = err
                    best_terms = (U_sub.copy(), V_sub.copy(), W_sub.copy())
                
                if err < 1e-6:
                    break
        
        if best_err < 1e-6:
            return True, best_terms
        return False, best_terms
    
    # Try to find rank-10 for 2x2 x 2x4
    found_r10, r10_terms = try_rank10_2x2x4()
    
    if found_r10 and time.time() < deadline - 10:
        # Great! We have a rank-10 decomposition for each 2x2 x 2x4 block.
        # Assemble into rank-30 for 2x2 x 2x12.
        U_sub, V_sub, W_sub = r10_terms
        rank_total = 30
        U_total = np.zeros((n * m, rank_total))
        V_total = np.zeros((m * p, rank_total))
        W_total = np.zeros((p * n, rank_total))
        
        for g in range(3):
            for r in range(10):
                col = 10 * g + r
                # U is the same for all groups (A doesn't change)
                U_total[:, col] = U_sub[:, r]
                # V: j*p+k where k = 4*g + k_local
                for b_idx in range(m_sub * p_sub):
                    j = b_idx // p_sub
                    k_local = b_idx % p_sub
                    k = 4 * g + k_local
                    V_total[j * p + k, col] = V_sub[b_idx, r]
                # W: k*n+i where k = 4*g + k_local
                for c_idx in range(p_sub * n_sub):
                    k_local = c_idx // n_sub
                    i = c_idx % n_sub
                    k = 4 * g + k_local
                    W_total[k * n + i, col] = W_sub[c_idx, r]
        
        # Verify
        T_rec = np.zeros((n * m, m * p, p * n))
        for col in range(rank_total):
            T_rec += np.einsum('i,j,k->ijk', U_total[:, col], V_total[:, col], W_total[:, col])
        err = np.linalg.norm(T_rec - T)
        
        if err < 1e-6:
            best_rank = 30
            best_U, best_V, best_W = U_total, V_total, W_total
        else:
            # Verification failed, stick with rank 42
            pass
    else:
        # Could not find rank-10, try rank 14 (two Strassens per group of 4)
        # This gives 3 * 14 = 42, same as before. Not helpful.
        # Try rank 12 per group -> total 36?
        # Or just stick with 42.
        pass
    
    # Final check and return
    if best_rank <= 42:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        return {
            "U": U42.tolist(),
            "V": V42.tolist(),
            "W": W42.tolist()
        }
# EVOLVE-BLOCK-END
