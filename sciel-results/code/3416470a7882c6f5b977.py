import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = start_time + 145
    n, m, p = 3, 4, 7

    U_vecs = []
    V_vecs = []
    W_vecs = []

    def add_rank1_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        u = np.zeros(n * m)
        v = np.zeros(m * p)
        w = np.zeros(p * n)
        for idx, val in zip(u_indices, u_vals):
            u[idx] += val
        for idx, val in zip(v_indices, v_vals):
            v[idx] += val
        for idx, val in zip(w_indices, w_vals):
            w[idx] += val
        U_vecs.append(u)
        V_vecs.append(v)
        W_vecs.append(w)

    def strassen_block(i0, j0, k0):
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1
        add_rank1_term([i0*m+j0, i1*m+j1], [1, 1],
                       [j0*p+k0, j1*p+k1], [1, 1],
                       [k0*n+i0, k1*n+i1], [1, 1])
        add_rank1_term([i1*m+j0, i1*m+j1], [1, 1],
                       [j0*p+k0], [1],
                       [k0*n+i1, k1*n+i1], [1, -1])
        add_rank1_term([i0*m+j0], [1],
                       [j0*p+k1, j1*p+k1], [1, -1],
                       [k1*n+i0, k1*n+i1], [1, 1])
        add_rank1_term([i1*m+j1], [1],
                       [j1*p+k0, j0*p+k0], [1, -1],
                       [k0*n+i0, k0*n+i1], [1, 1])
        add_rank1_term([i0*m+j0, i0*m+j1], [1, 1],
                       [j1*p+k1], [1],
                       [k0*n+i0, k1*n+i0], [-1, 1])
        add_rank1_term([i1*m+j0, i0*m+j0], [1, -1],
                       [j0*p+k0, j0*p+k1], [1, 1],
                       [k1*n+i1], [1])
        add_rank1_term([i0*m+j1, i1*m+j1], [1, -1],
                       [j1*p+k0, j1*p+k1], [1, 1],
                       [k0*n+i0], [1])

    def gen_terms(n_l, m_l, p_l, i_off, j_off, k_off):
        if n_l == 1 and m_l == 1 and p_l == 1:
            i = i_off; j = j_off; k = k_off
            add_rank1_term([i*m+j], [1], [j*p+k], [1], [k*n+i], [1])
            return
        if n_l == 1:
            for j in range(m_l):
                for k in range(p_l):
                    ii = i_off; jj = j_off + j; kk = k_off + k
                    add_rank1_term([ii*m+jj], [1], [jj*p+kk], [1], [kk*n+ii], [1])
            return
        if m_l == 1:
            for i in range(n_l):
                for k in range(p_l):
                    ii = i_off + i; j = j_off; kk = k_off + k
                    add_rank1_term([ii*m+j], [1], [j*p+kk], [1], [kk*n+ii], [1])
            return
        if p_l == 1:
            for i in range(n_l):
                for j in range(m_l):
                    ii = i_off + i; jj = j_off + j; k = k_off
                    add_rank1_term([ii*m+jj], [1], [jj*p+k], [1], [k*n+ii], [1])
            return
        if n_l == 2 and m_l == 2 and p_l == 2:
            strassen_block(i_off, j_off, k_off)
            return
        if n_l == 3 and m_l == 4 and p_l == 7:
            gen_terms(2, 4, 7, i_off, j_off, k_off)
            gen_terms(1, 4, 7, i_off + 2, j_off, k_off)
            return
        if n_l == 2 and m_l == 4 and p_l == 7:
            gen_terms(2, 2, 7, i_off, j_off, k_off)
            gen_terms(2, 2, 7, i_off, j_off + 2, k_off)
            return
        if n_l == 2 and m_l == 2 and p_l == 7:
            gen_terms(2, 2, 2, i_off, j_off, k_off)
            gen_terms(2, 2, 2, i_off, j_off, k_off + 2)
            gen_terms(2, 2, 2, i_off, j_off, k_off + 4)
            gen_terms(2, 2, 1, i_off, j_off, k_off + 6)
            return
        for i in range(n_l):
            for j in range(m_l):
                for k in range(p_l):
                    ii = i_off + i; jj = j_off + j; kk = k_off + k
                    add_rank1_term([ii*m+jj], [1], [jj*p+kk], [1], [kk*n+ii], [1])

    gen_terms(n, m, p, 0, 0, 0)

    U = np.array(U_vecs).T
    V = np.array(V_vecs).T
    W = np.array(W_vecs).T
    rank = U.shape[1]

    def full_tensor(U_mat, V_mat, W_mat):
        U_3d = U_mat.reshape(n, m, -1)
        V_3d = V_mat.reshape(m, p, -1)
        W_3d = W_mat.reshape(p, n, -1)
        return np.einsum('ijt,jkt,kit->ijk', U_3d, V_3d, W_3d)

    target = np.ones((n, m, p))

    # Verify initial decomposition
    try:
        full_T = full_tensor(U, V, W)
        if not np.allclose(full_T, target, atol=1e-10):
            U_vecs.clear(); V_vecs.clear(); W_vecs.clear()
            for ii in range(n):
                for jj in range(m):
                    for kk in range(p):
                        add_rank1_term([ii*m+jj], [1], [jj*p+kk], [1], [kk*n+ii], [1])
            U = np.array(U_vecs).T; V = np.array(V_vecs).T; W = np.array(W_vecs).T
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        U_vecs.clear(); V_vecs.clear(); W_vecs.clear()
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    add_rank1_term([ii*m+jj], [1], [jj*p+kk], [1], [kk*n+ii], [1])
        U = np.array(U_vecs).T; V = np.array(V_vecs).T; W = np.array(W_vecs).T
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Peeling: remove redundant rank-1 components
    if time.time() < DEADLINE - 35 and rank > 20:
        removed = set()
        current_rank = rank
        full_T = full_tensor(U, V, W)
        for pass_num in range(3):
            if time.time() > DEADLINE - 30:
                break
            any_removed = False
            for t in range(current_rank):
                if time.time() > DEADLINE - 30:
                    break
                if t in removed:
                    continue
                u_t = U[:, t]; v_t = V[:, t]; w_t = W[:, t]
                contrib = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            contrib[i, j, k] = u_t[i*m+j] * v_t[j*p+k] * w_t[k*n+i]
                residual = full_T - contrib
                if np.allclose(residual, target, atol=1e-10):
                    removed.add(t)
                    any_removed = True
            if not any_removed:
                break
        if removed:
            final_mask = [i for i in range(rank) if i not in removed]
            U = U[:, final_mask]; V = V[:, final_mask]; W = W[:, final_mask]
            rank = U.shape[1]

    # Embedding and projection: try to reduce rank via global linear projection
    if time.time() < DEADLINE - 15 and rank > 25:
        try:
            U_3d = U.reshape(n, m, rank)
            V_3d = V.reshape(m, p, rank)
            W_3d = W.reshape(p, n, rank)

            # Build the linear system: find P (rank x rank) such that
            # sum_r U_3d[:,:,r] * (P @ e_r) outer ... 
            # Actually, a global projection P applied to the rank index:
            # U' = U @ P, V' = V @ P^{-1}, W' = W (if P is invertible)
            # This preserves the tensor. But to REDUCE rank, we need P to have
            # rank < current_rank, which means we project onto a subspace.
            # 
            # Alternative: find a lower-rank decomposition by solving
            # for new factors with fewer columns.
            # 
            # Simpler approach: try to find linear dependencies among the
            # rank-1 terms. If term t can be expressed as a linear combination
            # of others in a way that preserves the tensor, we can eliminate it.
            #
            # Most practical: try random projections and check if they work.
            
            best_rank = rank
            best_U, best_V, best_W = U, V, W
            
            # Try projecting to rank-1 subspace: find a vector a such that
            # sum_r a_r * (U[:,:,r] * V[:,:,r] * W[:,:,r]) still equals target
            # This is a different kind of reduction - not standard.
            
            # Instead: try to find if any single term is a linear combination
            # of others when projected through the tensor structure.
            # For each pair of terms (a, b), check if removing a and adjusting b works.
            
            # More practical: use the fact that if two terms have proportional
            # U, V, or W vectors, they can be merged.
            for t in range(rank):
                if time.time() > DEADLINE - 15:
                    break
                u_t = U[:, t]
                # Check if u_t is in the span of other U columns
                others = [i for i in range(rank) if i != t]
                if len(others) < 2:
                    continue
                U_others = U[:, others]
                # Try to express u_t as linear combination of U_others
                try:
                    coeffs, residuals, rank_s, sv = np.linalg.lstsq(U_others, u_t, rcond=None)
                    if residuals and residuals[0] < 1e-10:
                        # u_t is in span of others. Now check if the tensor is preserved.
                        # We need: sum_r U_r V_r W_r = target
                        # If we remove term t and adjust others, we need:
                        # sum_{r != t} (U_r + c_r U_t) V_r W_r + ... = target
                        # This is complex. Skip for now.
                        pass
                except Exception:
                    pass
            
            # Try a simpler approach: Gaussian elimination on the rank index
            # to find a basis of lower dimension that still spans the tensor.
            # This is essentially what peeling does but more aggressive.
            
            # Most promising: try to reduce by 1 using a random direction.
            # Generate random vector r of length rank, compute new factors:
            # U' = U @ (I - r * r^T / ||r||^2)  (project out r direction)
            # This won't work directly. 
            
            # Instead: find if there's a linear relation sum_r alpha_r T_r = 0
            # where T_r = U_r * V_r * W_r (as tensor). If so, we can adjust
            # the coefficients to eliminate one term.
            
            # For rank R, we have R rank-1 tensors. The tensor space has
            # n*m*p = 84 dimensions. If R > 84, there must be a linear relation.
            # But R ~ 74 < 84, so no guaranteed relation.
            
            # However, we can still try: for each term t, check if T_t is in
            # the span of the other T_r's.
            flat_target = target.flatten()
            
            for t in range(rank):
                if time.time() > DEADLINE - 15:
                    break
                # Build the tensor for term t
                T_t = np.zeros(n * m * p)
                u_t = U[:, t]; v_t = V[:, t]; w_t = W[:, t]
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_t[i*m*p + j*p + k] = u_t[i*m+j] * v_t[j*p+k] * w_t[k*n+i]
                
                if np.allclose(T_t, 0, atol=1e-12):
                    # This term contributes nothing - remove it
                    mask = [i for i in range(rank) if i != t]
                    U = U[:, mask]; V = V[:, mask]; W = W[:, mask]
                    rank = U.shape[1]
                    best_rank = rank
                    best_U, best_V, best_W = U, V, W
                    break
                
                # Check if T_t is in span of others
                others = [i for i in range(rank) if i != t]
                if len(others) == 0:
                    continue
                T_others = np.zeros((n*m*p, len(others)))
                for idx, oi in enumerate(others):
                    u_o = U[:, oi]; v_o = V[:, oi]; w_o = W[:, oi]
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                T_others[i*m*p + j*p + k, idx] = u_o[i*m+j] * v_o[j*p+k] * w_o[k*n+i]
                
                try:
                    coeffs, residuals, rank_s, sv = np.linalg.lstsq(T_others, T_t, rcond=None)
                    if residuals and residuals[0] < 1e-8:
                        # T_t is in span of others. We can potentially remove it
                        # by adjusting the coefficients of others.
                        # New decomposition: sum_{r != t} (1 + alpha_r) T_r + ... 
                        # Actually: T_t = sum_{r != t} c_r T_r
                        # So: sum_r T_r = sum_{r != t} T_r + T_t = sum_{r != t} (1 + c_r) T_r
                        # This means we can merge T_t into the others!
                        # But we need to update the factors, not just the tensor.
                        # The factors U, V, W don't combine linearly this way.
                        # So this only works if we're willing to re-derive factors.
                        pass
                except Exception:
                    pass
        except Exception:
            pass

    # Final verification
    try:
        U_3d = U.reshape(n, m, -1)
        V_3d = V.reshape(m, p, -1)
        W_3d = W.reshape(p, n, -1)
        T = np.einsum('ijt,jkt,kit->ijk', U_3d, V_3d, W_3d)
        if not np.allclose(T, 1.0, atol=1e-10):
            U_vecs.clear(); V_vecs.clear(); W_vecs.clear()
            for ii in range(n):
                for jj in range(m):
                    for kk in range(p):
                        add_rank1_term([ii*m+jj], [1], [jj*p+kk], [1], [kk*n+ii], [1])
            U = np.array(U_vecs).T; V = np.array(V_vecs).T; W = np.array(W_vecs).T
    except Exception:
        U_vecs.clear(); V_vecs.clear(); W_vecs.clear()
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    add_rank1_term([ii*m+jj], [1], [jj*p+kk], [1], [kk*n+ii], [1])
        U = np.array(U_vecs).T; V = np.array(V_vecs).T; W = np.array(W_vecs).T

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
