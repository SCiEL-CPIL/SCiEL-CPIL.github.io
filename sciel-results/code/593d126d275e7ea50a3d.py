import numpy as np
import time

DIMENSIONS = [4, 5, 5]
_START = time.time()
_DEADLINE = 120.0  # seconds

def construct():
    n, m, p = DIMENSIONS  # n=4, m=5, p=5
    
    # Strassen 2x2x2 factors
    s_u = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    s_v = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    s_w = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # 4x4 Strassen terms (outer level)
    outer_terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_rank1(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    # Build 4x4x4 part using nested Strassen
    for t_idx, (a_coeffs, b_coeffs, c_coeffs) in enumerate(outer_terms):
        for s_idx in range(7):
            u_vec = np.zeros(n * m)
            v_vec = np.zeros(m * p)
            w_vec = np.zeros(p * n)
            
            for (ar, ac), a_val in a_coeffs.items():
                for sr in range(2):
                    for sc in range(2):
                        s_u_val = s_u[sr * 2 + sc][s_idx]
                        if s_u_val != 0:
                            gi = 2 * ar + sr
                            gk = 2 * ac + sc
                            u_vec[gi * m + gk] += a_val * s_u_val
            
            for (br, bc), b_val in b_coeffs.items():
                for sr in range(2):
                    for sc in range(2):
                        s_v_val = s_v[sr * 2 + sc][s_idx]
                        if s_v_val != 0:
                            gk = 2 * br + sr
                            gj = 2 * bc + sc
                            v_vec[gk * p + gj] += b_val * s_v_val
            
            for (cr, cc), c_val in c_coeffs.items():
                for sr in range(2):
                    for sc in range(2):
                        s_w_val = s_w[sr * 2 + sc][s_idx]
                        if s_w_val != 0:
                            gi = 2 * cr + sr
                            gj = 2 * cc + sc
                            w_vec[gj * n + gi] += c_val * s_w_val
            
            add_rank1(u_vec, v_vec, w_vec)
    
    # Part 2: 4x4x1 (k in 0..3, j=4) - schoolbook (16 terms)
    def add_term(i, k, j, coeff_u=1.0, coeff_v=1.0, coeff_w=1.0):
        u_vec = np.zeros(n * m)
        v_vec = np.zeros(m * p)
        w_vec = np.zeros(p * n)
        u_vec[i * m + k] = coeff_u
        v_vec[k * p + j] = coeff_v
        w_vec[j * n + i] = coeff_w
        add_rank1(u_vec, v_vec, w_vec)
    
    for i in range(4):
        for k in range(4):
            add_term(i, k, 4, 1.0, 1.0, 1.0)
    
    # Part 3: 4x1x5 (k=4, j in 0..4) - schoolbook (20 terms)
    for i in range(4):
        for j in range(5):
            add_term(i, 4, j, 1.0, 1.0, 1.0)
    
    rank = len(U_cols)
    
    if rank == 0:
        add_term(0, 0, 0, 1.0, 1.0, 1.0)
        rank = 1
    
    # Efficient linear dependence removal using vectorized verification
    if len(U_cols) > 1 and time.time() - _START < _DEADLINE - 30:
        U_mat = np.array(U_cols, dtype=float).T  # shape (n*m, rank)
        V_mat = np.array(V_cols, dtype=float).T  # shape (m*p, rank)
        W_mat = np.array(W_cols, dtype=float).T  # shape (p*n, rank)
        
        # Precompute random test matrices
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        # For each term, precompute its contribution to C
        # Term r contributes: sum_{i,j,k} U[r,i*m+k] * V[r,k*p+j] * W[r,j*n+i] * e_{i,j}
        # Vectorized: for each r, C_r = reshape from U[r], V[r], W[r]
        
        # Precompute U_flat, V_flat, W_flat as 3D arrays
        U_3d = U_mat.reshape(n, m, rank)  # U_3d[i,k,r]
        V_3d = V_mat.reshape(m, p, rank)  # V_3d[k,j,r]
        W_3d = W_mat.reshape(p, n, rank)  # W_3d[j,i,r]
        
        # For each term r, compute C_r[i,j] = sum_k U_3d[i,k,r] * V_3d[k,j,r] * W_3d[j,i,r]
        # This is a rank-1 contribution per term
        
        # Precompute all term contributions
        C_terms = np.zeros((n, p, rank))
        for r in range(rank):
            for i in range(n):
                for j in range(p):
                    C_terms[i, j, r] = np.sum(U_3d[i, :, r] * V_3d[:, j, r]) * W_3d[j, i, r]
        
        # Now find linear dependencies: we want to find coefficients c_r such that
        # sum_r c_r * C_terms[:, :, r] = 0 (as an n×p matrix)
        # This is equivalent to: sum_r c_r * vec(C_terms[:, :, r]) = 0 in R^{n*p}
        
        C_vec = C_terms.reshape(n * p, rank)  # shape (n*p, rank)
        
        # Find null space of C_vec
        if rank > n * p:
            C_svd, C_svals, C_Vt = np.linalg.svd(C_vec, full_matrices=False)
            tol = 1e-9
            num_indep = np.sum(C_svals > tol)
            
            if num_indep < rank:
                null_vectors = C_Vt[num_indep:]  # shape (rank-num_indep, rank)
                
                removed = set()
                for ns_vec in null_vectors:
                    if time.time() - _START > _DEADLINE - 10:
                        break
                    candidates = [i for i in range(rank) if i not in removed]
                    if not candidates:
                        break
                    abs_vals = np.array([abs(ns_vec[i]) for i in candidates])
                    best_local = np.argmax(abs_vals)
                    drop_idx = candidates[best_local]
                    
                    if abs(ns_vec[drop_idx]) < 1e-10:
                        continue
                    
                    # Verify by removing and checking
                    new_indices = [i for i in range(rank) if i != drop_idx and i not in removed]
                    C_computed = np.sum(C_vec[:, new_indices] * ns_vec[drop_idx] / ns_vec[drop_idx], axis=1) if False else None
                    
                    # Quick check: recompute C from remaining terms
                    C_check = np.sum(C_vec[:, new_indices], axis=1)  # This isn't quite right
                    # Actually, we need to verify the full decomposition still works
                    # Let's just verify by reconstructing
                    C_reconstructed = np.zeros(n * p)
                    for idx in new_indices:
                        C_reconstructed += C_vec[:, idx]
                    
                    # The sum of all terms should equal C_expected
                    # If we remove a term that's in the null space, the sum should still be C_expected
                    # But this isn't quite right either. Let me think...
                    
                    # Actually, if c is in the null space, then sum_r c_r * C_vec[:, r] = 0
                    # So if we have original sum = C_expected, and we subtract c_r * C_vec[:, r] for the removed term,
                    # we need to adjust the other terms. This is getting complicated.
                    
                    # Simpler approach: just try removing and verify with a fresh computation
                    C_fresh = np.zeros((n, p))
                    for idx in new_indices:
                        u_flat = U_cols[idx]
                        v_flat = V_cols[idx]
                        w_flat = W_cols[idx]
                        # Vectorized computation
                        U_reshaped = u_flat.reshape(n, m)
                        V_reshaped = v_flat.reshape(m, p)
                        W_reshaped = w_flat.reshape(p, n)
                        # Contribution: (U_reshaped @ V_reshaped) * W_reshaped (element-wise)
                        partial = U_reshaped @ V_reshaped
                        C_fresh += partial * W_reshaped
                    
                    if np.allclose(C_fresh, C_expected, atol=1e-6):
                        removed.add(drop_idx)
                
                if removed:
                    keep_indices = [i for i in range(rank) if i not in removed]
                    U_cols = [U_cols[i] for i in keep_indices]
                    V_cols = [V_cols[i] for i in keep_indices]
                    W_cols = [W_cols[i] for i in keep_indices]
                    rank = len(U_cols)
    
    U = np.stack(U_cols, axis=1)
    V = np.stack(V_cols, axis=1)
    W = np.stack(W_cols, axis=1)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
