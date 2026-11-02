import time
import numpy as np
from collections import defaultdict

DEADLINE = 115.0

def construct():
    start_time = time.time()
    n, m, p = 2, 5, 6
    total_entries = n * m * p  # 60
    
    # Build a large pool of candidate rank-1 tensors
    # A rank-1 tensor is (u_vec, v_vec, w_vec) where:
    #   u_vec has length n*m, v_vec has length m*p, w_vec has length p*n
    # and the contribution to the (i,j,k) entry is u[i*m+j] * v[j*p+k] * w[k*n+i]
    
    candidates = []  # list of (u_vec, v_vec, w_vec, contrib_vec)
    
    def add_candidate(u_sparse, v_sparse, w_sparse):
        """Add a candidate defined by sparse dicts."""
        u_vec = np.zeros(n * m)
        v_vec = np.zeros(m * p)
        w_vec = np.zeros(p * n)
        
        for (i, j), val in u_sparse.items():
            u_vec[i * m + j] = val
        for (j, k), val in v_sparse.items():
            v_vec[j * p + k] = val
        for (k, i), val in w_sparse.items():
            w_vec[k * n + i] = val
        
        # Compute contribution vector
        contrib = np.zeros(total_entries)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    contrib[i * m * p + j * p + k] = u_vec[i*m+j] * v_vec[j*p+k] * w_vec[k*n+i]
        
        # Only add if non-zero
        if np.any(contrib != 0):
            candidates.append((u_vec, v_vec, w_vec, contrib))
    
    # Type 1: Standard unit tensors
    for i in range(n):
        for j in range(m):
            for k in range(p):
                add_candidate({(i,j): 1}, {(j,k): 1}, {(k,i): 1})
    
    # Type 2: Strassen 2x2x2 blocks
    def strassen_terms(io, jo, ko):
        terms = []
        terms.append(({(io,jo): 1, (io+1,jo+1): 1}, 
                      {(jo,ko): 1, (jo+1,ko+1): 1}, 
                      {(ko,io): 1, (ko+1,io+1): 1}))
        terms.append(({(io+1,jo): 1, (io+1,jo+1): 1}, 
                      {(jo,ko): 1}, 
                      {(ko,io+1): 1, (ko+1,io+1): -1}))
        terms.append(({(io,jo): 1}, 
                      {(jo,ko+1): 1, (jo+1,ko+1): -1}, 
                      {(ko+1,io): 1, (ko+1,io+1): 1}))
        terms.append(({(io+1,jo+1): 1}, 
                      {(jo+1,ko): 1, (jo,ko): -1}, 
                      {(ko,io): 1, (ko,io+1): 1}))
        terms.append(({(io,jo): 1, (io,jo+1): 1}, 
                      {(jo+1,ko+1): 1}, 
                      {(ko,io): -1, (ko+1,io): 1}))
        terms.append(({(io+1,jo): 1, (io,jo): -1}, 
                      {(jo,ko): 1, (jo,ko+1): 1}, 
                      {(ko+1,io+1): 1}))
        terms.append(({(io,jo+1): 1, (io+1,jo+1): -1}, 
                      {(jo+1,ko): 1, (jo+1,ko+1): 1}, 
                      {(ko,io): 1}))
        return terms
    
    for io in range(n - 1):
        for jo in range(m - 1):
            for ko in range(p - 1):
                for u_c, v_c, w_c in strassen_terms(io, jo, ko):
                    add_candidate(u_c, v_c, w_c)
    
    # Type 3: Additional diverse candidates
    # 2x2x3 blocks - try to cover 3 columns with 2 rows
    # For each pair of rows (always 0,1), and 3 consecutive columns
    # We can use a combination of Strassen-like terms
    
    # Also add some "cross" terms that mix different j's
    for io in range(n):
        for j1 in range(m):
            for j2 in range(m):
                if j1 == j2: continue
                for k in range(p):
                    # (A[i,j1] + A[i,j2]) * B[j2,k] * W
                    add_candidate({(io,j1): 1, (io,j2): 1}, {(j2,k): 1}, {(k,io): 1})
                    # (A[i,j1] - A[i,j2]) * B[j2,k] * W
                    add_candidate({(io,j1): 1, (io,j2): -1}, {(j2,k): 1}, {(k,io): 1})
    
    # More structured candidates: 2x3x2 blocks
    def strassen_2x3x2_terms(io, jo, ko):
        """7+ terms for 2x3x2 blocks (2 rows, 3 cols, 2 output cols)"""
        terms = []
        # i: io, io+1; j: jo, jo+1, jo+2; k: ko, ko+1
        
        # Use Strassen on (io,io+1) x (jo,jo+1) x (ko,ko+1)
        for u_c, v_c, w_c in strassen_terms(io, jo, ko):
            terms.append((u_c, v_c, w_c))
        
        # Plus standard terms for the third column jo+2
        terms.append(({(io, jo+2): 1}, {(jo+2, ko): 1}, {(ko, io): 1}))
        terms.append(({(io+1, jo+2): 1}, {(jo+2, ko): 1}, {(ko, io+1): 1}))
        terms.append(({(io, jo+2): 1}, {(jo+2, ko+1): 1}, {(ko+1, io): 1}))
        terms.append(({(io+1, jo+2): 1}, {(jo+2, ko+1): 1}, {(ko+1, io+1): 1}))
        
        return terms
    
    for io in range(n - 1):
        for jo in range(m - 2):
            for ko in range(p - 1):
                for u_c, v_c, w_c in strassen_2x3x2_terms(io, jo, ko):
                    add_candidate(u_c, v_c, w_c)
    
    # Now use greedy to find a good decomposition
    T_target = np.ones(total_entries)
    residual = T_target.copy()
    
    selected_indices = []
    selected_coeffs = []
    
    num_candidates = len(candidates)
    
    # Precompute all contribution vectors as a matrix
    contrib_matrix = np.zeros((num_candidates, total_entries))
    for idx in range(num_candidates):
        contrib_matrix[idx] = candidates[idx][3]
    
    # Precompute norms
    norms = np.sum(contrib_matrix ** 2, axis=1)
    
    # Precompute Gram matrix (candidate dot products)
    if num_candidates <= 5000:
        Gram = contrib_matrix @ contrib_matrix.T
    else:
        Gram = None
    
    max_iter = 200
    for _ in range(max_iter):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        best_idx = -1
        best_reduction = 0
        best_coeff = 0
        
        if Gram is not None:
            dots = contrib_matrix @ residual
            valid = norms > 1e-12
            reductions = np.where(valid, dots**2 / np.where(norms>1e-12, norms, 1), 0)
            best_idx = int(np.argmax(reductions))
            best_reduction = reductions[best_idx]
            if best_reduction < 1e-9:
                break
            best_coeff = dots[best_idx] / norms[best_idx]
        else:
            for idx in range(num_candidates):
                if norms[idx] < 1e-12:
                    continue
                dot_rc = np.dot(residual, contrib_matrix[idx])
                coeff = dot_rc / norms[idx]
                reduction = (dot_rc ** 2) / norms[idx]
                
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_idx = idx
                    best_coeff = coeff
        
        if best_idx == -1 or best_reduction < 1e-9:
            break
        
        selected_indices.append(best_idx)
        selected_coeffs.append(best_coeff)
        residual -= best_coeff * contrib_matrix[best_idx]
        
        if np.linalg.norm(residual) < 1e-9:
            break
    
    # Rank reduction: find linear dependencies among selected terms
    def try_reduce_rank(sel_indices, sel_coeffs, U_arr, V_arr, W_arr):
        """Try to find linear dependencies among the rank-1 terms to reduce rank."""
        r = len(sel_indices)
        if r < 2:
            return None
        
        U_sel = U_arr.copy()
        V_sel = V_arr.copy()
        W_sel = W_arr.copy()
        
        n_entries = n * m * p
        tensor_matrix = np.zeros((n_entries, r))
        for col in range(r):
            idx = sel_indices[col]
            u_vec, v_vec, w_vec, _ = candidates[idx]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        tensor_matrix[i*m*p + j*p + k, col] = u_vec[i*m+j] * v_vec[j*p+k] * w_vec[k*n+i]
        
        try:
            U_sv, s_vals, Vt = np.linalg.svd(tensor_matrix, full_matrices=False)
            tol = 1e-10 * (s_vals[0] if len(s_vals) > 0 else 1)
            null_mask = s_vals < tol
            if not np.any(null_mask):
                return None
            
            null_space = Vt[null_mask].T
            
            best_reduced = None
            for ns_vec in null_space:
                max_idx = int(np.argmax(np.abs(ns_vec)))
                if abs(ns_vec[max_idx]) < 1e-10:
                    continue
                
                new_coeffs = list(sel_coeffs)
                c_max = new_coeffs[max_idx]
                for t in range(r):
                    if t != max_idx:
                        new_coeffs[t] = new_coeffs[t] - c_max * ns_vec[t] / ns_vec[max_idx]
                
                new_r = r - 1
                U_new = np.zeros((n * m, new_r))
                V_new = np.zeros((m * p, new_r))
                W_new = np.zeros((p * n, new_r))
                
                col = 0
                for t in range(r):
                    if t == max_idx:
                        continue
                    U_new[:, col] = U_sel[:, t]
                    V_new[:, col] = V_sel[:, t]
                    W_new[:, col] = W_sel[:, t]
                    col += 1
                
                A_test = np.random.randint(-10, 10, (n, m))
                B_test = np.random.randint(-10, 10, (m, p))
                C_test = A_test @ B_test
                
                C_calc = np.zeros((n, p))
                for i in range(n):
                    for k in range(p):
                        val = 0
                        for j in range(m):
                            for rr in range(new_r):
                                val += U_new[i*m+j, rr] * V_new[j*p+k, rr] * W_new[k*n+i, rr]
                        C_calc[i, k] = val
                
                if np.allclose(C_test, C_calc, atol=1e-5):
                    if best_reduced is None or new_r < best_reduced[0]:
                        best_reduced = (new_r, U_new, V_new, W_new)
            
            return best_reduced
        except np.linalg.LinAlgError:
            return None
    
    # Check if we got an exact solution
    if np.linalg.norm(residual) < 1e-9:
        rank_found = len(selected_indices)
        
        U = np.zeros((n * m, rank_found))
        V = np.zeros((m * p, rank_found))
        W = np.zeros((p * n, rank_found))
        
        for r, idx in enumerate(selected_indices):
            coeff = selected_coeffs[r]
            u_vec, v_vec, w_vec, _ = candidates[idx]
            U[:, r] = u_vec * coeff
            V[:, r] = v_vec
            W[:, r] = w_vec
        
        reduced = try_reduce_rank(selected_indices, selected_coeffs, U, V, W)
        if reduced is not None:
            new_rank, U, V, W = reduced
            rank_found = new_rank
        
        A_test = np.random.randint(-10, 10, (n, m))
        B_test = np.random.randint(-10, 10, (m, p))
        C_test = A_test @ B_test
        
        C_calc = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                val = 0
                for j in range(m):
                    for r in range(rank_found):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                C_calc[i, k] = val
        
        if np.allclose(C_test, C_calc, atol=1e-5):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: return the known 54-rank solution
    n, m, p = 2, 5, 6
    rank_new = 54
    U_new = np.zeros((n * m, rank_new))
    V_new = np.zeros((m * p, rank_new))
    W_new = np.zeros((p * n, rank_new))
    
    current_rank = 0
    
    def add_strassen_block(i_off, j_off, k_off):
        nonlocal current_rank
        U_new[(i_off)*m + (j_off), current_rank] = 1
        U_new[(i_off+1)*m + (j_off+1), current_rank] = 1
        V_new[(j_off)*p + (k_off), current_rank] = 1
        V_new[(j_off+1)*p + (k_off+1), current_rank] = 1
        W_new[(k_off)*n + (i_off), current_rank] = 1
        W_new[(k_off+1)*n + (i_off+1), current_rank] = 1
        current_rank += 1
        U_new[(i_off+1)*m + (j_off), current_rank] = 1
        U_new[(i_off+1)*m + (j_off+1), current_rank] = 1
        V_new[(j_off)*p + (k_off), current_rank] = 1
        W_new[(k_off)*n + (i_off+1), current_rank] = 1
        W_new[(k_off+1)*n + (i_off+1), current_rank] = -1
        current_rank += 1
        U_new[(i_off)*m + (j_off), current_rank] = 1
        V_new[(j_off)*p + (k_off+1), current_rank] = 1
        V_new[(j_off+1)*p + (k_off+1), current_rank] = -1
        W_new[(k_off+1)*n + (i_off), current_rank] = 1
        W_new[(k_off+1)*n + (i_off+1), current_rank] = 1
        current_rank += 1
        U_new[(i_off+1)*m + (j_off+1), current_rank] = 1
        V_new[(j_off+1)*p + (k_off), current_rank] = 1
        V_new[(j_off)*p + (k_off), current_rank] = -1
        W_new[(k_off)*n + (i_off), current_rank] = 1
        W_new[(k_off)*n + (i_off+1), current_rank] = 1
        current_rank += 1
        U_new[(i_off)*m + (j_off), current_rank] = 1
        U_new[(i_off)*m + (j_off+1), current_rank] = 1
        V_new[(j_off+1)*p + (k_off+1), current_rank] = 1
        W_new[(k_off)*n + (i_off), current_rank] = -1
        W_new[(k_off+1)*n + (i_off), current_rank] = 1
        current_rank += 1
        U_new[(i_off+1)*m + (j_off), current_rank] = 1
        U_new[(i_off)*m + (j_off), current_rank] = -1
        V_new[(j_off)*p + (k_off), current_rank] = 1
        V_new[(j_off)*p + (k_off+1), current_rank] = 1
        W_new[(k_off+1)*n + (i_off+1), current_rank] = 1
        current_rank += 1
        U_new[(i_off)*m + (j_off+1), current_rank] = 1
        U_new[(i_off+1)*m + (j_off+1), current_rank] = -1
        V_new[(j_off+1)*p + (k_off), current_rank] = 1
        V_new[(j_off+1)*p + (k_off+1), current_rank] = 1
        W_new[(k_off)*n + (i_off), current_rank] = 1
        current_rank += 1

    def add_standard_block(i_off, j_off, k_off):
        nonlocal current_rank
        U_new[(i_off)*m + (j_off), current_rank] = 1
        V_new[(j_off)*p + (k_off), current_rank] = 1
        W_new[(k_off)*n + (i_off), current_rank] = 1
        current_rank += 1
        U_new[(i_off+1)*m + (j_off), current_rank] = 1
        V_new[(j_off)*p + (k_off), current_rank] = 1
        W_new[(k_off)*n + (i_off+1), current_rank] = 1
        current_rank += 1

    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)
    
    for k in range(6):
        add_standard_block(0, 2, k)
        
    add_strassen_block(0, 3, 0)
    add_strassen_block(0, 3, 2)
    add_strassen_block(0, 3, 4)
    
    return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
