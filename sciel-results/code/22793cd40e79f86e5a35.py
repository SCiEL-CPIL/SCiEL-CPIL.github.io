import time
import numpy as np

DIMENSIONS = [2, 5, 6]
DEADLINE = 115.0  # Safety margin for 120s/150s limit

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Target tensor T is all 1s for the matrix multiplication domain
    # T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    T_target = np.ones(n * m * p)
    
    # 1. Generate Candidate Pool
    candidates = []
    
    # Type 1: Standard unit tensors
    for i in range(n):
        for j in range(m):
            for k in range(p):
                vec = np.zeros(n * m * p)
                vec[i * m * p + j * p + k] = 1.0
                u_c = {(i, j): 1}
                v_c = {(j, k): 1}
                w_c = {(k, i): 1}
                candidates.append((vec, u_c, v_c, w_c))
                
    # Type 2: Strassen 2x2x2 blocks
    def get_strassen_terms(io, jo, ko):
        terms = []
        # M1
        u = {(io, jo): 1, (io+1, jo+1): 1}
        v = {(jo, ko): 1, (jo+1, ko+1): 1}
        w = {(ko, io): 1, (ko+1, io+1): 1}
        terms.append((u, v, w))
        # M2
        u = {(io+1, jo): 1, (io+1, jo+1): 1}
        v = {(jo, ko): 1}
        w = {(ko, io+1): 1, (ko+1, io+1): -1}
        terms.append((u, v, w))
        # M3
        u = {(io, jo): 1}
        v = {(jo, ko+1): 1, (jo+1, ko+1): -1}
        w = {(ko+1, io): 1, (ko+1, io+1): 1}
        terms.append((u, v, w))
        # M4
        u = {(io+1, jo+1): 1}
        v = {(jo+1, ko): 1, (jo, ko): -1}
        w = {(ko, io): 1, (ko, io+1): 1}
        terms.append((u, v, w))
        # M5
        u = {(io, jo): 1, (io, jo+1): 1}
        v = {(jo+1, ko+1): 1}
        w = {(ko, io): -1, (ko+1, io): 1}
        terms.append((u, v, w))
        # M6
        u = {(io+1, jo): 1, (io, jo): -1}
        v = {(jo, ko): 1, (jo, ko+1): 1}
        w = {(ko+1, io+1): 1}
        terms.append((u, v, w))
        # M7
        u = {(io, jo+1): 1, (io+1, jo+1): -1}
        v = {(jo+1, ko): 1, (jo+1, ko+1): 1}
        w = {(ko, io): 1}
        terms.append((u, v, w))
        return terms

    for io in range(n - 1):
        for jo in range(m - 1):
            for ko in range(p - 1):
                for u_c, v_c, w_c in get_strassen_terms(io, jo, ko):
                    vec = np.zeros(n * m * p)
                    for (i, j), uv in u_c.items():
                        for (j2, k), vv in v_c.items():
                            if j == j2:
                                for (k2, i2), wv in w_c.items():
                                    if k == k2 and i == i2:
                                        vec[i * m * p + j * p + k] += uv * vv * wv
                    candidates.append((vec, u_c, v_c, w_c))
    
    # Type 3: Hybrid "half-Strassen" patterns (2 entries in one factor, 1 in others)
    # These are crucial for reducing rank below simple block decompositions.
    
    # 3a: U has 2 entries (same row i), V, W single
    for i in range(n):
        for j1 in range(m):
            for j2 in range(j1 + 1, m):
                for k in range(p):
                    # Pattern: (U[i,j1] + U[i,j2]) * V[j1,k] * W[k,i] -> covers (i,j1,k) and (i,j2,k)
                    # Wait, V[j1,k] only multiplies U[i,j1]. The product is U[i,j1]V[j1,k]W[k,i] + U[i,j2]V[j2,k]W[k,i]
                    # If we want a rank 1 tensor, U, V, W must be fixed vectors.
                    # A rank 1 tensor is outer product u v w.
                    # If u has support {(i,j1), (i,j2)}, v has {(j1,k)}, w has {(k,i)}.
                    # The product is non-zero only where indices match.
                    # u[i,j1]*v[j1,k]*w[k,i] -> (i,j1,k)
                    # u[i,j2]*v[j2,k]*w[k,i] -> v[j2,k] is 0 in this candidate.
                    # So this specific structure only covers 1 element, not 2.
                    # To cover 2 elements with rank 1, the supports must align.
                    # Example: U has (i,j1), (i,j2). V has (j1,k1), (j2,k2). W has (k1,i), (k2,i).
                    # This is a "diagonal" structure.
                    
                    # Let's generate "Diagonal" rank 1 tensors:
                    # U: (i,j1), (i,j2)
                    # V: (j1,k1), (j2,k2)
                    # W: (k1,i), (k2,i)
                    # This covers (i,j1,k1) and (i,j2,k2).
                    
                    # And "Off-diagonal":
                    # U: (i,j1), (i,j2)
                    # V: (j1,k2), (j2,k1)
                    # W: (k1,i), (k2,i)
                    # Covers (i,j1,k2) and (i,j2,k1).
                    
                    for k1 in range(p):
                        for k2 in range(p):
                            if k1 == k2: continue
                            # Diagonal
                            vec = np.zeros(n * m * p)
                            vec[i * m * p + j1 * p + k1] = 1.0
                            vec[i * m * p + j2 * p + k2] = 1.0
                            u_c = {(i, j1): 1, (i, j2): 1}
                            v_c = {(j1, k1): 1, (j2, k2): 1}
                            w_c = {(k1, i): 1, (k2, i): 1}
                            candidates.append((vec, u_c, v_c, w_c))
                            
                            # Off-diagonal
                            vec = np.zeros(n * m * p)
                            vec[i * m * p + j1 * p + k2] = 1.0
                            vec[i * m * p + j2 * p + k1] = 1.0
                            u_c = {(i, j1): 1, (i, j2): 1}
                            v_c = {(j1, k2): 1, (j2, k1): 1}
                            w_c = {(k1, i): 1, (k2, i): 1}
                            candidates.append((vec, u_c, v_c, w_c))

                            # Sign variations for Diagonal
                            vec = np.zeros(n * m * p)
                            vec[i * m * p + j1 * p + k1] = 1.0
                            vec[i * m * p + j2 * p + k2] = -1.0
                            u_c = {(i, j1): 1, (i, j2): 1}
                            v_c = {(j1, k1): 1, (j2, k2): 1}
                            w_c = {(k1, i): 1, (k2, i): -1}
                            candidates.append((vec, u_c, v_c, w_c))
                            
                            # Sign variations for Off-diagonal
                            vec = np.zeros(n * m * p)
                            vec[i * m * p + j1 * p + k2] = 1.0
                            vec[i * m * p + j2 * p + k1] = -1.0
                            u_c = {(i, j1): 1, (i, j2): 1}
                            v_c = {(j1, k2): 1, (j2, k1): 1}
                            w_c = {(k1, i): 1, (k2, i): -1}
                            candidates.append((vec, u_c, v_c, w_c))

    # Precompute candidate vectors and norms
    num_cands = len(candidates)
    cand_vecs = np.array([c[0] for c in candidates])
    cand_norms_sq = np.sum(cand_vecs ** 2, axis=1)
    cand_norms_sq_safe = np.where(cand_norms_sq > 1e-15, cand_norms_sq, 1.0)
    
    # Initial Greedy Selection
    residual = T_target.copy()
    selected_indices = []
    selected_coeffs = []
    
    # Precompute Gram matrix if size is manageable, otherwise compute on the fly
    # num_cands is likely a few thousand. G is (N, N). 5000x5000 is 25M floats = 200MB. Too big.
    # We will compute dots on the fly: cand_vecs @ residual.
    
    max_iter = 100
    for _ in range(max_iter):
        if time.time() - start_time > DEADLINE - 20:
            break
            
        dots = cand_vecs @ residual
        reductions = dots ** 2 / cand_norms_sq_safe
        
        best_idx = int(np.argmax(reductions))
        best_reduction = reductions[best_idx]
        
        if best_reduction < 1e-10:
            break
            
        best_coeff = dots[best_idx] / cand_norms_sq_safe[best_idx]
        
        selected_indices.append(best_idx)
        selected_coeffs.append(best_coeff)
        
        residual -= best_coeff * cand_vecs[best_idx]
        
        if np.linalg.norm(residual) < 1e-9:
            break
            
    # Local Search / Polish Pass
    # Try to improve the solution by swapping selected items with unselected ones, or removing items.
    # Goal: Reduce rank (number of selected items) while maintaining exactness.
    
    if len(selected_indices) > 0 and np.linalg.norm(residual) < 1e-8:
        # Build matrix of selected vectors
        sel_vecs = cand_vecs[selected_indices]
        sel_coeffs_arr = np.array(selected_coeffs)
        
        # Attempt to find a sparser representation using local search
        # We will iterate through selected items and try to remove them.
        # If removing one allows the rest to still span T (via least squares), we do it.
        
        current_rank = len(selected_indices)
        improved = True
        while improved and time.time() - start_time < DEADLINE - 10:
            improved = False
            for idx_in_sel in range(current_rank - 1, -1, -1):
                if time.time() - start_time > DEADLINE - 10:
                    break
                
                # Try removing idx_in_sel
                remaining_indices = [i for i in range(current_rank) if i != idx_in_sel]
                if not remaining_indices:
                    break
                    
                rem_vecs = sel_vecs[remaining_indices]
                
                try:
                    # Solve rem_vecs.T @ x = T_target
                    # This is an overdetermined system (60 equations, rank variables)
                    # We want an exact solution.
                    x_sol, residuals, rank, sv = np.linalg.lstsq(rem_vecs.T, T_target, rcond=None)
                    
                    # Check if the solution is exact
                    reconstruction = rem_vecs @ x_sol
                    if np.linalg.norm(reconstruction - T_target) < 1e-8:
                        # Success! We can drop this term.
                        # Update selected_indices and coeffs
                        new_selected_indices = [selected_indices[i] for i in remaining_indices]
                        new_selected_coeffs = x_sol.tolist()
                        
                        selected_indices = new_selected_indices
                        selected_coeffs = new_selected_coeffs
                        current_rank -= 1
                        improved = True
                        
                        # Update sel_vecs for next iteration
                        sel_vecs = cand_vecs[selected_indices]
                        
                        # Restart the loop to try removing more
                        break 
                except:
                    pass
            
            # If we didn't remove anything in this pass, try swapping?
            # Swapping is complex. Let's stick to removal first.
            # If no removal is possible, the set is likely minimal for this candidate pool.

    # Final Verification and Construction
    if len(selected_indices) > 0:
        # Reconstruct to ensure exactness
        final_vec = np.zeros(n * m * p)
        for r, idx in enumerate(selected_indices):
            final_vec += selected_coeffs[r] * cand_vecs[idx]
            
        if np.allclose(final_vec, T_target, atol=1e-8):
            rank_found = len(selected_indices)
            
            U = np.zeros((n * m, rank_found))
            V = np.zeros((m * p, rank_found))
            W = np.zeros((p * n, rank_found))
            
            for r, idx in enumerate(selected_indices):
                coeff = selected_coeffs[r]
                vec, u_c, v_c, w_c = candidates[idx]
                
                for (i, j), val in u_c.items():
                    U[i * m + j, r] = val * coeff
                for (j, k), val in v_c.items():
                    V[j * p + k, r] = val
                for (k, i), val in w_c.items():
                    W[k * n + i, r] = val
                    
            # Verify with random test
            np.random.seed(42)
            A_test = np.random.randint(-10, 10, (n, m))
            B_test = np.random.randint(-10, 10, (m, p))
            C_test = A_test @ B_test
            
            C_calc = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    val = 0
                    for j in range(m):
                        for r in range(rank_found):
                            val += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                    C_calc[i, k] = val
                    
            if np.allclose(C_test, C_calc, atol=1e-5):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: Return the known 54-rank solution
    n, m, p = DIMENSIONS
    rank_new = 54
    U_new = np.zeros((n * m, rank_new))
    V_new = np.zeros((m * p, rank_new))
    W_new = np.zeros((p * n, rank_new))
    
    current_rank = 0
    
    def add_strassen_block(i_off, j_off, k_off):
        nonlocal current_rank
        # r=0: M1
        U_new[(i_off)*m + (j_off), current_rank] = 1
        U_new[(i_off+1)*m + (j_off+1), current_rank] = 1
        V_new[(j_off)*p + (k_off), current_rank] = 1
        V_new[(j_off+1)*p + (k_off+1), current_rank] = 1
        W_new[(k_off)*n + (i_off), current_rank] = 1
        W_new[(k_off+1)*n + (i_off+1), current_rank] = 1
        current_rank += 1
        # r=1: M2
        U_new[(i_off+1)*m + (j_off), current_rank] = 1
        U_new[(i_off+1)*m + (j_off+1), current_rank] = 1
        V_new[(j_off)*p + (k_off), current_rank] = 1
        W_new[(k_off)*n + (i_off+1), current_rank] = 1
        W_new[(k_off+1)*n + (i_off+1), current_rank] = -1
        current_rank += 1
        # r=2: M3
        U_new[(i_off)*m + (j_off), current_rank] = 1
        V_new[(j_off)*p + (k_off+1), current_rank] = 1
        V_new[(j_off+1)*p + (k_off+1), current_rank] = -1
        W_new[(k_off+1)*n + (i_off), current_rank] = 1
        W_new[(k_off+1)*n + (i_off+1), current_rank] = 1
        current_rank += 1
        # r=3: M4
        U_new[(i_off+1)*m + (j_off+1), current_rank] = 1
        V_new[(j_off+1)*p + (k_off), current_rank] = 1
        V_new[(j_off)*p + (k_off), current_rank] = -1
        W_new[(k_off)*n + (i_off), current_rank] = 1
        W_new[(k_off)*n + (i_off+1), current_rank] = 1
        current_rank += 1
        # r=4: M5
        U_new[(i_off)*m + (j_off), current_rank] = 1
        U_new[(i_off)*m + (j_off+1), current_rank] = 1
        V_new[(j_off+1)*p + (k_off+1), current_rank] = 1
        W_new[(k_off)*n + (i_off), current_rank] = -1
        W_new[(k_off+1)*n + (i_off), current_rank] = 1
        current_rank += 1
        # r=5: M6
        U_new[(i_off+1)*m + (j_off), current_rank] = 1
        U_new[(i_off)*m + (j_off), current_rank] = -1
        V_new[(j_off)*p + (k_off), current_rank] = 1
        V_new[(j_off)*p + (k_off+1), current_rank] = 1
        W_new[(k_off+1)*n + (i_off+1), current_rank] = 1
        current_rank += 1
        # r=6: M7
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

    # Part 1: A[:, 0:2] * B[0:2, :]
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)
    
    # Part 2a: A[:, 2:3] * B[2:3, :]
    for k in range(6):
        add_standard_block(0, 2, k)
        
    # Part 2b: A[:, 3:5] * B[3:5, :]
    add_strassen_block(0, 3, 0)
    add_strassen_block(0, 3, 2)
    add_strassen_block(0, 3, 4)
    
    return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
