import time
import numpy as np
from itertools import product

DEADLINE = 115.0

def construct():
    start_time = time.time()
    n, m, p = 2, 5, 6
    total_entries = n * m * p  # 60
    
    # Target tensor T is all 1s
    T_target = np.ones(total_entries)
    
    candidates = []  # List of (u_vec, v_vec, w_vec, contrib_vec)
    
    def add_candidate(u_sparse, v_sparse, w_sparse):
        u_vec = np.zeros(n * m)
        v_vec = np.zeros(m * p)
        w_vec = np.zeros(p * n)
        
        for (i, j), val in u_sparse.items():
            u_vec[i * m + j] = val
        for (j, k), val in v_sparse.items():
            v_vec[j * p + k] = val
        for (k, i), val in w_sparse.items():
            w_vec[k * n + i] = val
        
        contrib = np.zeros(total_entries)
        # Optimization: Only compute non-zero entries
        # But for small dims, full loop is fine
        for i in range(n):
            for j in range(m):
                if u_vec[i*m+j] == 0: continue
                for k in range(p):
                    if v_vec[j*p+k] == 0: continue
                    if w_vec[k*n+i] == 0: continue
                    contrib[i * m * p + j * p + k] = u_vec[i*m+j] * v_vec[j*p+k] * w_vec[k*n+i]
        
        # Avoid adding zero candidates
        if np.any(contrib != 0):
            candidates.append((u_vec, v_vec, w_vec, contrib))
            
    # 1. Standard Unit Tensors
    for i in range(n):
        for j in range(m):
            for k in range(p):
                add_candidate({(i,j): 1}, {(j,k): 1}, {(k,i): 1})
                
    # 2. Strassen 2x2x2 Blocks
    def strassen_2x2x2(io, jo, ko):
        terms = []
        # M1
        terms.append(({(io,jo): 1, (io+1,jo+1): 1}, 
                      {(jo,ko): 1, (jo+1,ko+1): 1}, 
                      {(ko,io): 1, (ko+1,io+1): 1}))
        # M2
        terms.append(({(io+1,jo): 1, (io+1,jo+1): 1}, 
                      {(jo,ko): 1}, 
                      {(ko,io+1): 1, (ko+1,io+1): -1}))
        # M3
        terms.append(({(io,jo): 1}, 
                      {(jo,ko+1): 1, (jo+1,ko+1): -1}, 
                      {(ko+1,io): 1, (ko+1,io+1): 1}))
        # M4
        terms.append(({(io+1,jo+1): 1}, 
                      {(jo+1,ko): 1, (jo,ko): -1}, 
                      {(ko,io): 1, (ko,io+1): 1}))
        # M5
        terms.append(({(io,jo): 1, (io,jo+1): 1}, 
                      {(jo+1,ko+1): 1}, 
                      {(ko,io): -1, (ko+1,io): 1}))
        # M6
        terms.append(({(io+1,jo): 1, (io,jo): -1}, 
                      {(jo,ko): 1, (jo,ko+1): 1}, 
                      {(ko+1,io+1): 1}))
        # M7
        terms.append(({(io,jo+1): 1, (io+1,jo+1): -1}, 
                      {(jo+1,ko): 1, (jo+1,ko+1): 1}, 
                      {(ko,io): 1}))
        return terms

    for io in range(n - 1):
        for jo in range(m - 1):
            for ko in range(p - 1):
                for u_c, v_c, w_c in strassen_2x2x2(io, jo, ko):
                    add_candidate(u_c, v_c, w_c)
                    
    # 3. Strassen-like 2x3x2 Blocks (Extending Strassen to 3 cols)
    # Rank of 2x3x2 is 11? No, it can be done in 7? 
    # Actually, 2x3x2 is equivalent to 2x2x2 + 2x1x2.
    # Let's generate heuristic terms for 2x3x2.
    # We can use the 7 Strassen terms for the first 2 cols, and 2 standard terms for the 3rd col.
    # But we want "wide" terms that mix them.
    
    def strassen_2x3x2_terms(io, jo, ko):
        # Covers i: io..io+1, j: jo..jo+2, k: ko..ko+1
        terms = []
        # Use standard Strassen for j=jo, jo+1
        for u_c, v_c, w_c in strassen_2x2x2(io, jo, ko):
            # Filter V to only include j=jo, jo+1 (which is already the case)
            terms.append((u_c, v_c, w_c))
            
        # Add terms for the 3rd column j=jo+2
        # We need to cover A[i, jo+2] * B[jo+2, k]
        # Term 8: A02 * B20
        terms.append(({(io, jo+2): 1}, {(jo+2, ko): 1}, {(ko, io): 1}))
        # Term 9: A02 * B21
        terms.append(({(io, jo+2): 1}, {(jo+2, ko+1): 1}, {(ko+1, io): 1}))
        # Term 10: A12 * B20
        terms.append(({(io+1, jo+2): 1}, {(jo+2, ko): 1}, {(ko, io+1): 1}))
        # Term 11: A12 * B21
        terms.append(({(io+1, jo+2): 1}, {(jo+2, ko+1): 1}, {(ko+1, io+1): 1}))
        return terms

    for io in range(n - 1):
        for jo in range(m - 2):
            for ko in range(p - 1):
                for u_c, v_c, w_c in strassen_2x3x2_terms(io, jo, ko):
                    add_candidate(u_c, v_c, w_c)
                    
    # 4. Strassen-like 2x2x3 Blocks (Extending Strassen to 3 rows)
    def strassen_2x2x3_terms(io, jo, ko):
        # Covers i: io..io+1, j: jo..jo+1, k: ko..ko+2
        terms = []
        # Use standard Strassen for k=ko, ko+1
        for u_c, v_c, w_c in strassen_2x2x2(io, jo, ko):
            terms.append((u_c, v_c, w_c))
            
        # Add terms for the 3rd row k=ko+2
        # Term 8: A00 * B02
        terms.append(({(io, jo): 1}, {(jo, ko+2): 1}, {(ko+2, io): 1}))
        # Term 9: A01 * B12
        terms.append(({(io, jo+1): 1}, {(jo+1, ko+2): 1}, {(ko+2, io): 1}))
        # Term 10: A10 * B02
        terms.append(({(io+1, jo): 1}, {(jo, ko+2): 1}, {(ko+2, io+1): 1}))
        # Term 11: A11 * B12
        terms.append(({(io+1, jo+1): 1}, {(jo+1, ko+2): 1}, {(ko+2, io+1): 1}))
        return terms

    for io in range(n - 1):
        for jo in range(m - 1):
            for ko in range(p - 2):
                for u_c, v_c, w_c in strassen_2x2x3_terms(io, jo, ko):
                    add_candidate(u_c, v_c, w_c)
                    
    # 5. 2x3x3 Blocks (Heuristic Rank-11)
    def strassen_2x3x3_terms(io, jo, ko):
        terms = []
        # Combine 2x3x2 and 2x2x3?
        # Let's just use the 7 Strassen terms for 2x2x2, plus standard for the rest.
        # This is not minimal but provides diversity.
        for u_c, v_c, w_c in strassen_2x2x2(io, jo, ko):
            terms.append((u_c, v_c, w_c))
            
        # Add standard terms for the remaining parts of the 2x3x3 block
        # j=jo+2, k=ko, ko+1
        terms.append(({(io, jo+2): 1}, {(jo+2, ko): 1}, {(ko, io): 1}))
        terms.append(({(io, jo+2): 1}, {(jo+2, ko+1): 1}, {(ko+1, io): 1}))
        terms.append(({(io+1, jo+2): 1}, {(jo+2, ko): 1}, {(ko, io+1): 1}))
        terms.append(({(io+1, jo+2): 1}, {(jo+2, ko+1): 1}, {(ko+1, io+1): 1}))
        
        # k=ko+2, j=jo, jo+1
        terms.append(({(io, jo): 1}, {(jo, ko+2): 1}, {(ko+2, io): 1}))
        terms.append(({(io, jo+1): 1}, {(jo+1, ko+2): 1}, {(ko+2, io): 1}))
        terms.append(({(io+1, jo): 1}, {(jo, ko+2): 1}, {(ko+2, io+1): 1}))
        terms.append(({(io+1, jo+1): 1}, {(jo+1, ko+2): 1}, {(ko+2, io+1): 1}))
        
        # Corner j=jo+2, k=ko+2
        terms.append(({(io, jo+2): 1}, {(jo+2, ko+2): 1}, {(ko+2, io): 1}))
        terms.append(({(io+1, jo+2): 1}, {(jo+2, ko+2): 1}, {(ko+2, io+1): 1}))
        
        return terms

    for io in range(n - 1):
        for jo in range(m - 2):
            for ko in range(p - 2):
                for u_c, v_c, w_c in strassen_2x3x3_terms(io, jo, ko):
                    add_candidate(u_c, v_c, w_c)

    num_candidates = len(candidates)
    
    # Precompute norms
    norms = np.zeros(num_candidates)
    for idx in range(num_candidates):
        vec = candidates[idx][3]
        norms[idx] = np.dot(vec, vec)
        
    # Greedy Search with Polish
    best_rank = 54 # Fallback
    best_solution = None
    
    # Try to find an exact solution using Greedy + Least Squares
    # We will iterate with increasing number of allowed candidates
    # But greedy is sequential.
    
    selected_indices = []
    residual = T_target.copy()
    
    # Greedy selection
    max_iter = 30
    for _ in range(max_iter):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        best_idx = -1
        best_reduction = 0
        best_coeff = 0
        
        # To speed up, we can sample or just iterate all if num_candidates is small
        # num_candidates is roughly 60 + 20*7 + ... ~ 300-400. Full iteration is fast.
        for idx in range(num_candidates):
            if norms[idx] < 1e-12:
                continue
            dot_rc = np.dot(residual, candidates[idx][3])
            coeff = dot_rc / norms[idx]
            reduction = (dot_rc ** 2) / norms[idx]
            
            if reduction > best_reduction:
                best_reduction = reduction
                best_idx = idx
                best_coeff = coeff
        
        if best_idx == -1 or best_reduction < 1e-9:
            break
        
        selected_indices.append(best_idx)
        residual -= best_coeff * candidates[best_idx][3]
        
        # Polish Step: Solve least squares for the selected indices
        if len(selected_indices) > 0:
            A_mat = np.column_stack([candidates[idx][3] for idx in selected_indices])
            # Solve A_mat @ x = T_target
            try:
                x_sol, residuals, rank, sv = np.linalg.lstsq(A_mat, T_target, rcond=None)
                residual_check = T_target - A_mat @ x_sol
                err = np.linalg.norm(residual_check)
                
                if err < 1e-9:
                    # Found exact solution!
                    # Construct U, V, W
                    rank_found = len(selected_indices)
                    U = np.zeros((n * m, rank_found))
                    V = np.zeros((m * p, rank_found))
                    W = np.zeros((p * n, rank_found))
                    
                    for r, idx in enumerate(selected_indices):
                        coeff = x_sol[r]
                        u_vec, v_vec, w_vec, _ = candidates[idx]
                        U[:, r] = u_vec * coeff
                        V[:, r] = v_vec
                        W[:, r] = w_vec
                        
                    # Verify
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
                        
            except:
                pass

    # If greedy didn't find exact, try a few random restarts with different initial picks
    # Or just return the best approximate found or fallback
    
    # Fallback: Construct a known valid solution
    # We will use the 54-rank solution from the original code as a safety net
    # But we can try to improve it slightly by using the 2x3x2 blocks in the construction
    
    n, m, p = 2, 5, 6
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
