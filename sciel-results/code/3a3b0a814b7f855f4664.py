import numpy as np
import time

DIMENSIONS = [4, 5, 5]
_START = time.time()
_DEADLINE = 120.0  # seconds

def construct():
    n, m, p = DIMENSIONS  # n=4, m=5, p=5
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_rank1(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    def add_term(i, k, j, coeff_u=1.0, coeff_v=1.0, coeff_w=1.0):
        u_vec = np.zeros(n * m)
        v_vec = np.zeros(m * p)
        w_vec = np.zeros(p * n)
        u_vec[i * m + k] = coeff_u
        v_vec[k * p + j] = coeff_v
        w_vec[j * n + i] = coeff_w
        add_rank1(u_vec, v_vec, w_vec)
    
    # Part 1: 4x4x4 multiplication using Strassen (49 terms)
    strassen_products = [
        {'A': {0: 1, 3: 1}, 'B': {0: 1, 3: 1}},
        {'A': {2: 1, 3: 1}, 'B': {0: 1}},
        {'A': {0: 1}, 'B': {1: 1, 3: -1}},
        {'A': {3: 1}, 'B': {2: 1, 0: -1}},
        {'A': {0: 1, 1: 1}, 'B': {3: 1}},
        {'A': {2: 1, 0: -1}, 'B': {0: 1, 1: 1}},
        {'A': {1: 1, 3: -1}, 'B': {2: 1, 3: 1}},
    ]
    
    c_contributions = [
        {0: 1, 3: 1},
        {2: 1, 3: -1},
        {1: 1, 3: 1},
        {0: 1, 2: 1},
        {0: -1, 1: 1},
        {3: 1},
        {0: 1},
    ]
    
    def a_global(block, li, lk):
        if block == 0: return li, lk
        elif block == 1: return li, lk + 2
        elif block == 2: return li + 2, lk
        elif block == 3: return li + 2, lk + 2
    
    def b_global(block, lk, lj):
        if block == 0: return lk, lj
        elif block == 1: return lk, lj + 2
        elif block == 2: return lk + 2, lj
        elif block == 3: return lk + 2, lj + 2
    
    def c_global(block, li, lj):
        if block == 0: return li, lj
        elif block == 1: return li, lj + 2
        elif block == 2: return li + 2, lj
        elif block == 3: return li + 2, lj + 2
    
    inner_strassen = [
        {'A': [(0,0,1),(1,1,1)], 'B': [(0,0,1),(1,1,1)], 'C': [(0,0,1),(1,1,1)]},
        {'A': [(1,0,1),(1,1,1)], 'B': [(0,0,1)], 'C': [(1,0,1),(1,1,-1)]},
        {'A': [(0,0,1)], 'B': [(0,1,1),(1,1,-1)], 'C': [(0,1,1),(1,1,1)]},
        {'A': [(1,1,1)], 'B': [(1,0,1),(0,0,-1)], 'C': [(0,0,1),(1,0,1)]},
        {'A': [(0,0,1),(0,1,1)], 'B': [(1,1,1)], 'C': [(0,0,-1),(0,1,1)]},
        {'A': [(1,0,1),(0,0,-1)], 'B': [(0,0,1),(0,1,1)], 'C': [(1,1,1)]},
        {'A': [(0,1,1),(1,1,-1)], 'B': [(1,0,1),(1,1,1)], 'C': [(0,0,1)]},
    ]
    
    for prod_idx, prod in enumerate(strassen_products):
        a_form = prod['A']
        b_form = prod['B']
        c_contrib = c_contributions[prod_idx]
        for inner_idx, inner in enumerate(inner_strassen):
            u_vec = np.zeros(n * m)
            v_vec = np.zeros(m * p)
            w_vec = np.zeros(p * n)
            for a_block, a_sign in a_form.items():
                for li, lk, inner_sign in inner['A']:
                    gi, gk = a_global(a_block, li, lk)
                    u_vec[gi * m + gk] += a_sign * inner_sign
            for b_block, b_sign in b_form.items():
                for lk, lj, inner_sign in inner['B']:
                    gk, gj = b_global(b_block, lk, lj)
                    v_vec[gk * p + gj] += b_sign * inner_sign
            for c_block, c_sign in c_contrib.items():
                for li, lj, inner_sign in inner['C']:
                    gi, gj = c_global(c_block, li, lj)
                    w_vec[gj * n + gi] += c_sign * inner_sign
            add_rank1(u_vec, v_vec, w_vec)
    
    # Part 2: 4x4x1 (k in 0..3, j=4) - schoolbook (16 terms)
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
    
    # Improved rank reduction: use SVD to find all linearly dependent terms
    if len(U_cols) > 1 and time.time() - _START < _DEADLINE - 20:
        U_mat = np.array(U_cols, dtype=float).T  # shape (n*m, rank)
        V_mat = np.array(V_cols, dtype=float).T  # shape (m*p, rank)
        W_mat = np.array(W_cols, dtype=float).T  # shape (p*n, rank)
        
        # Find rank of U matrix
        U_svd, s_vals, Vt = np.linalg.svd(U_mat, full_matrices=True)
        tol = 1e-9
        num_indep_U = np.sum(s_vals > tol)
        
        if num_indep_U < len(U_cols):
            # The null space of U^T gives linear dependencies among columns
            # Vt rows num_indep_U to end span the null space of U_mat^T
            null_dim = len(U_cols) - num_indep_U
            null_vectors = Vt[num_indep_U:]  # shape (null_dim, rank)
            
            # For each null vector, we can add a linear combination of columns
            # to zero out one column. We want to find a set of columns to remove
            # such that the remaining columns are still valid.
            
            # Strategy: greedily remove columns that have large null-space components
            # But we need to verify the decomposition still works after removal.
            
            # Better approach: find the minimum number of columns to keep
            # that span the same space, then verify the tensor decomposition.
            
            # Actually, for tensor decomposition, we can't just drop columns.
            # We need to find a subset of terms that still gives exact decomposition.
            
            # Let's try: for each null vector, identify which column to drop
            # and verify. Try multiple null vectors.
            
            removed = set()
            for ns_vec in null_vectors:
                if time.time() - _START > _DEADLINE - 10:
                    break
                # Find the column with largest absolute value in this null vector
                # that hasn't been removed yet
                candidates = [i for i in range(len(U_cols)) if i not in removed]
                if not candidates:
                    break
                abs_vals = np.array([abs(ns_vec[i]) for i in candidates])
                best_local = np.argmax(abs_vals)
                drop_idx = candidates[best_local]
                
                if abs(ns_vec[drop_idx]) < 1e-10:
                    continue
                
                # Try removing this column and verify
                new_indices = [i for i in range(len(U_cols)) if i != drop_idx and i not in removed]
                
                # Quick verification with random test
                np.random.seed(42)
                A_test = np.random.randn(n, m)
                B_test = np.random.randn(m, p)
                C_expected = A_test @ B_test
                
                C_computed = np.zeros((n, p))
                for idx in new_indices:
                    u_flat = U_cols[idx]
                    v_flat = V_cols[idx]
                    w_flat = W_cols[idx]
                    # Vectorized computation
                    for i in range(n):
                        for j in range(p):
                            s = 0.0
                            for k in range(m):
                                s += u_flat[i*m+k] * v_flat[k*p+j]
                            C_computed[i,j] += s * w_flat[j*n+i]
                
                if np.allclose(C_computed, C_expected, atol=1e-6):
                    removed.add(drop_idx)
            
            if removed:
                keep_indices = [i for i in range(len(U_cols)) if i not in removed]
                U_cols = [U_cols[i] for i in keep_indices]
                V_cols = [V_cols[i] for i in keep_indices]
                W_cols = [W_cols[i] for i in keep_indices]
    
    U = np.stack(U_cols, axis=1)
    V = np.stack(V_cols, axis=1)
    W = np.stack(W_cols, axis=1)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
