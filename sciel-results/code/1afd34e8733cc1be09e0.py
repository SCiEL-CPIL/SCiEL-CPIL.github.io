import numpy as np
import time
from itertools import combinations

DIMENSIONS = [4, 5, 5]
_START = time.time()
_DEADLINE = 150.0  # seconds

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
    
    # Strassen for 4x4x4
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
    
    # Generate 49 terms for 4x4x4
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
    
    # Part 2: 4x4x1 (k in 0..3, j=4)
    for i in range(4):
        for k in range(4):
            add_term(i, k, 4, 1.0, 1.0, 1.0)
    
    # Part 3: 4x1x5 (k=4, j in 0..4)
    for i in range(4):
        for j in range(5):
            add_term(i, 4, j, 1.0, 1.0, 1.0)
    
    rank = len(U_cols)
    
    # Verification function
    def verify_decomposition(U_cols, V_cols, W_cols, n, m, p, seeds=None):
        if seeds is None:
            seeds = [42, 123, 456]
        for seed in seeds:
            np.random.seed(seed)
            A_test = np.random.randn(n, m)
            B_test = np.random.randn(m, p)
            C_expected = A_test @ B_test
            C_computed = np.zeros((n, p))
            for r in range(len(U_cols)):
                u_flat = U_cols[r]
                v_flat = V_cols[r]
                w_flat = W_cols[r]
                for i in range(n):
                    for j in range(p):
                        s = 0.0
                        for k in range(m):
                            s += u_flat[i*m+k] * v_flat[k*p+j]
                        C_computed[i,j] += s * w_flat[j*n+i]
            if not np.allclose(C_computed, C_expected, atol=1e-6):
                return False
        return True
    
    def get_null_space(mat, tol=1e-9):
        """Get null space vectors of a matrix."""
        s_vals = np.linalg.svd(mat, compute_uv=False)
        num_indep = np.sum(s_vals > tol)
        if num_indep >= mat.shape[0]:
            return [], num_indep
        _, _, Vt = np.linalg.svd(mat.T, full_matrices=True)
        null_vecs = Vt[num_indep:]
        return null_vecs, num_indep
    
    # Finite field rank-1 analysis
    def finite_field_null_space(mat, field=2):
        """Compute null space over finite field using Gaussian elimination."""
        mat_int = np.round(mat).astype(int) % field
        rows, cols = mat_int.shape
        rank = 0
        pivot_cols = []
        
        # Gaussian elimination over F_p
        for col in range(cols):
            if time.time() - _START > _DEADLINE - 5:
                break
            # Find pivot
            pivot_row = -1
            for row in range(rank, rows):
                if mat_int[row, col] % field != 0:
                    pivot_row = row
                    break
            if pivot_row == -1:
                continue
            
            # Swap rows
            if pivot_row != rank:
                mat_int[[rank, pivot_row]] = mat_int[[pivot_row, rank]]
            
            # Eliminate
            inv = pow(mat_int[rank, col] % field, -1, field) if field > 2 else 1
            for row in range(rows):
                if row != rank and mat_int[row, col] % field != 0:
                    factor = (mat_int[row, col] % field) * inv % field
                    mat_int[row] = (mat_int[row] - factor * mat_int[rank]) % field
            
            rank += 1
            pivot_cols.append(col)
        
        # Find null space basis
        null_vecs = []
        free_cols = [c for c in range(cols) if c not in pivot_cols]
        for free_col in free_cols:
            vec = np.zeros(cols, dtype=int)
            vec[free_col] = 1
            for i, pc in enumerate(pivot_cols):
                vec[pc] = -mat_int[i, free_col] % field
            null_vecs.append(vec)
        
        return null_vecs, rank
    
    # Combined reduction: use both real and finite-field null spaces
    if len(U_cols) > 1 and time.time() - _START < _DEADLINE - 20:
        max_iterations = 200
        for iteration in range(max_iterations):
            if time.time() - _START > _DEADLINE - 10:
                break
            
            U_mat = np.array(U_cols, dtype=float)
            V_mat = np.array(V_cols, dtype=float)
            W_mat = np.array(W_cols, dtype=float)
            num_terms = len(U_cols)
            
            # Get null spaces from both real and finite field
            null_U_real, rank_U_real = get_null_space(U_mat)
            null_V_real, rank_V_real = get_null_space(V_mat)
            null_W_real, rank_W_real = get_null_space(W_mat)
            
            # Finite field null spaces
            null_U_ff, rank_U_ff = finite_field_null_space(U_mat, 2)
            null_V_ff, rank_V_ff = finite_field_null_space(V_mat, 2)
            null_W_ff, rank_W_ff = finite_field_null_space(W_mat, 2)
            
            # Collect all candidate null vectors
            candidates = []
            for ns in null_U_real:
                candidates.append(('U', ns.astype(float)))
            for ns in null_V_real:
                candidates.append(('V', ns.astype(float)))
            for ns in null_W_real:
                candidates.append(('W', ns.astype(float)))
            for ns in null_U_ff:
                candidates.append(('U_ff', ns.astype(float)))
            for ns in null_V_ff:
                candidates.append(('V_ff', ns.astype(float)))
            for ns in null_W_ff:
                candidates.append(('W_ff', ns.astype(float)))
            
            if not candidates:
                break
            
            removed = False
            # Sort by magnitude of largest coefficient for better removal candidates
            candidates.sort(key=lambda x: -np.max(np.abs(x[1])))
            
            for factor_name, ns_vec in candidates:
                if removed or time.time() - _START > _DEADLINE - 10:
                    break
                    
                # Find the index with largest coefficient
                drop_idx = int(np.argmax(np.abs(ns_vec)))
                if abs(ns_vec[drop_idx]) < 1e-10:
                    continue
                
                new_U_cols = [U_cols[i] for i in range(num_terms) if i != drop_idx]
                new_V_cols = [V_cols[i] for i in range(num_terms) if i != drop_idx]
                new_W_cols = [W_cols[i] for i in range(num_terms) if i != drop_idx]
                
                if verify_decomposition(new_U_cols, new_V_cols, new_W_cols, n, m, p):
                    U_cols = new_U_cols
                    V_cols = new_V_cols
                    W_cols = new_W_cols
                    removed = True
                    break
            
            if not removed:
                break
        
        # Pair removal phase
        if time.time() - _START < _DEADLINE - 5:
            for polish_iter in range(30):
                if time.time() - _START > _DEADLINE - 3:
                    break
                
                num_terms = len(U_cols)
                if num_terms <= 1:
                    break
                
                found_removal = False
                for i in range(num_terms):
                    if found_removal or time.time() - _START > _DEADLINE - 3:
                        break
                    for j in range(i+1, num_terms):
                        if found_removal or time.time() - _START > _DEADLINE - 3:
                            break
                        
                        new_U_cols = [U_cols[k] for k in range(num_terms) if k != i and k != j]
                        new_V_cols = [V_cols[k] for k in range(num_terms) if k != i and k != j]
                        new_W_cols = [W_cols[k] for k in range(num_terms) if k != i and k != j]
                        
                        if verify_decomposition(new_U_cols, new_V_cols, new_W_cols, n, m, p, seeds=[42]):
                            U_cols = new_U_cols
                            V_cols = new_V_cols
                            W_cols = new_W_cols
                            found_removal = True
                            break
                
                if not found_removal:
                    break
    
    U = np.stack(U_cols, axis=1)
    V = np.stack(V_cols, axis=1)
    W = np.stack(W_cols, axis=1)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
