import numpy as np
import time

DIMENSIONS = [4, 5, 5]
_START = time.time()
_DEADLINE = 120.0  # seconds

def construct():
    n, m, p = DIMENSIONS  # n=4, m=5, p=5
    
    # Build the 4x4x4 core using Strassen (rank 49)
    # Then handle the remaining dimensions with schoolbook
    
    # Strassen 2x2x2 factors
    # U indices: 0:00, 1:01, 2:10, 3:11
    # V indices: 0:00, 1:01, 2:10, 3:11
    # W indices: 0:00, 1:01, 2:10, 3:11 (using k*n+i, n=2)
    
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
    
    # 4x4 Strassen terms
    # Each term has A-block coeffs, B-block coeffs, C-block coeffs
    # Block (r,c) means rows 2r..2r+1, cols 2c..2c+1
    
    outer_terms = [
        # (A_coeffs, B_coeffs, C_coeffs)
        # A11+A22, B11+B22 -> C11+C22
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # A21+A22, B11 -> C21-C22
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # A11, B12-B22 -> C12+C22
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # A22, B21-B11 -> C11+C21
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # A11+A12, B22 -> C11-C12
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # A21-A11, B11+B12 -> C22
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # A12-A22, B21+B22 -> C11
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
            
            # Fill U: for each A block with coeff a, for each Strassen U entry
            for (ar, ac), a_val in a_coeffs.items():
                for sr in range(2):
                    for sc in range(2):
                        s_u_val = s_u[sr * 2 + sc][s_idx]
                        if s_u_val != 0:
                            gi = 2 * ar + sr
                            gk = 2 * ac + sc
                            u_vec[gi * m + gk] += a_val * s_u_val
            
            # Fill V: for each B block with coeff b, for each Strassen V entry
            for (br, bc), b_val in b_coeffs.items():
                for sr in range(2):
                    for sc in range(2):
                        s_v_val = s_v[sr * 2 + sc][s_idx]
                        if s_v_val != 0:
                            gk = 2 * br + sr
                            gj = 2 * bc + sc
                            v_vec[gk * p + gj] += b_val * s_v_val
            
            # Fill W: for each C block with coeff c, for each Strassen W entry
            # W index is j*n + i
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
    
    # Try to reduce rank by finding and removing linearly dependent terms
    if len(U_cols) > 1 and time.time() - _START < _DEADLINE - 20:
        U_mat = np.array(U_cols, dtype=float).T  # shape (n*m, rank)
        V_mat = np.array(V_cols, dtype=float).T  # shape (m*p, rank)
        W_mat = np.array(W_cols, dtype=float).T  # shape (p*n, rank)
        
        # Find rank of U matrix
        U_svd, s_vals, Vt = np.linalg.svd(U_mat, full_matrices=True)
        tol = 1e-9
        num_indep_U = np.sum(s_vals > tol)
        
        if num_indep_U < len(U_cols):
            null_dim = len(U_cols) - num_indep_U
            null_vectors = Vt[num_indep_U:]  # shape (null_dim, rank)
            
            removed = set()
            for ns_vec in null_vectors:
                if time.time() - _START > _DEADLINE - 10:
                    break
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
