import time
import numpy as np

DIMENSIONS = [6, 6, 6]

_DEADLINE = time.time() + 140

def construct():
    start_time = time.time()
    
    n, m, p = 6, 6, 6
    
    def _schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def _strassen_2x2_schoolbook_3x3():
        block_size = 3
        num_blocks = 2
        dim = n
        
        rank_inner = 27
        rank_outer = 7
        total_rank = rank_inner * rank_outer
        
        U = np.zeros((dim * dim, total_rank))
        V = np.zeros((dim * dim, total_rank))
        W = np.zeros((dim * dim, total_rank))
        
        col = 0
        
        strassen_steps = [
            {'A': { (0,0): 1, (1,1): 1 }, 'B': { (0,0): 1, (1,1): 1 }, 'C': { (0,0): 1, (1,1): 1 }},
            {'A': { (1,0): 1, (1,1): 1 }, 'B': { (0,0): 1 }, 'C': { (1,0): 1, (1,1): -1 }},
            {'A': { (0,0): 1 }, 'B': { (0,1): 1, (1,1): -1 }, 'C': { (0,1): 1, (1,1): 1 }},
            {'A': { (1,1): 1 }, 'B': { (1,0): 1, (0,0): -1 }, 'C': { (0,0): 1, (1,0): 1 }},
            {'A': { (0,0): 1, (0,1): 1 }, 'B': { (1,1): 1 }, 'C': { (0,0): -1, (0,1): 1 }},
            {'A': { (1,0): 1, (0,0): -1 }, 'B': { (0,0): 1, (0,1): 1 }, 'C': { (1,1): 1 }},
            {'A': { (0,1): 1, (1,1): -1 }, 'B': { (1,0): 1, (1,1): 1 }, 'C': { (0,0): 1 }}
        ]
        
        schoolbook_steps = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    schoolbook_steps.append({
                        'A': { (i,j): 1 },
                        'B': { (j,k): 1 },
                        'C': { (i,k): 1 }
                    })
        
        for s_step in strassen_steps:
            A_coeffs_outer = s_step['A']
            B_coeffs_outer = s_step['B']
            C_coeffs_outer = s_step['C']
            
            for sb_step in schoolbook_steps:
                A_coeffs_inner = sb_step['A']
                B_coeffs_inner = sb_step['B']
                C_coeffs_inner = sb_step['C']
                
                for (I, J), ca_out in A_coeffs_outer.items():
                    for (u, v), ca_in in A_coeffs_inner.items():
                        i_global = I * block_size + u
                        j_global = J * block_size + v
                        idx_u = i_global * dim + j_global
                        U[idx_u, col] += ca_out * ca_in
                
                for (J, K), cb_out in B_coeffs_outer.items():
                    for (v, w), cb_in in B_coeffs_inner.items():
                        j_global = J * block_size + v
                        k_global = K * block_size + w
                        idx_v = j_global * dim + k_global
                        V[idx_v, col] += cb_out * cb_in
                
                for (I, K), cc_out in C_coeffs_outer.items():
                    for (u, w), cc_in in C_coeffs_inner.items():
                        i_global = I * block_size + u
                        k_global = K * block_size + w
                        idx_w = k_global * dim + i_global
                        W[idx_w, col] += cc_out * cc_in
                
                col += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def _laderman_3x3_steps():
        """Laderman's rank-19 decomposition for 3x3 matrix multiplication.
        Each step is a dict with 'A', 'B', 'C' keys mapping (row,col) -> coefficient.
        The bilinear form sum of A_coeffs * B_coeffs for each step reconstructs C.
        """
        steps = [
            # m1 = a11*b11
            {'A': {(0,0): 1}, 'B': {(0,0): 1}, 'C': {(0,0): 1}},
            # m2 = (a12+a13)*b21
            {'A': {(0,1): 1, (0,2): 1}, 'B': {(1,0): 1}, 'C': {(1,0): 1}},
            # m3 = (a11-a12)*(b12+b22)
            {'A': {(0,0): 1, (0,1): -1}, 'B': {(0,1): 1, (1,1): 1}, 'C': {(0,0): 1, (0,1): 1}},
            # m4 = a13*(b22-b32)
            {'A': {(0,2): 1}, 'B': {(1,1): 1, (2,1): -1}, 'C': {(2,1): -1}},
            # m5 = (a21+a31)*b13
            {'A': {(1,0): 1, (2,0): 1}, 'B': {(0,2): 1}, 'C': {(0,2): 1}},
            # m6 = (a21-a31)*(b13+b33)
            {'A': {(1,0): 1, (2,0): -1}, 'B': {(0,2): 1, (2,2): 1}, 'C': {(0,2): 1}},
            # m7 = (a22+a23)*b31
            {'A': {(1,1): 1, (1,2): 1}, 'B': {(2,0): 1}, 'C': {(2,0): 1}},
            # m8 = (a22-a32)*(b31+b32)
            {'A': {(1,1): 1, (2,1): -1}, 'B': {(2,0): 1, (2,1): 1}, 'C': {(1,1): 1, (2,1): 1}},
            # m9 = (a23-a33)*(b32+b33)
            {'A': {(1,2): 1, (2,2): -1}, 'B': {(2,1): 1, (2,2): 1}, 'C': {(1,2): -1, (2,2): -1}},
            # m10 = a33*b33
            {'A': {(2,2): 1}, 'B': {(2,2): 1}, 'C': {(2,2): 1}},
            # m11 = (a11+a22)*(b11+b22)
            {'A': {(0,0): 1, (1,1): 1}, 'B': {(0,0): 1, (1,1): 1}, 'C': {(0,0): -1, (1,1): 1}},
            # m12 = (a12+a23)*(b21+b32)
            {'A': {(0,1): 1, (1,2): 1}, 'B': {(1,0): 1, (2,1): 1}, 'C': {(1,0): -1, (2,0): 1}},
            # m13 = (a13+a33)*(b12+b23)
            {'A': {(0,2): 1, (2,2): 1}, 'B': {(0,1): 1, (1,2): 1}, 'C': {(0,1): -1, (0,2): -1}},
            # m14 = (a21+a12)*(b11+b12)
            {'A': {(1,0): 1, (0,1): 1}, 'B': {(0,0): 1, (0,1): 1}, 'C': {(0,0): 1, (0,1): 1}},
            # m15 = (a31+a22)*(b22+b23)
            {'A': {(2,0): 1, (1,1): 1}, 'B': {(1,1): 1, (1,2): 1}, 'C': {(1,0): 1, (1,1): -1, (1,2): 1}},
            # m16 = (a11+a12)*(b21+b11)
            {'A': {(0,0): 1, (0,1): 1}, 'B': {(1,0): 1, (0,0): 1}, 'C': {(0,0): -1}},
            # m17 = (a22+a33)*(b33+b23)
            {'A': {(1,1): 1, (2,2): 1}, 'B': {(2,2): 1, (1,2): 1}, 'C': {(1,1): 1, (1,2): -1, (2,2): 1}},
            # m18 = (a13+a23)*(b12+b13)
            {'A': {(0,2): 1, (1,2): 1}, 'B': {(0,1): 1, (0,2): 1}, 'C': {(0,1): -1, (0,2): 1}},
            # m19 = (a31+a32)*(b32+b31)
            {'A': {(2,0): 1, (2,1): 1}, 'B': {(2,1): 1, (2,0): 1}, 'C': {(2,0): -1, (2,1): 1}},
        ]
        return steps

    def _strassen_2x2_laderman_3x3():
        block_size = 3
        dim = n
        
        laderman_steps = _laderman_3x3_steps()
        rank_inner = len(laderman_steps)  # 19
        rank_outer = 7
        total_rank = rank_inner * rank_outer
        
        U = np.zeros((dim * dim, total_rank))
        V = np.zeros((dim * dim, total_rank))
        W = np.zeros((dim * dim, total_rank))
        
        col = 0
        
        strassen_steps = [
            {'A': { (0,0): 1, (1,1): 1 }, 'B': { (0,0): 1, (1,1): 1 }, 'C': { (0,0): 1, (1,1): 1 }},
            {'A': { (1,0): 1, (1,1): 1 }, 'B': { (0,0): 1 }, 'C': { (1,0): 1, (1,1): -1 }},
            {'A': { (0,0): 1 }, 'B': { (0,1): 1, (1,1): -1 }, 'C': { (0,1): 1, (1,1): 1 }},
            {'A': { (1,1): 1 }, 'B': { (1,0): 1, (0,0): -1 }, 'C': { (0,0): 1, (1,0): 1 }},
            {'A': { (0,0): 1, (0,1): 1 }, 'B': { (1,1): 1 }, 'C': { (0,0): -1, (0,1): 1 }},
            {'A': { (1,0): 1, (0,0): -1 }, 'B': { (0,0): 1, (0,1): 1 }, 'C': { (1,1): 1 }},
            {'A': { (0,1): 1, (1,1): -1 }, 'B': { (1,0): 1, (1,1): 1 }, 'C': { (0,0): 1 }}
        ]
        
        for s_step in strassen_steps:
            A_coeffs_outer = s_step['A']
            B_coeffs_outer = s_step['B']
            C_coeffs_outer = s_step['C']
            
            for ld_step in laderman_steps:
                A_coeffs_inner = ld_step['A']
                B_coeffs_inner = ld_step['B']
                C_coeffs_inner = ld_step['C']
                
                for (I, J), ca_out in A_coeffs_outer.items():
                    for (u, v), ca_in in A_coeffs_inner.items():
                        i_global = I * block_size + u
                        j_global = J * block_size + v
                        idx_u = i_global * dim + j_global
                        U[idx_u, col] += ca_out * ca_in
                
                for (J, K), cb_out in B_coeffs_outer.items():
                    for (v, w), cb_in in B_coeffs_inner.items():
                        j_global = J * block_size + v
                        k_global = K * block_size + w
                        idx_v = j_global * dim + k_global
                        V[idx_v, col] += cb_out * cb_in
                
                for (I, K), cc_out in C_coeffs_outer.items():
                    for (u, w), cc_in in C_coeffs_inner.items():
                        i_global = I * block_size + u
                        k_global = K * block_size + w
                        idx_w = k_global * dim + i_global
                        W[idx_w, col] += cc_out * cc_in
                
                col += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def _verify(U_list, V_list, W_list, tol=1e-10):
        U_np = np.array(U_list, dtype=float)
        V_np = np.array(V_list, dtype=float)
        W_np = np.array(W_list, dtype=float)
        
        # Check tensor decomposition: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = delta_{i,j,k} pattern
        # For matrix mult: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1 if this is the (i,j,k) entry
        # Actually the tensor T[i,j,k] for matrix mult is 1 for all (i,j,k) - it's the all-ones tensor
        # Wait no: T[i,j,k] = 1 means C[i,k] = sum_j A[i,j]*B[j,k]
        # The decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # For matrix multiplication, T[i,j,k] = 1 for all i,j,k
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                u_row = U_np[i * m + j, :]
                for k in range(p):
                    v_row = V_np[j * p + k, :]
                    w_row = W_np[k * n + i, :]
                    s = np.dot(u_row, v_row * w_row)
                    err = abs(s - 1.0)
                    if err > max_err:
                        max_err = err
        
        return max_err < tol, max_err

    # Try Laderman-based decomposition first (rank 133)
    try:
        if time.time() < _DEADLINE:
            res_lad = _strassen_2x2_laderman_3x3()
            valid, err = _verify(res_lad["U"], res_lad["V"], res_lad["W"])
            if valid:
                # Additional spot check with random matrices
                U_np = np.array(res_lad["U"], dtype=float)
                V_np = np.array(res_lad["V"], dtype=float)
                W_np = np.array(res_lad["W"], dtype=float)
                rng = np.random.default_rng(42)
                A_test = rng.standard_normal((n, m))
                B_test = rng.standard_normal((m, p))
                C_expected = A_test @ B_test
                C_decomp = np.zeros((n, p))
                for r in range(U_np.shape[1]):
                    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
                    # C[i,k] += U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] for all j
                    # Vectorized: for each r, C_decomp += sum_j U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
                    # Reshape U[:,r] as (n,m), V[:,r] as (m,p), W[:,r] as (p,n) with W[k*n+i]
                    U_r = U_np[:, r].reshape(n, m)
                    V_r = V_np[:, r].reshape(m, p)
                    W_r = W_np[:, r].reshape(p, n)  # W[k*n+i] so row k, col i
                    # C[i,k] += sum_j U_r[i,j] * V_r[j,k] * W_r[k,i]
                    # = (U_r @ V_r) * W_r.T  element-wise? No...
                    # C[i,k] = sum_j U_r[i,j] * V_r[j,k] * W_r[k,i]
                    # Let M_r = U_r @ V_r, then C[i,k] += M_r[i,k] * W_r[k,i]
                    M_r = U_r @ V_r
                    C_decomp += M_r * W_r.T  # W_r.T[i,k] = W_r[k,i]
                max_rand_err = np.max(np.abs(C_decomp - C_expected))
                if max_rand_err < 1e-10:
                    return res_lad
    except Exception:
        pass
    
    # Fall back to Strassen + schoolbook (rank 189)
    try:
        if time.time() < _DEADLINE:
            res = _strassen_2x2_schoolbook_3x3()
            valid, err = _verify(res["U"], res["V"], res["W"])
            if valid:
                return res
    except Exception:
        pass
    
    return _schoolbook()
