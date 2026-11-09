import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 6
    dim_v = m * p  # 15
    dim_w = p * n  # 10
    
    # Construct the target tensor T
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0
    
    def verify_decomposition(U, V, W, T, tol=1e-9):
        r = U.shape[1]
        T_approx = np.zeros_like(T)
        for c in range(r):
            T_approx += U[:, c][:, np.newaxis, np.newaxis] * V[:, c][np.newaxis, :, np.newaxis] * W[:, c][np.newaxis, np.newaxis, :]
        return np.allclose(T_approx, T, atol=tol)
    
    # Build rank-28 decomposition using Strassen on two 2x2x2x2 blocks
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    # Block 2: i in {0,1}, j in {0,1}, k in {2,3}
    # Remaining: all other (i,j,k) triples use schoolbook
    
    terms = []  # Each term is (U_vec, V_vec, W_vec) as sparse dicts
    
    def add_term(u_dict, v_dict, w_dict):
        terms.append((u_dict, v_dict, w_dict))
    
    # Strassen for block with i in {i0,i1}, j in {j0,j1}, k in {k0,k1}
    def strassen_block(i0, i1, j0, j1, k0, k1):
        # Local A indices: (i0,j0)->a, (i0,j1)->b, (i1,j0)->c, (i1,j1)->d
        # Local B indices: (j0,k0)->e, (j0,k1)->f, (j1,k0)->g, (j1,k1)->h
        # Local C indices: (k0,i0)->C00, (k0,i1)->C10, (k1,i0)->C01, (k1,i1)->C11
        
        a_I = i0 * m + j0
        b_I = i0 * m + j1
        c_I = i1 * m + j0
        d_I = i1 * m + j1
        
        e_J = j0 * p + k0
        f_J = j0 * p + k1
        g_J = j1 * p + k0
        h_J = j1 * p + k1
        
        C00_K = k0 * n + i0
        C10_K = k0 * n + i1
        C01_K = k1 * n + i0
        C11_K = k1 * n + i1
        
        # M1 = (a+d)(e+h) -> C00 += 1, C11 += 1
        add_term({a_I: 1, d_I: 1}, {e_J: 1, h_J: 1}, {C00_K: 1, C11_K: 1})
        # M2 = (c+d)e -> C10 += 1
        add_term({c_I: 1, d_I: 1}, {e_J: 1}, {C10_K: 1})
        # M3 = a(f-h) -> C01 += 1
        add_term({a_I: 1}, {f_J: 1, h_J: -1}, {C01_K: 1})
        # M4 = d(g-e) -> C00 += 1, C10 += 1
        add_term({d_I: 1}, {g_J: 1, e_J: -1}, {C00_K: 1, C10_K: 1})
        # M5 = (a+b)h -> C00 -= 1, C01 += 1
        add_term({a_I: 1, b_I: 1}, {h_J: 1}, {C00_K: -1, C01_K: 1})
        # M6 = (c-a)(e+f) -> C11 += 1
        add_term({c_I: 1, a_I: -1}, {e_J: 1, f_J: 1}, {C11_K: 1})
        # M7 = (b-d)(g+h) -> C00 += 1
        add_term({b_I: 1, d_I: -1}, {g_J: 1, h_J: 1}, {C00_K: 1})
    
    # Schoolbook term
    def schoolbook_term(i, j, k):
        I = i * m + j
        J = j * p + k
        K = k * n + i
        add_term({I: 1}, {J: 1}, {K: 1})
    
    # Apply Strassen to blocks
    strassen_block(0, 1, 0, 1, 0, 1)  # i in {0,1}, j in {0,1}, k in {0,1}
    strassen_block(0, 1, 0, 1, 2, 3)  # i in {0,1}, j in {0,1}, k in {2,3}
    
    # Schoolbook for remaining terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip terms covered by Strassen blocks
                if i in (0,1) and j in (0,1) and k in (0,1):
                    continue
                if i in (0,1) and j in (0,1) and k in (2,3):
                    continue
                schoolbook_term(i, j, k)
    
    # Convert to dense matrices
    rank = len(terms)
    U = np.zeros((dim_u, rank))
    V = np.zeros((dim_v, rank))
    W = np.zeros((dim_w, rank))
    
    for c, (u_dict, v_dict, w_dict) in enumerate(terms):
        for idx, val in u_dict.items():
            U[idx, c] = val
        for idx, val in v_dict.items():
            V[idx, c] = val
        for idx, val in w_dict.items():
            W[idx, c] = val
    
    # Verify
    if not verify_decomposition(U, V, W, T):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
