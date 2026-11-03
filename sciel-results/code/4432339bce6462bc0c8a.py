import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify_exact(U, V, W):
        n, m, p = DIMENSIONS
        rank = len(U[0])
        
        # Check entries are half-integers
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if abs(val * 2 - round(val * 2)) > 1e-9:
                        return False
        
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        
        T_calc = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
        
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0

        if not np.allclose(T_calc, T, atol=1e-9, rtol=1e-9):
            return False
            
        return True

    def fast_als_decompose(T, R, max_iters=50, tol=1e-5, seed=42):
        np.random.seed(seed)
        U = np.random.randn(n*m, R) * 0.5
        V = np.random.randn(m*p, R) * 0.5
        W = np.random.randn(p*n, R) * 0.5
        
        # Precompute index mappings
        a_to_bc = {}
        for i in range(n):
            for j in range(m):
                a = i * m + j
                a_to_bc[a] = [(j * p + k, k * n + i) for k in range(p)]
        
        b_to_ac = {}
        for j in range(m):
            for k in range(p):
                b = j * p + k
                b_to_ac[b] = [(i * m + j, k * n + i) for i in range(n)]
        
        c_to_ab = {}
        for k in range(p):
            for i in range(n):
                c = k * n + i
                c_to_ab[c] = [(i * m + j, j * p + k) for j in range(m)]
                
        for it in range(max_iters):
            if time.time() > deadline - 10:
                return None
            
            # Update U
            for a in range(n*m):
                bc_list = a_to_bc[a]
                A_mat = np.zeros((len(bc_list), R))
                y = np.zeros(len(bc_list))
                for idx, (b, c) in enumerate(bc_list):
                    A_mat[idx, :] = V[b, :] * W[c, :]
                    y[idx] = T[a, b, c]
                sol, _, _, _ = np.linalg.lstsq(A_mat, y, rcond=None)
                U[a, :] = sol

            # Update V
            for b in range(m*p):
                ac_list = b_to_ac[b]
                A_mat = np.zeros((len(ac_list), R))
                y = np.zeros(len(ac_list))
                for idx, (a, c) in enumerate(ac_list):
                    A_mat[idx, :] = U[a, :] * W[c, :]
                    y[idx] = T[a, b, c]
                sol, _, _, _ = np.linalg.lstsq(A_mat, y, rcond=None)
                V[b, :] = sol

            # Update W
            for c in range(p*n):
                ab_list = c_to_ab[c]
                A_mat = np.zeros((len(ab_list), R))
                y = np.zeros(len(ab_list))
                for idx, (a, b) in enumerate(ab_list):
                    A_mat[idx, :] = U[a, :] * V[b, :]
                    y[idx] = T[a, b, c]
                sol, _, _, _ = np.linalg.lstsq(A_mat, y, rcond=None)
                W[c, :] = sol

            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(T_calc - T)
            if err < tol:
                U = np.round(U * 2) / 2.0
                V = np.round(V * 2) / 2.0
                W = np.round(W * 2) / 2.0
                
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                
                if verify_exact(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
                    
        return None

    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Try ranks from 24 to 35
    for R in range(24, 36):
        if time.time() > deadline - 20:
            break
        for seed in range(20):
            if time.time() > deadline - 10:
                break
            res = fast_als_decompose(T, R, max_iters=30, seed=seed)
            if res is not None:
                return res

    return schoolbook()
