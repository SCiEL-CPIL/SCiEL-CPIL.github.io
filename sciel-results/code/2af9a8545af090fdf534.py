import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin for safety
    
    n, m, p = 2, 4, 5
    dim_u = n * m # 8
    dim_v = m * p # 20
    dim_w = p * n # 10
    
    # Construct target tensor T
    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            u_idx = i * m + j
            for k in range(p):
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    def solve_als(U, V, W, max_iter=20):
        for _ in range(max_iter):
            if time.time() > deadline:
                break
                
            # Solve for U
            T_u = T.reshape(dim_u, dim_v * dim_w)
            A_u = (V[:, :, None] * W[None, :, :]).reshape(dim_v * dim_w, -1)
            try:
                U = np.linalg.lstsq(A_u, T_u.T, rcond=None)[0].T
            except:
                break
            
            # Solve for V
            T_v = T.reshape(dim_v, dim_u * dim_w)
            A_v = (U[:, :, None] * W[None, :, :]).reshape(dim_u * dim_w, -1)
            try:
                V = np.linalg.lstsq(A_v, T_v.T, rcond=None)[0].T
            except:
                break
            
            # Solve for W
            T_w = T.reshape(dim_w, dim_u * dim_v)
            A_w = (U[:, :, None] * V[None, :, :]).reshape(dim_u * dim_v, -1)
            try:
                W = np.linalg.lstsq(A_w, T_w.T, rcond=None)[0].T
            except:
                break
                
        # Calculate error
        T_calc = np.einsum('ur,vr,wr->uvw', U, V, W)
        err = np.linalg.norm(T - T_calc)
        return U, V, W, err

    def verify_exact(U, V, W):
        T_calc = np.einsum('ur,vr,wr->uvw', U, V, W)
        return np.allclose(T_calc, T, atol=1e-9)

    # Try ranks from lower bound
    min_rank = max(dim_u, dim_v, dim_w) # 20
    
    for R in range(min_rank, 41):
        if time.time() > deadline:
            break
            
        # More trials for smaller ranks
        trials = 20 if R < 25 else 10
        
        for trial in range(trials):
            if time.time() > deadline:
                break
                
            # Random initialization
            U = np.random.randn(dim_u, R)
            V = np.random.randn(dim_v, R)
            W = np.random.randn(dim_w, R)
            
            # Run ALS
            U, V, W, err = solve_als(U, V, W, max_iter=15)
            
            if err < 1e-4:
                # Try rounding to half-integers
                U_int = np.round(U * 2) / 2
                V_int = np.round(V * 2) / 2
                W_int = np.round(W * 2) / 2
                
                if verify_exact(U_int, V_int, W_int):
                    # Convert to lists of lists
                    U_list = U_int.tolist()
                    V_list = V_int.tolist()
                    W_list = W_int.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}
            
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
