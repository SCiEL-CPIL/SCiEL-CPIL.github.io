import numpy as np
import time

DIMENSIONS = [2, 2, 4]

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Build the tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    start_time = time.time()
    deadline = start_time + 140

    # Try to find rank 14
    def try_rank(r):
        np.random.seed(42)
        # Reshape T for least squares
        T_w = T.reshape(n*m * m*p, p*n)
        T_perm_v = np.transpose(T, (0, 2, 1))
        T_v = T_perm_v.reshape(n*m * p*n, m*p)
        T_perm_u = np.transpose(T, (1, 2, 0))
        T_u = T_perm_u.reshape(m*p * p*n, n*m)

        for attempt in range(5):
            if time.time() > deadline:
                return None
            
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            for it in range(100):
                if time.time() > deadline:
                    return None
                
                # Solve for W
                A_w = np.einsum('ad,bd->abd', U, V).reshape(n*m * m*p, r)
                W_sol, _, _, _ = np.linalg.lstsq(A_w, T_w, rcond=None)
                W = W_sol.T
                
                # Solve for V
                A_v = np.einsum('ad,cd->acd', U, W).reshape(n*m * p*n, r)
                V_sol, _, _, _ = np.linalg.lstsq(A_v, T_v, rcond=None)
                V = V_sol.T
                
                # Solve for U
                A_u = np.einsum('bd,cd->bcd', V, W).reshape(m*p * p*n, r)
                U_sol, _, _, _ = np.linalg.lstsq(A_u, T_u, rcond=None)
                U = U_sol.T
            
            # Check error
            T_recon = np.einsum('ad,bd,cd->abc', U, V, W)
            error = np.linalg.norm(T - T_recon)
            
            if error < 1e-6:
                # Round to half-integers or integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                # Verify rounded version
                T_recon_r = np.einsum('ad,bd,cd->abc', U_r, V_r, W_r)
                error_r = np.linalg.norm(T - T_recon_r)
                
                if error_r < 1e-6:
                    U_list = [[float(x) for x in row] for row in U_r]
                    V_list = [[float(x) for x in row] for row in V_r]
                    W_list = [[float(x) for x in row] for row in W_r]
                    return U_list, V_list, W_list
        
        return None

    result = try_rank(14)
    if result is not None:
        U, V, W = result
        return {"U": U, "V": V, "W": W}

    return {"U": U_school, "V": V_school, "W": W_school}
