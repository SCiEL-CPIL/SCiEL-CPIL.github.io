import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 2, 4, 6
    
    def get_fallback():
        rank = 42
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        strassen_terms = [
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
            ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
            ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
            ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
            ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
            ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
            ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
        ]
        
        blocks = [
            (0, 0, 0, 0),
            (0, 0, 0, 2),
            (0, 0, 0, 4),
            (0, 2, 2, 0),
            (0, 2, 2, 2),
            (0, 2, 2, 4)
        ]
        
        col = 0
        for a_r_off, a_c_off, b_r_off, b_c_off in blocks:
            for a_coeffs, b_coeffs, c_coeffs in strassen_terms:
                for i, j, coeff in a_coeffs:
                    u_idx = (i + a_r_off) * m + (j + a_c_off)
                    U[u_idx][col] += coeff
                for i, j, coeff in b_coeffs:
                    v_idx = (i + b_r_off) * p + (j + b_c_off)
                    V[v_idx][col] += coeff
                for i, j, coeff in c_coeffs:
                    global_i = i + a_r_off
                    global_k = j + b_c_off
                    w_idx = global_k * n + global_i
                    W[w_idx][col] += coeff
                col += 1
        
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(rank):
                        val += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                    if abs(val - 1) > 1e-6:
                        return False
        return True

    def als_search(rank, max_iters=50, restarts=5):
        for _ in range(restarts):
            if time.time() > deadline - 10:
                return None
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            for _ in range(max_iters):
                if time.time() > deadline - 10:
                    return None
                # Solve for U
                U_new = np.zeros_like(U)
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((p, rank))
                        for k in range(p):
                            for r in range(rank):
                                A[k, r] = V[j*p + k, r] * W[k*n + i, r]
                        b = np.ones(p)
                        U_new[i*m + j, :], residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                U = U_new
                
                # Solve for V
                V_new = np.zeros_like(V)
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, rank))
                        for i in range(n):
                            for r in range(rank):
                                A[i, r] = U[i*m + j, r] * W[k*n + i, r]
                        b = np.ones(n)
                        V_new[j*p + k, :], residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                V = V_new
                
                # Solve for W
                W_new = np.zeros_like(W)
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, rank))
                        for j in range(m):
                            for r in range(rank):
                                A[j, r] = U[i*m + j, r] * V[j*p + k, r]
                        b = np.ones(m)
                        W_new[k*n + i, :], residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                W = W_new
            
            # Round to nearest 0.5
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            # Verify
            if verify(U_r.tolist(), V_r.tolist(), W_r.tolist()):
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        
        return None

    # Try ranks
    for rank in range(26, 42):
        if time.time() > deadline - 20:
            break
        result = als_search(rank)
        if result:
            return result

    return get_fallback()
