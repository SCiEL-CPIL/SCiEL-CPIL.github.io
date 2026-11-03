import time
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Schoolbook as fallback
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
    
    best = schoolbook()
    best_rank = n * m * p
    
    def verify(U, V, W, tol=1e-6):
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def is_half_integer(x):
        return abs(x * 2 - round(x * 2)) < 1e-6
    
    def round_to_half(U, V, W):
        U_r = [[round(x * 2) / 2 for x in row] for row in U]
        V_r = [[round(x * 2) / 2 for x in row] for row in V]
        W_r = [[round(x * 2) / 2 for x in row] for row in W]
        return U_r, V_r, W_r
    
    def try_rank(rank, max_iters=500, seed=42):
        if time.time() > deadline - 10:
            return None
        
        rng = random.Random(seed)
        np_rng = np.random.RandomState(seed)
        
        for attempt in range(20):
            # Initialize
            scale = 1.0 / (rank ** 0.25)
            U = [[rng.uniform(-scale, scale) for _ in range(rank)] for _ in range(n*m)]
            V = [[rng.uniform(-scale, scale) for _ in range(rank)] for _ in range(m*p)]
            W = [[rng.uniform(-scale, scale) for _ in range(rank)] for _ in range(p*n)]
            
            # Alternating least squares
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    return None
                
                # Fix V and W, solve for U
                for i in range(n):
                    for j in range(m):
                        # For each k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                        # This is a linear system in U[i*m+j, :]
                        A_mat = np.zeros((p, rank))
                        b_vec = np.ones(p)
                        for k in range(p):
                            for r in range(rank):
                                A_mat[k, r] = V[j*p+k][r] * W[k*n+i][r]
                        # Solve A_mat @ u = b_vec
                        try:
                            u_sol, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                            U[i*m+j] = u_sol.tolist()
                        except:
                            pass
                
                # Fix U and W, solve for V
                for j in range(m):
                    for k in range(p):
                        # For each i: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                        A_mat = np.zeros((n, rank))
                        b_vec = np.ones(n)
                        for i in range(n):
                            for r in range(rank):
                                A_mat[i, r] = U[i*m+j][r] * W[k*n+i][r]
                        try:
                            v_sol, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                            V[j*p+k] = v_sol.tolist()
                        except:
                            pass
                
                # Fix U and V, solve for W
                for i in range(n):
                    for k in range(p):
                        # For each j: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                        A_mat = np.zeros((m, rank))
                        b_vec = np.ones(m)
                        for j in range(m):
                            for r in range(rank):
                                A_mat[j, r] = U[i*m+j][r] * V[j*p+k][r]
                        try:
                            w_sol, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                            W[k*n+i] = w_sol.tolist()
                        except:
                            pass
                
                # Check convergence
                if verify(U, V, W, tol=1e-4):
                    # Try rounding
                    U_r, V_r, W_r = round_to_half(U, V, W)
                    if verify(U_r, V_r, W_r, tol=1e-6):
                        return {"U": U_r, "V": V_r, "W": W_r}
                    # Also try with smaller tolerance
                    if verify(U_r, V_r, W_r, tol=1e-3):
                        # Accept if close enough
                        return {"U": U_r, "V": V_r, "W": W_r}
        
        return None
    
    # Try ranks from low to high
    for rank in range(18, 36):
        if time.time() > deadline - 15:
            break
        result = try_rank(rank, max_iters=200, seed=rank)
        if result is not None:
            if verify(result["U"], result["V"], result["W"], tol=1e-6):
                best = result
                best_rank = rank
                break
        # Also try with different seeds
        if time.time() > deadline - 15:
            break
        result = try_rank(rank, max_iters=100, seed=rank*7+1)
        if result is not None:
            if verify(result["U"], result["V"], result["W"], tol=1e-6):
                best = result
                best_rank = rank
                break
    
    return best
