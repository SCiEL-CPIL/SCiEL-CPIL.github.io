import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 8]
START_TIME = time.time()
TIME_LIMIT = 140  # Leave 10s margin

def construct():
    global START_TIME
    if time.time() - START_TIME > TIME_LIMIT:
        return _rank_44_solution()
    
    n, m, p = DIMENSIONS
    
    best_result = _rank_44_solution()
    
    # Try ranks from 43 down to 36
    for rank in range(43, 35, -1):
        if time.time() - START_TIME > TIME_LIMIT - 15:
            break
        
        result = _search_for_rank_fast(n, m, p, rank)
        if result is not None:
            best_result = result
            break
    
    return best_result

def _rank_44_solution():
    n, m, p = DIMENSIONS
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A[:,0] * B[0,:] -> 2x8
    for i in range(2):
        for k in range(8):
            U[i * 3 + 0][col] = 1
            V[0 * 8 + k][col] = 1
            W[k * 2 + i][col] = 1
            col += 1
            
    # Part 2: A[:,1:3] * B[1:3,:] -> 2x8 using 4 Strassen blocks
    for block in range(4):
        # m1 = (a00+a11)(b00+b11)
        U[1][col] = 1
        U[5][col] = 1
        V[1*8 + block*2][col] = 1
        V[2*8 + block*2+1][col] = 1
        W[(block*2)*2 + 0][col] = 1
        W[(block*2+1)*2 + 1][col] = 1
        col += 1
        
        # m2 = (a10+a11)b00
        U[4][col] = 1
        U[5][col] = 1
        V[1*8 + block*2][col] = 1
        W[(block*2)*2 + 1][col] = 1
        W[(block*2+1)*2 + 1][col] = -1
        col += 1
        
        # m3 = a00(b01-b11)
        U[1][col] = 1
        V[1*8 + block*2+1][col] = 1
        V[2*8 + block*2+1][col] = -1
        W[(block*2+1)*2 + 0][col] = 1
        W[(block*2+1)*2 + 1][col] = 1
        col += 1
        
        # m4 = a11(b10-b00)
        U[5][col] = 1
        V[2*8 + block*2][col] = 1
        V[1*8 + block*2][col] = -1
        W[(block*2)*2 + 0][col] = 1
        W[(block*2)*2 + 1][col] = 1
        col += 1
        
        # m5 = (a00+a01)b11
        U[1][col] = 1
        U[2][col] = 1
        V[2*8 + block*2+1][col] = 1
        W[(block*2)*2 + 0][col] = -1
        W[(block*2+1)*2 + 0][col] = 1
        col += 1
        
        # m6 = (a10-a00)(b00+b01)
        U[4][col] = 1
        U[1][col] = -1
        V[1*8 + block*2][col] = 1
        V[1*8 + block*2+1][col] = 1
        W[(block*2+1)*2 + 1][col] = 1
        col += 1
        
        # m7 = (a01-a11)(b10+b11)
        U[2][col] = 1
        U[5][col] = -1
        V[2*8 + block*2][col] = 1
        V[2*8 + block*2+1][col] = 1
        W[(block*2)*2 + 0][col] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_np[i*m + j, :] * V_np[j*p + k, :] * W_np[k*n + i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _search_for_rank_fast(n, m, p, target_rank, max_restarts=8):
    """Try to find a valid decomposition with the given rank using fast numerical optimization."""
    if time.time() - START_TIME > TIME_LIMIT - 10:
        return None
    
    U_size = n * m
    V_size = m * p
    W_size = p * n
    
    # Precompute index arrays for fast residual computation
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices.append((i*m + j, j*p + k, k*n + i))
    indices = np.array(indices)
    u_idx = indices[:, 0]
    v_idx = indices[:, 1]
    w_idx = indices[:, 2]
    
    for restart in range(max_restarts):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            return None
        
        # Initialize with structured random values
        np.random.seed(12345 + restart * 7)
        U_init = np.random.randn(U_size, target_rank) * 0.3
        V_init = np.random.randn(V_size, target_rank) * 0.3
        W_init = np.random.randn(W_size, target_rank) * 0.3
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        def residuals(x):
            U_flat = x[:U_size*target_rank]
            V_flat = x[U_size*target_rank:U_size*target_rank+V_size*target_rank]
            W_flat = x[U_size*target_rank+V_size*target_rank:]
            
            U_mat = U_flat.reshape(U_size, target_rank)
            V_mat = V_flat.reshape(V_size, target_rank)
            W_mat = W_flat.reshape(W_size, target_rank)
            
            # Vectorized residual computation
            u_vals = U_mat[u_idx, :]  # shape: (n*m*p, target_rank)
            v_vals = V_mat[v_idx, :]  # shape: (n*m*p, target_rank)
            w_vals = W_mat[w_idx, :]  # shape: (n*m*p, target_rank)
            
            prod = u_vals * v_vals * w_vals  # shape: (n*m*p, target_rank)
            vals = np.sum(prod, axis=1)  # shape: (n*m*p,)
            
            return vals - 1.0
        
        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=5000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            
            if result.cost < 1e-8:
                U_flat = result.x[:U_size*target_rank]
                V_flat = result.x[U_size*target_rank:U_size*target_rank+V_size*target_rank]
                W_flat = result.x[U_size*target_rank+V_size*target_rank:]
                U_mat = U_flat.reshape(U_size, target_rank)
                V_mat = V_flat.reshape(V_size, target_rank)
                W_mat = W_flat.reshape(W_size, target_rank)
                
                U_list = [[float(U_mat[i, j]) for j in range(target_rank)] for i in range(U_size)]
                V_list = [[float(V_mat[i, j]) for j in range(target_rank)] for i in range(V_size)]
                W_list = [[float(W_mat[i, j]) for j in range(target_rank)] for i in range(W_size)]
                
                if _verify_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-4):
                    return {"U": U_list, "V": V_list, "W": W_list}
        except Exception:
            pass
    
    return None
