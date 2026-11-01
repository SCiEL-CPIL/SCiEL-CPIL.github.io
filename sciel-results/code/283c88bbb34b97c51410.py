import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Seed solution (Rank 24)
    rank_seed = n * m * p
    U_seed = np.zeros((n * m, rank_seed), dtype=np.float64)
    V_seed = np.zeros((m * p, rank_seed), dtype=np.float64)
    W_seed = np.zeros((p * n, rank_seed), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_seed[i * m + j, col] = 1
                V_seed[j * p + k, col] = 1
                W_seed[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_seed, V_seed, W_seed
    best_rank = rank_seed
    
    start_time = time.time()
    timeout = 130.0  # Safety margin before 150s kill
    
    # Build constraint list: (u_idx, v_idx, w_idx) for each (i,j,k)
    constraints = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                constraints.append((u_idx, v_idx, w_idx))
    
    num_constraints = len(constraints)
    
    def verify(U, V, W, tol=1e-9):
        """Check if decomposition is exact."""
        for u_idx, v_idx, w_idx in constraints:
            val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
            if abs(val - 1.0) > tol:
                return False
        return True
    
    def round_to_half_int(U, V, W):
        """Round to nearest 0.5 and verify."""
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if verify(U_r, V_r, W_r):
            return U_r, V_r, W_r
        return None
    
    # Strategy: Try ranks from 20 upward using sparse random search + gradient descent
    for target_rank in [20, 21, 22, 23]:
        if time.time() - start_time > timeout:
            break
        
        found = False
        
        # Method 1: Sparse random search over half-integer values
        for attempt in range(300):
            if time.time() - start_time > timeout:
                break
            
            np.random.seed(attempt)
            
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            for c in range(target_rank):
                nz_u = np.random.choice(n * m, size=np.random.randint(1, 3), replace=False)
                for idx in nz_u:
                    U[idx, c] = np.random.choice([-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2])
                
                nz_v = np.random.choice(m * p, size=np.random.randint(1, 3), replace=False)
                for idx in nz_v:
                    V[idx, c] = np.random.choice([-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2])
                
                nz_w = np.random.choice(p * n, size=np.random.randint(1, 3), replace=False)
                for idx in nz_w:
                    W[idx, c] = np.random.choice([-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2])
            
            if verify(U, V, W):
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                found = True
                break
        
        if found:
            break
        
        # Method 2: Gradient descent with multiple restarts
        if time.time() - start_time > timeout:
            break
        
        for attempt in range(80):
            if time.time() - start_time > timeout:
                break
            
            np.random.seed(5000 + attempt)
            
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
            
            lr = 0.05
            for step in range(800):
                if time.time() - start_time > timeout:
                    break
                
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for u_idx, v_idx, w_idx in constraints:
                    val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    diff = val - 1.0
                    dU[u_idx, :] += 2 * diff * V[v_idx, :] * W[w_idx, :]
                    dV[v_idx, :] += 2 * diff * U[u_idx, :] * W[w_idx, :]
                    dW[w_idx, :] += 2 * diff * U[u_idx, :] * V[v_idx, :]
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                err = sum((np.dot(U[u, :], V[v, :] * W[w, :]) - 1.0)**2 
                         for u, v, w in constraints)
                if err < 1e-8:
                    result = round_to_half_int(U, V, W)
                    if result:
                        best_U, best_V, best_W = result
                        best_rank = target_rank
                        found = True
                        break
            
            if found:
                break
        
        if found:
            break
    
    if best_rank < rank_seed:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": U_seed.tolist(), "V": V_seed.tolist(), "W": W_seed.tolist()}
