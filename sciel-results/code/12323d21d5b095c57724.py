"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    
    # The tensor is T_{i,j,k} = 1 for all valid indices.
    # We want sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c} = 1
    
    # Strategy: Use Strassen-like block decomposition or random search with verification.
    # Given the time limit, we will try to find a rank < 36 decomposition.
    # If not found quickly, return schoolbook (rank 36).
    
    # Schoolbook
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    # Check if we can find a better rank via random initialization + Newton/Levenberg-Marquardt
    # or simple gradient descent with restarts.
    
    def check_decomposition(U, V, W, tol=1e-6):
        # Returns True if sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c} == 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def optimize_rank(r, seed, max_iters=500):
        np.random.seed(seed)
        U = np.random.randn(n*m, r) * 0.5
        V = np.random.randn(m*p, r) * 0.5
        W = np.random.randn(p*n, r) * 0.5
        
        # Gradient Descent
        lr = 0.01
        for it in range(max_iters):
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            loss = 0.0
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_vec = U[i*m+j, :]
                        v_vec = V[j*p+k, :]
                        w_vec = W[k*n+i, :]
                        
                        val = np.sum(u_vec * v_vec * w_vec)
                        diff = val - 1.0
                        loss += diff * diff
                        
                        # Gradients
                        dU[i*m+j, :] += 2 * diff * (v_vec * w_vec)
                        dV[j*p+k, :] += 2 * diff * (u_vec * w_vec)
                        dW[k*n+i, :] += 2 * diff * (u_vec * v_vec)
                        
            if loss < 1e-10:
                return U, V, W
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            if loss > 1e6:
                break
                
        return U, V, W

    # Try to find a lower rank
    # Start from 35 down to 30
    # This is a heuristic search
    start_time = time.time()
    time_limit = 100  # Leave 20s buffer
    
    best_rank = rank_schoolbook
    best_decomp = (U_school, V_school, W_school)
    
    # Try a few random seeds for rank 35
    for r in range(35, 29, -1):
        if time.time() - start_time > time_limit:
            break
            
        found = False
        for seed in range(10):
            if time.time() - start_time > time_limit:
                break
                
            U, V, W = optimize_rank(r, seed, max_iters=200)
            if check_decomposition(U, V, W):
                best_rank = r
                best_decomp = (U, V, W)
                found = True
                break
                
        if found:
            # Try to go lower
            continue
        else:
            # If we didn't find rank r, break? Or continue to lower r?
            # Usually if r fails, r-1 might also fail, but let's try one more step down
            # Actually, let's just break to save time if we can't find r.
            # But maybe r-1 is easier? Unlikely.
            # Let's break.
            break

    U, V, W = best_decomp
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
