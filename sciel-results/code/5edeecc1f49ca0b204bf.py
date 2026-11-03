"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

    def make_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best_result = make_schoolbook()
    best_rank = n * m * p

    # Precompute index arrays for efficient verification
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    u_idx = i_idx * m + j_idx
    v_idx = j_idx * p + k_idx
    w_idx = k_idx * n + i_idx

    def verify(U, V, W, r):
        """Check if U, V, W form a valid tensor decomposition. Returns max error."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        
        u_vals = Ua[u_idx, :]  # (n*m*p, r)
        v_vals = Va[v_idx, :]  # (n*m*p, r)
        w_vals = Wa[w_idx, :]  # (n*m*p, r)
        
        prod = u_vals * v_vals * w_vals
        result = prod.sum(axis=1)
        
        return np.max(np.abs(result - 1.0))

    def refine(U, V, W, r, max_iter=50):
        """Gradient descent refinement with L-BFGS-like momentum. Returns refined arrays."""
        Ua = np.array(U, dtype=float).copy()
        Va = np.array(V, dtype=float).copy()
        Wa = np.array(W, dtype=float).copy()
        
        # Momentum buffers
        mU = np.zeros_like(Ua)
        mV = np.zeros_like(Va)
        mW = np.zeros_like(Wa)
        beta = 0.9
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            u_vals = Ua[u_idx, :]
            v_vals = Va[v_idx, :]
            w_vals = Wa[w_idx, :]
            
            prod = u_vals * v_vals * w_vals
            result = prod.sum(axis=1)
            error = result - 1.0
            err_norm = np.linalg.norm(error)
            
            if err_norm < 1e-8:
                break
            
            # Compute gradients
            err_4d = error.reshape(n, m, p, 1)
            u_4d = u_vals.reshape(n, m, p, r)
            v_4d = v_vals.reshape(n, m, p, r)
            w_4d = w_vals.reshape(n, m, p, r)
            
            # grad_U[i*m+j, c] = sum_k 2*err[i,j,k]*V[j*p+k,c]*W[k*n+i,c]
            grad_U_4d = 2 * err_4d * v_4d * w_4d
            grad_U = np.sum(grad_U_4d, axis=2).reshape(n*m, r)
            
            # grad_V[j*p+k, c] = sum_i 2*err[i,j,k]*U[i*m+j,c]*W[k*n+i,c]
            grad_V_4d = 2 * err_4d * u_4d * w_4d
            grad_V = np.zeros((m*p, r))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        grad_V[j*p+k, :] += grad_V_4d[i,j,k,:]
            
            # grad_W[k*n+i, c] = sum_j 2*err[i,j,k]*U[i*m+j,c]*V[j*p+k,c]
            grad_W_4d = 2 * err_4d * u_4d * v_4d
            grad_W = np.zeros((p*n, r))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        grad_W[k*n+i, :] += grad_W_4d[i,j,k,:]
            
            # Adaptive learning rate with momentum
            lr = 0.1 / (1.0 + it * 0.3)
            
            mU = beta * mU + (1 - beta) * grad_U
            mV = beta * mV + (1 - beta) * grad_V
            mW = beta * mW + (1 - beta) * grad_W
            
            Ua -= lr * mU
            Va -= lr * mV
            Wa -= lr * mW
        
        return Ua, Va, Wa

    def polish(U, V, W, r, max_iter=100):
        """High-precision polish with decreasing learning rate."""
        Ua = np.array(U, dtype=float).copy()
        Va = np.array(V, dtype=float).copy()
        Wa = np.array(W, dtype=float).copy()
        
        for it in range(max_iter):
            if time.time() > deadline - 2:
                break
            
            u_vals = Ua[u_idx, :]
            v_vals = Va[v_idx, :]
            w_vals = Wa[w_idx, :]
            
            prod = u_vals * v_vals * w_vals
            result = prod.sum(axis=1)
            error = result - 1.0
            err_norm = np.linalg.norm(error)
            
            if err_norm < 1e-10:
                break
            
            err_4d = error.reshape(n, m, p, 1)
            u_4d = u_vals.reshape(n, m, p, r)
            v_4d = v_vals.reshape(n, m, p, r)
            w_4d = w_vals.reshape(n, m, p, r)
            
            grad_U_4d = 2 * err_4d * v_4d * w_4d
            grad_U = np.sum(grad_U_4d, axis=2).reshape(n*m, r)
            
            grad_V_4d = 2 * err_4d * u_4d * w_4d
            grad_V = np.zeros((m*p, r))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        grad_V[j*p+k, :] += grad_V_4d[i,j,k,:]
            
            grad_W_4d = 2 * err_4d * u_4d * v_4d
            grad_W = np.zeros((p*n, r))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        grad_W[k*n+i, :] += grad_W_4d[i,j,k,:]
            
            lr = 0.01 / (1.0 + it * 0.1)
            
            Ua -= lr * grad_U
            Va -= lr * grad_V
            Wa -= lr * grad_W
        
        return Ua, Va, Wa

    # Try to find lower rank decompositions
    # Start from a reasonable lower bound and go up
    # For 4x4x4, best known is 49 (Lader). For 4x4x7, try various ranks.
    
    # Try candidate ranks with structured random initialization
    candidate_ranks = list(range(60, min(best_rank, 100)))
    
    for r in candidate_ranks:
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations
        for trial in range(20):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(trial * 7919 + r)
            
            # Initialize with small random values
            U = np.random.randn(n * m, r) * 0.3
            V = np.random.randn(m * p, r) * 0.3
            W = np.random.randn(p * n, r) * 0.3
            
            # Refine with momentum
            U_ref, V_ref, W_ref = refine(U, V, W, r, max_iter=40)
            
            # Verify
            err = verify(U_ref, V_ref, W_ref, r)
            if err < 1e-6:
                # Polish for higher precision
                U_pol, V_pol, W_pol = polish(U_ref, V_ref, W_ref, r, max_iter=50)
                err_pol = verify(U_pol, V_pol, W_pol, r)
                if err_pol < err:
                    U_ref, V_ref, W_ref = U_pol, V_pol, W_pol
                    err = err_pol
                
                if err < 1e-5:
                    best_rank = r
                    best_result = {
                        "U": U_ref.tolist(),
                        "V": V_ref.tolist(),
                        "W": W_ref.tolist()
                    }
                    break
        
        if best_rank < n * m * p:
            break
    
    if time.time() > deadline:
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
