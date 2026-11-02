import numpy as np
import time
import itertools

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # Helper to get indices
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    triplets = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    n_triplets = len(triplets)
    
    def get_residuals(U, V, W):
        res = np.zeros(n_triplets)
        for t, (i, j, k) in enumerate(triplets):
            val = np.sum(U[u_idx(i, j)] * V[v_idx(j, k)] * W[w_idx(k, i)])
            res[t] = val - 1.0
        return res

    def total_error(U, V, W):
        res = get_residuals(U, V, W)
        return float(np.sum(res * res))

    def optimize(U, V, W, max_iters=500, tol=1e-12):
        # Adaptive Gradient Descent with momentum
        lr = 0.05
        momentum = 0.9
        vU = np.zeros_like(U)
        vV = np.zeros_like(V)
        vW = np.zeros_like(W)
        
        prev_err = float('inf')
        
        for it in range(max_iters):
            res = get_residuals(U, V, W)
            err = float(np.sum(res * res))
            if err < tol:
                break
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for t, (i, j, k) in enumerate(triplets):
                r = res[t]
                ui = u_idx(i, j)
                vi = v_idx(j, k)
                wi = w_idx(k, i)
                
                term_VW = V[vi] * W[wi]
                term_UW = U[ui] * W[wi]
                term_UV = U[ui] * V[vi]
                
                dU[ui] += 2 * r * term_VW
                dV[vi] += 2 * r * term_UW
                dW[wi] += 2 * r * term_UV
            
            # Momentum update
            vU = momentum * vU - lr * dU
            vV = momentum * vV - lr * dV
            vW = momentum * vW - lr * dW
            
            U += vU
            V += vV
            W += vW
            
            # Adaptive learning rate
            if err > prev_err:
                lr *= 0.5
            else:
                lr = min(lr * 1.05, 0.1)
            prev_err = err
            
        return U, V, W

    def schoolbook(rank=None):
        if rank is None:
            rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col >= rank:
                        break
                    U[u_idx(i, j), col] = 1.0
                    V[v_idx(j, k), col] = 1.0
                    W[w_idx(k, i), col] = 1.0
                    col += 1
                if col >= rank:
                    break
            if col >= rank:
                break
        return U, V, W

    def randomized_algebraic_search(max_restarts=100, max_iters=300):
        best_rank = n * m * p
        best_U, best_V, best_W = schoolbook()
        best_err = 0.0
        
        start_time = time.time()
        deadline = 120 # 30s margin before 150s kill
        
        # Try ranks from 40 to 48
        for r in range(40, 49):
            if time.time() > deadline:
                break
            
            for restart in range(max_restarts):
                if time.time() > deadline:
                    break
                
                # Initialize with schoolbook terms (first r)
                U_init, V_init, W_init = schoolbook(r)
                
                # Add small noise to help escape local minima
                noise_level = 0.1
                U_init += noise_level * np.random.randn(n*m, r)
                V_init += noise_level * np.random.randn(m*p, r)
                W_init += noise_level * np.random.randn(p*n, r)
                
                # Run optimization
                U_opt, V_opt, W_opt = optimize(U_init, V_init, W_init, max_iters=max_iters)
                
                err = total_error(U_opt, V_opt, W_opt)
                
                if err < 1e-6:
                    # Success!
                    return U_opt, V_opt, W_opt, r
                
                # Keep the best found so far (lowest error)
                if best_err == 0.0 or err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U_opt, V_opt, W_opt
                    best_rank = r
        
        return best_U, best_V, best_W, best_rank

    U, V, W, rank = randomized_algebraic_search()
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
