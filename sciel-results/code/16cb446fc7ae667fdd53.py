"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    def check_valid(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for t in range(rank):
                        s += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                    if abs(s - 1.0) > 1e-8:
                        return False
        return True
    
    def compute_error(U, V, W, rank):
        """Vectorized error computation using NumPy."""
        U_arr = np.array(U)  # (n*m, rank)
        V_arr = np.array(V)  # (m*p, rank)
        W_arr = np.array(W)  # (p*n, rank)
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_arr[i*m+j]  # (rank,)
                    v_val = V_arr[j*p+k]  # (rank,)
                    w_val = W_arr[k*n+i]  # (rank,)
                    s = np.dot(u_val, v_val * w_val)
                    err += (s - 1.0)**2
        return err
    
    def try_gradient_descent(rank, n_starts=10, iters_per_start=5000):
        """Try gradient descent to find a rank-`rank` decomposition."""
        np.random.seed(12345)
        for s in range(n_starts):
            if time.time() > deadline - 20:
                break
            
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            lr = 0.01
            best_err = float('inf')
            
            for it in range(iters_per_start):
                if time.time() > deadline - 15:
                    break
                
                # Compute error and gradients
                U_arr = U  # (n*m, rank)
                V_arr = V  # (m*p, rank)
                W_arr = W  # (p*n, rank)
                
                grad_U = np.zeros_like(U_arr)
                grad_V = np.zeros_like(V_arr)
                grad_W = np.zeros_like(W_arr)
                
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i*m + j
                            v_idx = j*p + k
                            w_idx = k*n + i
                            
                            u_val = U_arr[u_idx]  # (rank,)
                            v_val = V_arr[v_idx]  # (rank,)
                            w_val = W_arr[w_idx]  # (rank,)
                            
                            s = np.dot(u_val, v_val * w_val)
                            diff = s - 1.0
                            err += diff**2
                            
                            # d/dU[u_idx,t] = 2*diff * V[v_idx,t] * W[w_idx,t]
                            grad_U[u_idx] += 2 * diff * (v_val * w_val)
                            grad_V[v_idx] += 2 * diff * (u_val * w_val)
                            grad_W[w_idx] += 2 * diff * (u_val * v_val)
                
                if err < best_err:
                    best_err = err
                    if err < 1e-14:
                        U_list = U.tolist()
                        V_list = V.tolist()
                        W_list = W.tolist()
                        if check_valid(U_list, V_list, W_list, rank):
                            return U_list, V_list, W_list
                
                # Update with gradient descent
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Adaptive learning rate
                if err < best_err * 0.9:
                    lr = min(lr * 1.1, 0.1)
                else:
                    lr = max(lr * 0.95, 0.001)
        
        return None
    
    # Try ranks below 28 with gradient descent
    for target_rank in [20, 22, 24, 26]:
        if time.time() > deadline - 30:
            break
        result = try_gradient_descent(target_rank, n_starts=5, iters_per_start=3000)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback: rank 28 block Strassen
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_terms = [
        {'U': [(0, 1), (3, 1)], 'V': [(0, 1), (3, 1)], 'W': [(0, 1), (3, 1)]},
        {'U': [(2, 1), (3, 1)], 'V': [(0, 1)], 'W': [(1, 1), (3, -1)]},
        {'U': [(0, 1)], 'V': [(1, 1), (3, -1)], 'W': [(2, 1), (3, 1)]},
        {'U': [(3, 1)], 'V': [(2, 1), (0, -1)], 'W': [(0, 1), (1, 1)]},
        {'U': [(0, 1), (1, 1)], 'V': [(3, 1)], 'W': [(0, -1), (2, 1)]},
        {'U': [(2, 1), (0, -1)], 'V': [(0, 1), (1, 1)], 'W': [(3, 1)]},
        {'U': [(1, 1), (3, -1)], 'V': [(2, 1), (3, 1)], 'W': [(0, 1)]}
    ]
    
    blocks = [
        {'j_offset': 0, 'k_offset': 0},
        {'j_offset': 0, 'k_offset': 2},
        {'j_offset': 2, 'k_offset': 0},
        {'j_offset': 2, 'k_offset': 2}
    ]
    
    col = 0
    for block in blocks:
        j_off = block['j_offset']
        k_off = block['k_offset']
        for term in strassen_terms:
            for (idx, coeff) in term['U']:
                i = idx // 2
                j = idx % 2
                global_i = i
                global_j = j + j_off
                U[global_i * m + global_j][col] += coeff
            for (idx, coeff) in term['V']:
                j = idx // 2
                k = idx % 2
                global_j = j + j_off
                global_k = k + k_off
                V[global_j * p + global_k][col] += coeff
            for (idx, coeff) in term['W']:
                k = idx // 2
                i = idx % 2
                global_k = k + k_off
                global_i = i
                W[global_k * n + global_i][col] += coeff
            col += 1
        
    return {"U": U, "V": V, "W": W}
