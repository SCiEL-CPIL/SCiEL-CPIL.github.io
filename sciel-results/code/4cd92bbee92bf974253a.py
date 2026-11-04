import numpy as np
import time
import random

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # 100s limit for search
    
    # Trivial solution
    rank_trivial = n * m * p
    U_trivial = [[0] * rank_trivial for _ in range(n * m)]
    V_trivial = [[0] * rank_trivial for _ in range(m * p)]
    W_trivial = [[0] * rank_trivial for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j][col] = 1
                V_trivial[j * p + k][col] = 1
                W_trivial[k * n + i][col] = 1
                col += 1
                
    best_U, best_V, best_W = U_trivial, V_trivial, W_trivial
    best_rank = rank_trivial
    
    # Target rank to try
    # Known lower bound is roughly n*m*p / 2? No.
    # Let's try ranks from 60 down to 70
    for rank in range(60, 80):
        if time.time() > deadline - 10:
            break
            
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
                
            # Random initialization
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # Simple gradient descent steps
            # Error E = sum_r U_r o V_r o W_r - T
            # T is all ones
            T = np.ones((n, m, p))
            
            # We want to minimize || sum_r U_r o V_r o W_r - T ||^2
            # Gradient with respect to U:
            # dL/dU_r = sum_{i,j} (sum_k (sum_s U_s[i,j] V_s[j,k] W_s[k,i]) - 1) * V_r[j,k] W_r[k,i]
            
            # This is complex to vectorize efficiently in pure Python/NumPy without loops.
            # Given the small size, loops are fine.
            
            lr = 0.01
            for step in range(500):
                if time.time() > deadline - 5:
                    break
                
                # Compute current tensor
                current_T = np.zeros((n, m, p))
                for r in range(rank):
                    u_r = U[:, r].reshape(n, m)
                    v_r = V[:, r].reshape(m, p)
                    w_r = W[:, r].reshape(p, n)
                    
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                current_T[i, j, k] += u_r[i, j] * v_r[j, k] * w_r[k, i]
                
                error = current_T - T
                
                # Check convergence
                if np.linalg.norm(error) < 1e-5:
                    # Round and verify
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    # Verify exactness
                    check_T = np.zeros((n, m, p))
                    for r in range(rank):
                        u_r = U_rounded[:, r].reshape(n, m)
                        v_r = V_rounded[:, r].reshape(m, p)
                        w_r = W_rounded[:, r].reshape(p, n)
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    check_T[i, j, k] += u_r[i, j] * v_r[j, k] * w_r[k, i]
                                    
                    if np.allclose(check_T, T, atol=1e-9):
                        # Convert to lists
                        U_list = U_rounded.flatten().reshape(n * m, rank).tolist()
                        V_list = V_rounded.flatten().reshape(m * p, rank).tolist()
                        W_list = W_rounded.flatten().reshape(p * n, rank).tolist()
                        
                        if rank < best_rank:
                            best_U, best_V, best_W = U_list, V_list, W_list
                            best_rank = rank
                        break # Found a good one for this rank
                
                # Compute gradients
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for r in range(rank):
                    u_r = U[:, r].reshape(n, m)
                    v_r = V[:, r].reshape(m, p)
                    w_r = W[:, r].reshape(p, n)
                    
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                e = error[i, j, k]
                                # dL/dU_r[i,j] = e * v_r[j,k] * w_r[k,i]
                                dU[i * m + j, r] += e * v_r[j, k] * w_r[k, i]
                                # dL/dV_r[j,k] = e * u_r[i,j] * w_r[k,i]
                                dV[j * p + k, r] += e * u_r[i, j] * w_r[k, i]
                                # dL/dW_r[k,i] = e * u_r[i,j] * v_r[j,k]
                                dW[k * n + i, r] += e * u_r[i, j] * v_r[j, k]
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
