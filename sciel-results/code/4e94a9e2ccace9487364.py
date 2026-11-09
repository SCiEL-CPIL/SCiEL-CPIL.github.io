import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 100
    
    # Schoolbook decomposition (rank 100)
    U_school = [[0.0] * rank_school for _ in range(n * m)]
    V_school = [[0.0] * rank_school for _ in range(m * p)]
    W_school = [[0.0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1.0
                V_school[j * p + k][column] = 1.0
                W_school[k * n + i][column] = 1.0
                column += 1
    
    # Target tensor T with T[a, b, c] = 1 if a=5i+j, b=5j+k, c=5k+i
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s limit
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    def check_and_round(U, V, W, r):
        """Check if U, V, W with r columns exactly decompose T, and round to half-integers."""
        T_calc = np.zeros((n * m, m * p, p * n))
        for k in range(r):
            uk = U[:, k]
            vk = V[:, k]
            wk = W[:, k]
            T_calc += np.einsum('a,b,c->abc', uk, vk, wk)
        
        error = np.linalg.norm(T - T_calc)
        if error < 1e-4:
            # Round to nearest half-integer
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            # Re-check with rounded values
            T_calc_r = np.zeros((n * m, m * p, p * n))
            for k in range(r):
                uk = U_r[:, k]
                vk = V_r[:, k]
                wk = W_r[:, k]
                T_calc_r += np.einsum('a,b,c->abc', uk, vk, wk)
            
            error_r = np.linalg.norm(T - T_calc_r)
            if error_r < 1e-9:
                U_list = [[float(val) for val in row] for row in U_r]
                V_list = [[float(val) for val in row] for row in V_r]
                W_list = [[float(val) for val in row] for row in W_r]
                return True, U_list, V_list, W_list
        return False, None, None, None
    
    # Search for lower rank using gradient descent with random restarts
    # Try ranks from rank_school - 1 down to a minimum
    for r in range(rank_school - 1, max(40, rank_school // 2), -1):
        if time.time() > deadline:
            break
        
        # Number of restarts decreases as we go to lower ranks (harder)
        max_restarts = max(3, 10 - (rank_school - r) // 10)
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            np.random.seed(12345 + restart * 7 + r)
            U = np.random.randn(n * m, r) * 0.1
            V = np.random.randn(m * p, r) * 0.1
            W = np.random.randn(p * n, r) * 0.1
            
            # Adaptive learning rate gradient descent
            lr = 0.05
            min_lr = 0.001
            prev_err = float('inf')
            
            for step in range(100):
                if time.time() > deadline:
                    break
                
                # Compute current tensor approximation
                T_calc = np.zeros((n * m, m * p, p * n))
                for k in range(r):
                    uk = U[:, k]
                    vk = V[:, k]
                    wk = W[:, k]
                    T_calc += np.einsum('a,b,c->abc', uk, vk, wk)
                
                diff = T - T_calc
                err = np.linalg.norm(diff)
                
                if err < 1e-5:
                    break
                
                # Adaptive learning rate
                if err > prev_err:
                    lr *= 0.5
                    if lr < min_lr:
                        break
                else:
                    lr = min(lr * 1.1, 0.05)
                prev_err = err
                
                # Compute gradients
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for k in range(r):
                    vk = V[:, k]
                    wk = W[:, k]
                    uk = U[:, k]
                    grad_U[:, k] = -2.0 * np.einsum('abc,b,c->a', diff, vk, wk)
                    grad_V[:, k] = -2.0 * np.einsum('abc,a,c->b', diff, uk, wk)
                    grad_W[:, k] = -2.0 * np.einsum('abc,a,b->c', diff, uk, vk)
                
                # Gradient step
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
            
            # Final check
            success, U_list, V_list, W_list = check_and_round(U, V, W, r)
            if success:
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # If no better rank found, return schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}
