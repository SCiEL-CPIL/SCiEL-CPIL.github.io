<<<<<<< SEARCH
import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    def check_decomposition(U, V, W, tol=1e-8):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    start_time = time.time()
    deadline = start_time + 110
    
    # Try to find a rank-32 decomposition
    # Use a structured approach
    
    # For each rank, try to solve the system
    for rank in [32, 33, 34, 35]:
        if time.time() > deadline - 10:
            break
        
        for seed in range(50):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(seed * 7919 + 1000003)
            
            # Initialize
            U = np.random.randn(n*m, rank) * 0.3
            V = np.random.randn(m*p, rank) * 0.3
            W = np.random.randn(p*n, rank) * 0.3
            
            # Adam optimizer
            m_U = np.zeros_like(U)
            v_U = np.zeros_like(U)
            m_V = np.zeros_like(V)
            v_V = np.zeros_like(V)
            m_W = np.zeros_like(W)
            v_W = np.zeros_like(W)
            t = 0
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            lr = 0.01
            
            for it in range(200):
                if time.time() > deadline - 5:
                    break
                t += 1
                
                # Compute gradients
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                            err = s - 1.0
                            dU[i*m+j] += err * V[j*p+k] * W[k*n+i]
                            dV[j*p+k] += err * U[i*m+j] * W[k*n+i]
                            dW[k*n+i] += err * U[i*m+j] * V[j*p+k]
                
                # Adam update
                m_U = beta1 * m_U + (1-beta1) * dU
                v_U = beta2 * v_U + (1-beta2) * dU**2
                m_U_hat = m_U / (1 - beta1**t)
                v_U_hat = v_U / (1 - beta2**t)
                U -= lr * m_U_hat / (np.sqrt(v_U_hat) + eps)
                
                m_V = beta1 * m_V + (1-beta1) * dV
                v_V = beta2 * v_V + (1-beta2) * dV**2
                m_V_hat = m_V / (1 - beta1**t)
                v_V_hat = v_V / (1 - beta2**t)
                V -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)
                
                m_W = beta1 * m_W + (1-beta1) * dW
                v_W = beta2 * v_W + (1-beta2) * dW**2
                m_W_hat = m_W / (1 - beta1**t)
                v_W_hat = v_W / (1 - beta2**t)
                W -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
                
                # Check convergence
                if it % 20 == 0:
                    U_list = [[float(U[i, r]) for r in range(rank)] for i in range(n*m)]
                    V_list = [[float(V[i, r]) for r in range(rank)] for i in range(m*p)]
                    W_list = [[float(W[i, r]) for r in range(rank)] for i in range(p*n)]
                    if check_decomposition(U_list, V_list, W_list, tol=1e-5):
                        U_rounded = [[round(x * 2) / 2 for x in row] for row in U_list]
                        V_rounded = [[round(x * 2) / 2 for x in row] for row in V_list]
                        W_rounded = [[round(x * 2) / 2 for x in row] for row in W_list]
                        if check_decomposition(U_rounded, V_rounded, W_rounded, tol=1e-6):
                            return {"U": U_rounded, "V": V_rounded, "W": W_rounded}
                        return {"U": U_list, "V": V_list, "W": W_list}
            
            # Final check
            U_list = [[float(U[i, r]) for r in range(rank)] for i in range(n*m)]
            V_list = [[float(V[i, r]) for r in range(rank)] for i in range(m*p)]
            W_list = [[float(W[i, r]) for r in range(rank)] for i in range(p*n)]
            if check_decomposition(U_list, V_list, W_list, tol=1e-6):
                U_rounded = [[round(x * 2) / 2 for x in row] for row in U_list]
                V_rounded = [[round(x * 2) / 2 for x in row] for row in V_list]
                W_rounded = [[round(x * 2) / 2 for x in row] for row in W_list]
                if check_decomposition(U_rounded, V_rounded, W_rounded, tol=1e-6):
                    return {"U": U_rounded, "V": V_rounded, "W": W_rounded}
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return best_result
=======
import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    def check_decomposition(U, V, W, tol=1e-8):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    start_time = time.time()
    deadline = start_time + 110
    
    # Precompute index arrays for vectorized operations
    idx_U = []
    idx_V = []
    idx_W = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_U.append(i * m + j)
                idx_V.append(j * p + k)
                idx_W.append(k * n + i)
    idx_U = np.array(idx_U)
    idx_V = np.array(idx_V)
    idx_W = np.array(idx_W)
    
    # Try to find a rank-32 decomposition
    for rank in [32, 33, 34, 35]:
        if time.time() > deadline - 10:
            break
        
        for seed in range(80):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(seed * 7919 + 1000003)
            
            # Initialize with small random values
            U = np.random.randn(n*m, rank) * 0.5
            V = np.random.randn(m*p, rank) * 0.5
            W = np.random.randn(p*n, rank) * 0.5
            
            # Adam optimizer
            m_U = np.zeros_like(U)
            v_U = np.zeros_like(U)
            m_V = np.zeros_like(V)
            v_V = np.zeros_like(V)
            m_W = np.zeros_like(W)
            v_W = np.zeros_like(W)
            t = 0
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            lr = 0.05
            
            for it in range(300):
                if time.time() > deadline - 5:
                    break
                t += 1
                
                # Compute gradients using vectorized operations
                U_sel = U[idx_U, :]  # shape (n*m*p, rank)
                V_sel = V[idx_V, :]  # shape (n*m*p, rank)
                W_sel = W[idx_W, :]  # shape (n*m*p, rank)
                
                # Tensor values: sum over rank dimension
                tensor_vals = np.sum(U_sel * V_sel * W_sel, axis=1)  # shape (n*m*p,)
                err = tensor_vals - 1.0  # shape (n*m*p,)
                
                # Gradients
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                grad_U = err[:, None] * V_sel * W_sel  # shape (n*m*p, rank)
                grad_V = err[:, None] * U_sel * W_sel
                grad_W = err[:, None] * U_sel * V_sel
                
                np.add.at(dU, idx_U, grad_U)
                np.add.at(dV, idx_V, grad_V)
                np.add.at(dW, idx_W, grad_W)
                
                # Adam update
                m_U = beta1 * m_U + (1-beta1) * dU
                v_U = beta2 * v_U + (1-beta2) * dU**2
                m_U_hat = m_U / (1 - beta1**t)
                v_U_hat = v_U / (1 - beta2**t)
                U -= lr * m_U_hat / (np.sqrt(v_U_hat) + eps)
                
                m_V = beta1 * m_V + (1-beta1) * dV
                v_V = beta2 * v_V + (1-beta2) * dV**2
                m_V_hat = m_V / (1 - beta1**t)
                v_V_hat = v_V / (1 - beta2**t)
                V -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)
                
                m_W = beta1 * m_W + (1-beta1) * dW
                v_W = beta2 * v_W + (1-beta2) * dW**2
                m_W_hat = m_W / (1 - beta1**t)
                v_W_hat = v_W / (1 - beta2**t)
                W -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
                
                # Check convergence periodically
                if it % 10 == 0:
                    U_list = [[float(U[i, r]) for r in range(rank)] for i in range(n*m)]
                    V_list = [[float(V[i, r]) for r in range(rank)] for i in range(m*p)]
                    W_list = [[float(W[i, r]) for r in range(rank)] for i in range(p*n)]
                    if check_decomposition(U_list, V_list, W_list, tol=1e-6):
                        # Try rounding to integers first, then half-integers
                        U_int = [[int(round(x)) for x in row] for row in U_list]
                        V_int = [[int(round(x)) for x in row] for row in V_list]
                        W_int = [[int(round(x)) for x in row] for row in W_list]
                        if check_decomposition(U_int, V_int, W_int, tol=1e-6):
                            return {"U": U_int, "V": V_int, "W": W_int}
                        
                        U_half = [[round(x * 2) / 2 for x in row] for row in U_list]
                        V_half = [[round(x * 2) / 2 for x in row] for row in V_list]
                        W_half = [[round(x * 2) / 2 for x in row] for row in W_list]
                        if check_decomposition(U_half, V_half, W_half, tol=1e-6):
                            return {"U": U_half, "V": V_half, "W": W_half}
                        
                        return {"U": U_list, "V": V_list, "W": W_list}
            
            # Final check
            U_list = [[float(U[i, r]) for r in range(rank)] for i in range(n*m)]
            V_list = [[float(V[i, r]) for r in range(rank)] for i in range(m*p)]
            W_list = [[float(W[i, r]) for r in range(rank)] for i in range(p*n)]
            if check_decomposition(U_list, V_list, W_list, tol=1e-6):
                U_int = [[int(round(x)) for x in row] for row in U_list]
                V_int = [[int(round(x)) for x in row] for row in V_list]
                W_int = [[int(round(x)) for x in row] for row in W_list]
                if check_decomposition(U_int, V_int, W_int, tol=1e-6):
                    return {"U": U_int, "V": V_int, "W": W_int}
                
                U_half = [[round(x * 2) / 2 for x in row] for row in U_list]
                V_half = [[round(x * 2) / 2 for x in row] for row in V_list]
                W_half = [[round(x * 2) / 2 for x in row] for row in W_list]
                if check_decomposition(U_half, V_half, W_half, tol=1e-6):
                    return {"U": U_half, "V": V_half, "W": W_half}
                
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return best_result
>>>>>>> REPLACE
