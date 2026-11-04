import numpy as np
import time
import random

def construct():
    n, m, p = 3, 3, 3
    start_time = time.time()
    deadline = start_time + 120
    
    # Build the target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0
    
    best_U = None
    best_V = None
    best_W = None
    best_rank = n*m*p
    
    # Try to find a rank 20 decomposition using the laser method idea
    # Over Z_5 x Z_5, we can use 25 points, and select a 3x3 subset
    # This is complex, so let's try a numerical approach first
    
    # Use a smarter initialization: start with schoolbook and try to merge terms
    # Or use a local search
    
    # Let's try gradient descent with better parameters
    for rank in [20, 21, 22, 23]:
        if time.time() > deadline - 20:
            break
        
        for attempt in range(100):
            if time.time() > deadline - 20:
                break
            
            # Initialize
            U = np.random.randn(n*m, rank) * 0.5
            V = np.random.randn(m*p, rank) * 0.5
            W = np.random.randn(p*n, rank) * 0.5
            
            # Use Adam optimizer
            m_U = np.zeros_like(U)
            v_U = np.zeros_like(U)
            m_V = np.zeros_like(V)
            v_V = np.zeros_like(V)
            m_W = np.zeros_like(W)
            v_W = np.zeros_like(W)
            b1 = 0.9
            b2 = 0.999
            eps = 1e-8
            lr = 0.01
            
            for step in range(2000):
                if time.time() > deadline - 20:
                    break
                
                # Compute error
                T_est = np.einsum('at,bt,ct->abc', U, V, W)
                error = T - T_est
                err_norm = np.linalg.norm(error)
                
                if err_norm < 1e-10:
                    if rank < best_rank:
                        best_rank = rank
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                    break
                
                # Gradients
                grad_U = np.einsum('abc,bt,ct->at', error, V, W)
                grad_V = np.einsum('abc,at,ct->bt', error, U, W)
                grad_W = np.einsum('abc,at,bt->ct', error, U, V)
                
                # Adam update
                t = step + 1
                m_U = b1*m_U + (1-b1)*grad_U
                v_U = b2*v_U + (1-b2)*grad_U**2
                m_U_hat = m_U/(1-b1**t)
                v_U_hat = v_U/(1-b2**t)
                U -= lr * m_U_hat / (np.sqrt(v_U_hat) + eps)
                
                m_V = b1*m_V + (1-b1)*grad_V
                v_V = b2*v_V + (1-b2)*grad_V**2
                m_V_hat = m_V/(1-b1**t)
                v_V_hat = v_V/(1-b2**t)
                V -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)
                
                m_W = b1*m_W + (1-b1)*grad_W
                v_W = b2*v_W + (1-b2)*grad_W**2
                m_W_hat = m_W/(1-b1**t)
                v_W_hat = v_W/(1-b2**t)
                W -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
            
            # Check
            T_est = np.einsum('at,bt,ct->abc', U, V, W)
            err_norm = np.linalg.norm(T - T_est)
            if err_norm < 1e-10 and rank < best_rank:
                best_rank = rank
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
    
    if best_U is None:
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
