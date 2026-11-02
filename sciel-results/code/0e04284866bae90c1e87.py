import numpy as np
import time

def construct():
    n, m, p = 2, 2, 6
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Generate the target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for k in range(m):
            for j in range(p):
                T[i*m+k, k*p+j, j*n+i] = 1
    
    def compute_residual(U, V, W):
        T_hat = np.einsum('ar,br,cr->abc', U, V, W)
        return np.linalg.norm(T_hat - T)
    
    def adam_decompose(r, max_iter=5000, lr=0.01, seed=0, init_U=None, init_V=None, init_W=None):
        if init_U is not None:
            U = init_U.copy()
            V = init_V.copy()
            W = init_W.copy()
        else:
            np.random.seed(seed)
            U = np.random.randn(n*m, r) * 0.5
            V = np.random.randn(m*p, r) * 0.5
            W = np.random.randn(p*n, r) * 0.5
        
        # Adam state
        m_U = np.zeros_like(U)
        v_U = np.zeros_like(U)
        m_V = np.zeros_like(V)
        v_V = np.zeros_like(V)
        m_W = np.zeros_like(W)
        v_W = np.zeros_like(W)
        
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        t = 0
        
        prev_residual = np.inf
        stagnant_count = 0
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            t += 1
            # Compute Gradient
            # T_hat = U @ V^T @ W^T (conceptually)
            # grad_U = (T_hat - T) * (V * W)
            # grad_V = (T_hat - T) * (U * W)
            # grad_W = (T_hat - T) * (U * V)
            
            T_hat = np.einsum('ar,br,cr->abc', U, V, W)
            diff = T_hat - T
            
            # Gradient for U: sum over b, c of diff[a,b,c] * V[b,r] * W[c,r]
            # grad_U[a,r] = sum_{b,c} diff[a,b,c] V[b,r] W[c,r]
            # This is (diff * V[:,None,:] * W[None,:,:]) summed over b,c? No.
            # grad_U = np.einsum('abc,br,cr->ar', diff, V, W)
            # grad_V = np.einsum('abc,ar,cr->br', diff, U, W)
            # grad_W = np.einsum('abc,ar,br->cr', diff, U, V)
            
            g_U = np.einsum('abc,br,cr->ar', diff, V, W)
            g_V = np.einsum('abc,ar,cr->br', diff, U, W)
            g_W = np.einsum('abc,ar,br->cr', diff, U, V)
            
            # Update Adam states
            m_U = beta1 * m_U + (1 - beta1) * g_U
            v_U = beta2 * v_U + (1 - beta2) * (g_U ** 2)
            m_U_hat = m_U / (1 - beta1 ** t)
            v_U_hat = v_U / (1 - beta2 ** t)
            
            m_V = beta1 * m_V + (1 - beta1) * g_V
            v_V = beta2 * v_V + (1 - beta2) * (g_V ** 2)
            m_V_hat = m_V / (1 - beta1 ** t)
            v_V_hat = v_V / (1 - beta2 ** t)
            
            m_W = beta1 * m_W + (1 - beta1) * g_W
            v_W = beta2 * v_W + (1 - beta2) * (g_W ** 2)
            m_W_hat = m_W / (1 - beta1 ** t)
            v_W_hat = v_W / (1 - beta2 ** t)
            
            # Update parameters
            U -= lr * m_U_hat / (np.sqrt(v_U_hat) + eps)
            V -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)
            W -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
            
            # Check convergence
            residual = compute_residual(U, V, W)
            
            if residual < 1e-10:
                break
            
            if residual > prev_residual * 0.999:
                stagnant_count += 1
            else:
                stagnant_count = 0
            prev_residual = residual
            
            if stagnant_count > 200:
                break
        
        residual = compute_residual(U, V, W)
        return U, V, W, residual
    
    # Build the rank 21 Strassen solution
    U_local = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    V_local = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    W_local = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    
    rank_21 = 21
    U_21 = np.zeros((n*m, rank_21))
    V_21 = np.zeros((m*p, rank_21))
    W_21 = np.zeros((p*n, rank_21))
    
    num_blocks = p // 2
    for k in range(num_blocks):
        base = k * 7
        U_21[:, base:base+7] = U_local
        V_21[2*k, base:base+7] = V_local[0]
        V_21[2*k+1, base:base+7] = V_local[1]
        V_21[6+2*k, base:base+7] = V_local[2]
        V_21[7+2*k, base:base+7] = V_local[3]
        W_21[4*k, base:base+7] = W_local[0]
        W_21[4*k+2, base:base+7] = W_local[1]
        W_21[4*k+1, base:base+7] = W_local[2]
        W_21[4*k+3, base:base+7] = W_local[3]
    
    best_rank = 21
    best_U, best_V, best_W = U_21, V_21, W_21
    
    # Strategy 1: Try to find rank 20 by removing one term from rank 21 and using Adam
    for term_to_remove in range(21):
        if time.time() > deadline:
            break
        
        U_init = np.delete(U_21, term_to_remove, axis=1)
        V_init = np.delete(V_21, term_to_remove, axis=1)
        W_init = np.delete(W_21, term_to_remove, axis=1)
        
        # Add small noise to break symmetry
        np.random.seed(term_to_remove + 42)
        U_init += np.random.randn(n*m, 20) * 0.01
        V_init += np.random.randn(m*p, 20) * 0.01
        W_init += np.random.randn(p*n, 20) * 0.01
        
        U, V, W, residual = adam_decompose(20, max_iter=3000, lr=0.05, init_U=U_init, init_V=V_init, init_W=W_init)
        
        if residual < 1e-10:
            best_rank = 20
            best_U, best_V, best_W = U, V, W
            break
    
    # Strategy 2: Random restarts for rank 20 with Adam
    if best_rank == 21:
        for trial in range(50):
            if time.time() > deadline:
                break
            U, V, W, residual = adam_decompose(20, max_iter=3000, lr=0.05, seed=trial + 1000)
            if residual < 1e-10:
                best_rank = 20
                best_U, best_V, best_W = U, V, W
                break
    
    # Strategy 3: Try rank 19 if time permits
    if best_rank > 19 and time.time() < deadline - 30:
        for trial in range(20):
            if time.time() > deadline:
                break
            U, V, W, residual = adam_decompose(19, max_iter=2000, lr=0.05, seed=trial + 2000)
            if residual < 1e-10:
                best_rank = 19
                best_U, best_V, best_W = U, V, W
                break
    
    if best_rank < 21:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": U_21.tolist(), "V": V_21.tolist(), "W": W_21.tolist()}
