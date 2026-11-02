import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def _verify_decomposition(U, V, W, n, m, p):
    rank = U.shape[1]
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    T_reconstructed = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    T_target = np.ones((n, m, p))
    return np.allclose(T_reconstructed, T_target, atol=1e-9)

def _build_schoolbook(n, m, p):
    R_school = n * m * p
    U_school = np.zeros((n * m, R_school))
    V_school = np.zeros((m * p, R_school))
    W_school = np.zeros((p * n, R_school))
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, r] = 1.0
                V_school[j * p + k, r] = 1.0
                W_school[k * n + i, r] = 1.0
                r += 1
    return U_school, V_school, W_school

def _try_optimize_rank(target_rank, n, m, p, deadline, max_iters=3000, seed=42):
    np.random.seed(seed)
    
    U = np.random.randn(n * m, target_rank) * 0.1
    V = np.random.randn(m * p, target_rank) * 0.1
    W = np.random.randn(p * n, target_rank) * 0.1
    
    best_loss = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    m_U = np.zeros_like(U)
    v_U = np.zeros_like(U)
    m_V = np.zeros_like(V)
    v_V = np.zeros_like(V)
    m_W = np.zeros_like(W)
    v_W = np.zeros_like(W)
    b1, b2 = 0.9, 0.999
    lr = 0.01
    eps = 1e-8
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        t = it + 1
        lr_t = lr * np.sqrt(1 - b2**t) / (1 - b1**t)
        
        U_r = U.reshape(n, m, target_rank)
        V_r = V.reshape(m, p, target_rank)
        W_r = W.reshape(p, n, target_rank)
        
        T_rec = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        T_target = np.ones((n, m, p))
        diff = T_rec - T_target
        loss = np.sum(diff**2)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if loss < 1e-10:
                if _verify_decomposition(U, V, W, n, m, p):
                    return U, V, W, True
        
        grad_U = 2 * np.einsum('ijk,jkr,kir->ijr', diff, V_r, W_r)
        grad_V = 2 * np.einsum('ijk,ijr,kir->jkr', diff, U_r, W_r)
        grad_W = 2 * np.einsum('ijk,ijr,jkr->kir', diff, U_r, V_r)
        
        grad_U = grad_U.reshape(n * m, target_rank)
        grad_V = grad_V.reshape(m * p, target_rank)
        grad_W = grad_W.reshape(p * n, target_rank)
        
        m_U = b1 * m_U + (1 - b1) * grad_U
        v_U = b2 * v_U + (1 - b2) * grad_U**2
        U -= lr_t * (m_U / (1 - b1**t)) / (np.sqrt(v_U / (1 - b2**t)) + eps)
        
        m_V = b1 * m_V + (1 - b1) * grad_V
        v_V = b2 * v_V + (1 - b2) * grad_V**2
        V -= lr_t * (m_V / (1 - b1**t)) / (np.sqrt(v_V / (1 - b2**t)) + eps)
        
        m_W = b1 * m_W + (1 - b1) * grad_W
        v_W = b2 * v_W + (1 - b2) * grad_W**2
        W -= lr_t * (m_W / (1 - b1**t)) / (np.sqrt(v_W / (1 - b2**t)) + eps)
    
    if _verify_decomposition(best_U, best_V, best_W, n, m, p):
        return best_U, best_V, best_W, True
    return None, None, None, False

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    U_school, V_school, W_school = _build_schoolbook(n, m, p)
    
    target_ranks = [40, 42, 44, 46, 48, 50, 52]
    
    for target_rank in target_ranks:
        if time.time() > deadline:
            break
        U, V, W, success = _try_optimize_rank(target_rank, n, m, p, deadline, max_iters=1500)
        if success:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
