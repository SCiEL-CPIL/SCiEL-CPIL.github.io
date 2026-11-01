"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[k*n+i, i*m+j, j*p+k] = delta."""
    # Actually the tensor indices for matmul: T[i,j,k] where C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor T has entries T[i, j, k] = 1 if this term contributes, 0 otherwise
    # For decomposition: T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # But the task uses specific index conventions:
    # U has shape (n*m, rank), indexed by (i,j) -> i*m+j
    # V has shape (m*p, rank), indexed by (j,k) -> j*p+k  
    # W has shape (p*n, rank), indexed by (k,i) -> k*n+i
    # The tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1
    # And we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i*m+j, j*p+k, k*n+i]
    # The non-zero entries of T are at positions where the three indices are consistent:
    # For each (i,j,k): T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that the decomposition is exact."""
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Compute reconstructed tensor
    # T_recon[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Build target tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0
    
    max_err = np.max(np.abs(T_recon - T_target))
    return max_err < tol, max_err

def _try_rank(rank, n, m, p, time_limit=30, seed=0):
    """Try to find a rank decomposition with given rank using optimization."""
    np.random.seed(seed)
    U = np.random.randn(n*m, rank) * 0.5
    V = np.random.randn(m*p, rank) * 0.5
    W = np.random.randn(p*n, rank) * 0.5
    
    T_target = _build_tensor(n, m, p)
    
    # Use Adam-like optimization
    lr = 0.01
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    m_U = np.zeros_like(U)
    v_U = np.zeros_like(U)
    m_V = np.zeros_like(V)
    v_V = np.zeros_like(V)
    m_W = np.zeros_like(W)
    v_W = np.zeros_like(W)
    t = 0
    start_time = time.time()
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    while time.time() - start_time < time_limit:
        t += 1
        # Compute loss and gradients
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        loss = np.sum((T_recon - T_target)**2)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < 1e-12:
            break
        
        # Gradients
        dL_dU = 2 * np.einsum('abc,br,cr->ar', T_recon - T_target, V, W)
        dL_dV = 2 * np.einsum('abc,ar,cr->br', T_recon - T_target, U, W)
        dL_dW = 2 * np.einsum('abc,ar,br->cr', T_recon - T_target, U, V)
        
        # Adam update
        m_U = beta1 * m_U + (1 - beta1) * dL_dU
        v_U = beta2 * v_U + (1 - beta2) * dL_dU**2
        m_hat_U = m_U / (1 - beta1**t)
        v_hat_U = v_U / (1 - beta2**t)
        U -= lr * m_hat_U / (np.sqrt(v_hat_U) + eps)
        
        m_V = beta1 * m_V + (1 - beta1) * dL_dV
        v_V = beta2 * v_V + (1 - beta2) * dL_dV**2
        m_hat_V = m_V / (1 - beta1**t)
        v_hat_V = v_V / (1 - beta2**t)
        V -= lr * m_hat_V / (np.sqrt(v_hat_V) + eps)
        
        m_W = beta1 * m_W + (1 - beta1) * dL_dW
        v_W = beta2 * v_W + (1 - beta2) * dL_dW**2
        m_hat_W = m_W / (1 - beta1**t)
        v_hat_W = v_W / (1 - beta2**t)
        W -= lr * m_hat_W / (np.sqrt(v_hat_W) + eps)
    
    # Verify
    ok, err = _verify_decomposition(best_U, best_V, best_W, n, m, p)
    if ok:
        return best_U, best_V, best_W
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with schoolbook as fallback
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = schoolbook_rank
    
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    # Try ranks from 40 to schoolbook_rank-1
    for rank in range(40, schoolbook_rank):
        if time.time() - start_time > deadline:
            break
        
        # Try multiple seeds for this rank
        for seed in range(0, 5):
            if time.time() - start_time > deadline:
                break
            
            result = _try_rank(rank, n, m, p, time_limit=20, seed=seed)
            if result is not None:
                U, V, W = result
                # Convert to lists and verify once more
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                
                ok, err = _verify_decomposition(U_list, V_list, W_list, n, m, p)
                if ok:
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = rank
                    break  # Found a valid decomposition at this rank, move to next lower rank
    
    return best_result
# EVOLVE-BLOCK-END
