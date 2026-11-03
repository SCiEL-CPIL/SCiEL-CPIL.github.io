import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the n×m×p matrix multiplication tensor."""
    rank = U.shape[1]
    # Build target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    # Compute decomposition tensor
    T_dec = np.einsum('ir,jr,kr->ijk', U, V, W)
    return np.allclose(T_dec, T, atol=tol)

def _schoolbook(n, m, p):
    """Return the rank n*m*p schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W, rank

def _optimize_subproblem(n, m, p, R, seeds, max_iters=3000, tol=1e-10):
    """Optimize rank-R decomposition for n×m×p subproblem."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    for seed in seeds:
        np.random.seed(seed)
        
        # Diverse initialization strategies
        strategy = seed % 3
        if strategy == 0:
            U = np.random.randn(n * m, R) * 0.5
            V = np.random.randn(m * p, R) * 0.5
            W = np.random.randn(p * n, R) * 0.5
        elif strategy == 1:
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            U /= np.linalg.norm(U, axis=0, keepdims=True) + 1e-8
            V /= np.linalg.norm(V, axis=0, keepdims=True) + 1e-8
            W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-8
        else:
            U = np.random.randn(n * m, R) * 0.3
            V = np.random.randn(m * p, R) * 0.3
            W = np.random.randn(p * n, R) * 0.3
            # Add some structure: initialize with schoolbook-like pattern
            for r in range(min(R, n * m * p)):
                i, j, k = r % n, (r // n) % m, (r // (n * m)) % p
                U[i * m + j, r] += 1.0
                V[j * p + k, r] += 1.0
                W[k * n + i, r] += 1.0
        
        # Adam optimization with adaptive learning rate
        lr = 0.05
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        m_U = np.zeros_like(U)
        m_V = np.zeros_like(V)
        m_W = np.zeros_like(W)
        v_U = np.zeros_like(U)
        v_V = np.zeros_like(V)
        v_W = np.zeros_like(W)
        
        converged = False
        best_loss = np.inf
        
        for it in range(max_iters):
            T_dec = np.einsum('ir,jr,kr->ijk', U, V, W)
            diff = T_dec - T
            loss = np.sum(diff ** 2)
            
            if loss < tol:
                converged = True
                break
            
            best_loss = min(best_loss, loss)
            
            # Gradients
            dU = 2 * np.einsum('ijk,jr,kr->ir', diff, V, W)
            dV = 2 * np.einsum('ijk,ir,kr->jr', diff, U, W)
            dW = 2 * np.einsum('ijk,ir,jr->kr', diff, U, V)
            
            # Adam update
            it1 = it + 1
            m_U = beta1 * m_U + (1 - beta1) * dU
            m_V = beta1 * m_V + (1 - beta1) * dV
            m_W = beta1 * m_W + (1 - beta1) * dW
            v_U = beta2 * v_U + (1 - beta2) * dU ** 2
            v_V = beta2 * v_V + (1 - beta2) * dV ** 2
            v_W = beta2 * v_W + (1 - beta2) * dW ** 2
            m_U_hat = m_U / (1 - beta1 ** it1)
            m_V_hat = m_V / (1 - beta1 ** it1)
            m_W_hat = m_W / (1 - beta1 ** it1)
            v_U_hat = v_U / (1 - beta2 ** it1)
            v_V_hat = v_V / (1 - beta2 ** it1)
            v_W_hat = v_W / (1 - beta2 ** it1)
            
            # Adaptive learning rate
            current_lr = lr * (0.995 ** (it // 500))
            
            U -= current_lr * m_U_hat / (np.sqrt(v_U_hat) + eps)
            V -= current_lr * m_V_hat / (np.sqrt(v_V_hat) + eps)
            W -= current_lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
        
        if converged:
            T_dec = np.einsum('ir,jr,kr->ijk', U, V, W)
            if np.allclose(T_dec, T, atol=tol):
                return U, V, W
    
    return None

def _try_block_decomposition(deadline):
    """
    Exploit the 3×2 block structure: split B (3x6) into three 3x2 blocks.
    For each 3x2 block, we multiply A (2x3) by B_block (3x2) to get C_block (2x2).
    
    If we can find a rank-r decomposition for 2x3x2, then for 2x3x6 we need
    rank = 3*r (one independent copy per block, since the blocks are independent).
    """
    n, m, p = 2, 3, 2  # Subproblem dimensions
    
    # Try ranks from 8 upward with time budget
    for R in [8, 9, 10, 11, 12]:
        if time.time() > deadline - 5:
            break
        
        # Number of seeds based on time remaining
        time_remaining = deadline - time.time() - 5
        max_seeds = min(30, max(5, int(time_remaining * 2)))
        
        result = _optimize_subproblem(n, m, p, R, range(max_seeds), max_iters=2000)
        
        if result is not None:
            U, V, W = result
            # Build full 2x3x6 decomposition
            n_full, m_full, p_full = 2, 3, 6
            R_full = 3 * R
            
            U_full = np.zeros((n_full * m_full, R_full))
            V_full = np.zeros((m_full * p_full, R_full))
            W_full = np.zeros((p_full * n_full, R_full))
            
            for block in range(3):
                offset = block * R
                U_full[:, offset:offset+R] = U
                
                for j in range(m_full):
                    for k in range(2):
                        V_full[j * p_full + (block * 2 + k), offset:offset+R] = V[j * 2 + k]
                
                for k in range(2):
                    for i in range(n_full):
                        W_full[(block * 2 + k) * n_full + i, offset:offset+R] = W[k * n_full + i]
            
            if _verify_decomposition(U_full, V_full, W_full, n_full, m_full, p_full):
                return U_full, V_full, W_full, R_full
    
    return None

def _try_direct_optimization(deadline):
    """
    Try direct optimization on the full 2x3x6 problem with various ranks.
    """
    n, m, p = 2, 3, 6
    
    # Try ranks from 20 upward (schoolbook is 36)
    for R in [20, 22, 24, 26, 28, 30]:
        if time.time() > deadline - 3:
            break
        
        time_remaining = deadline - time.time() - 3
        max_seeds = min(15, max(3, int(time_remaining)))
        
        result = _optimize_subproblem(n, m, p, R, range(max_seeds), max_iters=1500)
        
        if result is not None:
            U, V, W = result
            return U, V, W, R
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    best_solution = None
    best_rank = n * m * p  # 36
    
    # Try block decomposition approach (more likely to find low rank)
    result = _try_block_decomposition(deadline)
    if result is not None:
        U, V, W, R = result
        if _verify_decomposition(U, V, W, n, m, p):
            best_solution = (U, V, W, R)
            best_rank = R
    
    # Try direct optimization if we have time and haven't found something great
    if (best_solution is None or best_rank > 24) and time.time() < deadline - 5:
        result = _try_direct_optimization(deadline)
        if result is not None:
            U, V, W, R = result
            if _verify_decomposition(U, V, W, n, m, p):
                if best_solution is None or R < best_rank:
                    best_solution = (U, V, W, R)
                    best_rank = R
    
    # Fall back to schoolbook
    if best_solution is None:
        U, V, W, rank = _schoolbook(n, m, p)
        best_solution = (U, V, W, rank)
        best_rank = rank
    
    U, V, W, R = best_solution
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
