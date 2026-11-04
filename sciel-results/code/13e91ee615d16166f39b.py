"""Program for matrix multiplication 2x3x7 rank minimization."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 7]

def _verify(U, V, W, n=2, m=3, p=7, tol=1e-9):
    """Verify that U, V, W decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reshape to (n, m, rank), (m, p, rank), (p, n, rank)
    U_t = U_arr.reshape(n, m, rank)
    V_t = V_arr.reshape(m, p, rank)
    W_t = W_arr.reshape(p, n, rank)
    
    # Compute S[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Using einsum: 'ijc,jkc,kic->ijk'
    S = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
    
    # Check if S is all ones
    if not np.allclose(S, 1.0, atol=tol):
        return False
    return True

def _structured_init(n, m, p, rank, seed=None):
    """Initialize with structured patterns to help convergence."""
    if seed is not None:
        np.random.seed(seed)
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Try to assign each (i,j,k) to a column with some probability
    # This gives a better starting point than pure random
    for c in range(rank):
        # Each column gets a random assignment
        if c < n*m*p // rank:
            # Assign to a specific (i,j,k)
            idx = c * (n*m*p) // rank
            i = idx // (m*p)
            j = (idx // p) % m
            k = idx % p
            U[i*m+j, c] = 1.0
            V[j*p+k, c] = 1.0
            W[k*n+i, c] = 1.0
    
    # Add small random perturbation
    U += np.random.randn(n*m, rank) * 0.1
    V += np.random.randn(m*p, rank) * 0.1
    W += np.random.randn(p*n, rank) * 0.1
    
    return U.tolist(), V.tolist(), W.tolist()

def _random_init(n, m, p, rank, seed=None):
    if seed is not None:
        np.random.seed(seed)
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    return U.tolist(), V.tolist(), W.tolist()

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _adam_gradient_descent(U, V, W, n, m, p, max_iter=3000, lr=0.05, tol=1e-10, deadline=None):
    """Adam optimizer for gradient descent to minimize ||S - 1||^2."""
    U_arr = np.array(U, dtype=float).reshape(n, m, -1)
    V_arr = np.array(V, dtype=float).reshape(m, p, -1)
    W_arr = np.array(W, dtype=float).reshape(p, n, -1)
    rank = U_arr.shape[2]
    
    # Adam state
    m_U = np.zeros_like(U_arr)
    v_U = np.zeros_like(U_arr)
    m_V = np.zeros_like(V_arr)
    v_V = np.zeros_like(V_arr)
    m_W = np.zeros_like(W_arr)
    v_W = np.zeros_like(W_arr)
    
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    
    best_loss = np.inf
    best_U = U_arr.copy()
    best_V = V_arr.copy()
    best_W = W_arr.copy()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        t += 1
        
        # Compute S = sum_c U[:,:,c] * V[:,:,c] * W[:,:,c] (transposed indices)
        # S[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        S = np.einsum('ijc,jkc,kic->ijk', U_arr, V_arr, W_arr)
        
        # Loss = sum (S - 1)^2
        diff = S - 1.0
        loss = np.sum(diff ** 2)
        
        if loss < best_loss:
            best_loss = loss
            best_U = U_arr.copy()
            best_V = V_arr.copy()
            best_W = W_arr.copy()
        
        if loss < tol:
            break
        
        # Compute gradients
        # dL/dU[i,j,c] = 2 * (S[i,j,k] - 1) * V[j,k,c] * W[k,i,c]
        dU = 2.0 * np.einsum('ijk,jkc,kic->ijc', diff, V_arr, W_arr)
        dV = 2.0 * np.einsum('ijk,ijc,kic->jkc', diff, U_arr, W_arr)
        dW = 2.0 * np.einsum('ijk,ijc,jkc->kic', diff, U_arr, V_arr)
        
        # Adam update
        m_U = beta1 * m_U + (1 - beta1) * dU
        v_U = beta2 * v_U + (1 - beta2) * (dU ** 2)
        m_U_hat = m_U / (1 - beta1 ** t)
        v_U_hat = v_U / (1 - beta2 ** t)
        U_arr -= lr * m_U_hat / (np.sqrt(v_U_hat) + eps)
        
        m_V = beta1 * m_V + (1 - beta1) * dV
        v_V = beta2 * v_V + (1 - beta2) * (dV ** 2)
        m_V_hat = m_V / (1 - beta1 ** t)
        v_V_hat = v_V / (1 - beta2 ** t)
        V_arr -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)
        
        m_W = beta1 * m_W + (1 - beta1) * dW
        v_W = beta2 * v_W + (1 - beta2) * (dW ** 2)
        m_W_hat = m_W / (1 - beta1 ** t)
        v_W_hat = v_W / (1 - beta2 ** t)
        W_arr -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
    
    return best_U.reshape(n*m, rank).tolist(), best_V.reshape(m*p, rank).tolist(), best_W.reshape(p*n, rank).tolist()

def _try_round_and_verify(U, V, W, n, m, p, tol=1e-9):
    """Try to round to half-integers and verify."""
    U_r = [[round(x * 2) / 2 for x in row] for row in U]
    V_r = [[round(x * 2) / 2 for x in row] for row in V]
    W_r = [[round(x * 2) / 2 for x in row] for row in W]
    
    if _verify(U_r, V_r, W_r, n, m, p, tol=tol):
        return U_r, V_r, W_r
    return None

def _polish(U, V, W, n, m, p, max_iter=1000, lr=0.001, tol=1e-12, deadline=None):
    """Final high-precision polish pass with smaller learning rate."""
    U_arr = np.array(U, dtype=float).reshape(n, m, -1)
    V_arr = np.array(V, dtype=float).reshape(m, p, -1)
    W_arr = np.array(W, dtype=float).reshape(p, n, -1)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        S = np.einsum('ijc,jkc,kic->ijk', U_arr, V_arr, W_arr)
        diff = S - 1.0
        loss = np.sum(diff ** 2)
        
        if loss < tol:
            break
        
        dU = 2.0 * np.einsum('ijk,jkc,kic->ijc', diff, V_arr, W_arr)
        dV = 2.0 * np.einsum('ijk,ijc,kic->jkc', diff, U_arr, W_arr)
        dW = 2.0 * np.einsum('ijk,ijc,jkc->kic', diff, U_arr, V_arr)
        
        U_arr -= lr * dU
        V_arr -= lr * dV
        W_arr -= lr * dW
    
    return U_arr.reshape(n*m, -1).tolist(), V_arr.reshape(m*p, -1).tolist(), W_arr.reshape(p*n, -1).tolist()

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin out of 150s
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 42
    
    # Start with schoolbook decomposition
    best = _schoolbook(n, m, p)
    best_rank = schoolbook_rank
    
    # Lower bound is max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    lower_bound = max(n*m, m*p, p*n)
    
    # Try ranks from lower_bound to best_rank - 1
    for target_rank in range(lower_bound, best_rank):
        if time.time() > deadline - 20:
            break
        
        # Try multiple initializations with Adam gradient descent
        max_attempts = 50
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 15:
                break
            
            seed = target_rank * 10000 + attempt
            
            # Alternate between structured and random initialization
            if attempt % 2 == 0:
                U_init, V_init, W_init = _structured_init(n, m, p, target_rank, seed=seed)
            else:
                U_init, V_init, W_init = _random_init(n, m, p, target_rank, seed=seed)
            
            # Run Adam gradient descent
            U_opt, V_opt, W_opt = _adam_gradient_descent(
                U_init, V_init, W_init, n, m, p,
                max_iter=2000, lr=0.05, tol=1e-10,
                deadline=deadline - 10
            )
            
            # Check if we found a valid solution
            if _verify(U_opt, V_opt, W_opt, n, m, p, tol=1e-6):
                # Try to round to half-integers
                result = _try_round_and_verify(U_opt, V_opt, W_opt, n, m, p, tol=1e-9)
                if result is not None:
                    U_r, V_r, W_r = result
                    best = {"U": U_r, "V": V_r, "W": W_r}
                    best_rank = target_rank
                    found = True
                    break
                else:
                    # If rounding fails, do a polish pass and try again
                    if _verify(U_opt, V_opt, W_opt, n, m, p, tol=1e-8):
                        U_pol, V_pol, W_pol = _polish(
                            U_opt, V_opt, W_opt, n, m, p,
                            max_iter=500, lr=0.001, tol=1e-12,
                            deadline=deadline - 5
                        )
                        result = _try_round_and_verify(U_pol, V_pol, W_pol, n, m, p, tol=1e-9)
                        if result is not None:
                            U_r, V_r, W_r = result
                            best = {"U": U_r, "V": V_r, "W": W_r}
                            best_rank = target_rank
                            found = True
                            break
        
        if best_rank == target_rank:
            break
    
    # Return the best found
    return best
