import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build the full tensor and check
    T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    # Expected: T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(T[a, b, c] - 1.0) > 1e-9:
                    return False
    
    # Check all other entries are zero
    T_check = T.copy()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_check[a, b, c] = 0
    if np.max(np.abs(T_check)) > 1e-9:
        return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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

def _build_target_tensor(n, m, p):
    """Build the target multiplication tensor T[a, b, c] where a=i*m+j, b=j*p+k, c=k*n+i."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _structured_init(n, m, p, target_rank, seed=0):
    """
    Initialize U, V, W with a structured pattern that is close to schoolbook
    but with random perturbations to escape local minima.
    """
    np.random.seed(seed)
    U = np.random.normal(0, 0.5, size=(n * m, target_rank))
    V = np.random.normal(0, 0.5, size=(m * p, target_rank))
    W = np.random.normal(0, 0.5, size=(p * n, target_rank))
    
    # Add schoolbook-like structure to the first min(target_rank, n*m*p) columns
    n_terms = min(target_rank, n * m * p)
    for col in range(n_terms):
        i = col // (m * p)
        rem = col % (m * p)
        j = rem // p
        k = rem % p
        a = i * m + j
        b = j * p + k
        c = k * n + i
        U[a, col] += 1.0
        V[b, col] += 1.0
        W[c, col] += 1.0
    
    return U, V, V, W

def _gradient_descent_momentum(U, V, W, T, lr=0.01, momentum=0.9, n_iter=500, tol=1e-10):
    """
    Perform gradient descent with momentum to minimize ||sum_r U[:,r] V[:,r] W[:,r] - T||_F^2.
    """
    n_m = U.shape[0]
    m_p = V.shape[0]
    p_n = W.shape[0]
    rank = U.shape[1]
    
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    vU = np.zeros_like(U)
    vV = np.zeros_like(V)
    vW = np.zeros_like(W)
    
    prev_err = float('inf')
    
    for it in range(n_iter):
        # Compute current tensor
        T_cur = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.sum((T_cur - T) ** 2)
        
        if abs(prev_err - err) < tol:
            break
        prev_err = err
        
        # Compute gradients
        # d/dU[a,r] = sum_{b,c} (T_cur[a,b,c] - T[a,b,c]) * V[b,r] * W[c,r]
        # d/dV[b,r] = sum_{a,c} (T_cur[a,b,c] - T[a,b,c]) * U[a,r] * W[c,r]
        # d/dW[c,r] = sum_{a,b} (T_cur[a,b,c] - T[a,b,c]) * U[a,r] * V[b,r]
        
        diff = T_cur - T
        
        gU = np.einsum('abc,br,cr->ar', diff, V, W)
        gV = np.einsum('abc,ar,cr->br', diff, U, W)
        gW = np.einsum('abc,ar,br->cr', diff, U, V)
        
        # Momentum update
        vU = momentum * vU + lr * gU
        vV = momentum * vV + lr * gV
        vW = momentum * vW + lr * gW
        
        U -= vU
        V -= vV
        W -= vW
    
    return U, V, W, err

def _polish(U, V, W, n, m, p, n_iter=200, lr=0.001):
    """
    Final high-precision polish pass using smaller learning rate.
    """
    T = _build_target_tensor(n, m, p)
    return _gradient_descent_momentum(U, V, W, T, lr=lr, momentum=0.9, n_iter=n_iter, tol=1e-12)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Start with schoolbook as fallback
    schoolbook_result = _schoolbook(n, m, p)
    best_result = schoolbook_result
    best_rank = n * m * p
    
    T = _build_target_tensor(n, m, p)
    
    # Try to find better decompositions
    # Strategy: Use gradient descent with structured initialization
    # Try various ranks from low to high
    
    # Known: R(2,2,2) = 7 (Strassen)
    # For (2,3,7), we can try various ranks
    
    # Try ranks from 24 to 41
    for target_rank in range(24, best_rank):
        if time.time() > deadline - 15:
            break
        
        # Try multiple restarts for this rank
        max_restarts = 3
        for r in range(max_restarts):
            if time.time() > deadline - 15:
                break
            
            # Structured initialization
            U, V, W = _structured_init(n, m, p, target_rank, seed=r)
            
            # Gradient descent with momentum
            U, V, W, err = _gradient_descent_momentum(U, V, W, T, lr=0.01, momentum=0.9, n_iter=300)
            
            # If error is small enough, try polish
            if err < 1e-4:
                U, V, W, err = _polish(U, V, W, n, m, p, n_iter=100, lr=0.001)
            
            # Check if we found a valid decomposition
            if err < 1e-8:
                # Round to nearest 0.5
                U = np.round(U * 2) / 2
                V = np.round(V * 2) / 2
                W = np.round(W * 2) / 2
                
                # Verify exactness
                if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = target_rank
                    break
                else:
                    # Rounding broke it, try without rounding
                    if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        best_rank = target_rank
                        break
        
        # If we found a valid decomposition at this rank, we can stop
        if best_rank == target_rank:
            break
    
    return best_result
