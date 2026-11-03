"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START

def _validate(U, V, W, n, m, p):
    """Check if the decomposition is exact."""
    rank = len(V[0]) if V else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Reshape to 3D
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # Compute tensor
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    # Check if T is all ones
    if not np.allclose(T, np.ones((n, m, p)), atol=1e-9):
        return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition."""
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

def _error_fast(U, V, W, n, m, p, r):
    """Compute the error using vectorized operations."""
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.sum((T - 1.0) ** 2)

def _gradient_descent(n, m, p, rank, max_iters=2000, lr=0.01, tol=1e-8, seed=42):
    """
    Run gradient descent to find a low-rank decomposition.
    Returns (U, V, W, loss)
    """
    np.random.seed(seed)
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Initialize with small random values
    U = np.random.randn(dim_U, rank) * 0.1
    V = np.random.randn(dim_V, rank) * 0.1
    W = np.random.randn(dim_W, rank) * 0.1
    
    vU = np.zeros_like(U)
    vV = np.zeros_like(V)
    vW = np.zeros_like(W)
    momentum = 0.9
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        target = np.ones((n, m, p))
        loss = np.sum((T - target) ** 2)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < tol:
            break
        
        # Compute gradients
        dT = 2.0 * (T - target)
        
        dU3 = np.einsum('ijk,jkc,kic->ijc', dT, V3, W3)
        dU = dU3.reshape(dim_U, rank)
        
        dV3 = np.einsum('ijk,ijc,kic->jkc', dT, U3, W3)
        dV = dV3.reshape(dim_V, rank)
        
        dW3 = np.einsum('ijk,ijc,jkc->kic', dT, U3, V3)
        dW = dW3.reshape(dim_W, rank)
        
        # Update
        vU = momentum * vU + (1 - momentum) * dU
        vV = momentum * vV + (1 - momentum) * dV
        vW = momentum * vW + (1 - momentum) * dW
        
        U -= lr * vU
        V -= lr * vV
        W -= lr * vW
        
    return best_U, best_V, best_W, best_loss

def _discrete_polish(U, V, W, n, m, p, r, iters=500, seed=123):
    """
    Perform a short discrete local search to polish a continuous solution.
    Assumes U, V, W are already close to integer/half-integer.
    """
    coeffs = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    rng = random.Random(seed)
    
    # Round to nearest 0.5 first
    U = np.round(U * 2) / 2
    V = np.round(V * 2) / 2
    W = np.round(W * 2) / 2
    
    cur_err = _error_fast(U, V, W, n, m, p, r)
    best_err = cur_err
    best_state = (U.copy(), V.copy(), W.copy())
    
    for it in range(iters):
        which = rng.randint(0, 2)
        if which == 0:
            row = rng.randint(0, n*m-1); col = rng.randint(0, r-1)
            old = U[row, col]; U[row, col] = rng.choice(coeffs)
            new_err = _error_fast(U, V, W, n, m, p, r)
            if new_err < cur_err:
                cur_err = new_err
                if new_err < best_err:
                    best_err = new_err
                    best_state = (U.copy(), V.copy(), W.copy())
                    if best_err < 1e-9:
                        break
            else:
                U[row, col] = old
        elif which == 1:
            row = rng.randint(0, m*p-1); col = rng.randint(0, r-1)
            old = V[row, col]; V[row, col] = rng.choice(coeffs)
            new_err = _error_fast(U, V, W, n, m, p, r)
            if new_err < cur_err:
                cur_err = new_err
                if new_err < best_err:
                    best_err = new_err
                    best_state = (U.copy(), V.copy(), W.copy())
                    if best_err < 1e-9:
                        break
            else:
                V[row, col] = old
        else:
            row = rng.randint(0, p*n-1); col = rng.randint(0, r-1)
            old = W[row, col]; W[row, col] = rng.choice(coeffs)
            new_err = _error_fast(U, V, W, n, m, p, r)
            if new_err < cur_err:
                cur_err = new_err
                if new_err < best_err:
                    best_err = new_err
                    best_state = (U.copy(), V.copy(), W.copy())
                    if best_err < 1e-9:
                        break
            else:
                W[row, col] = old
                
    return best_err, best_state

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    schoolbook = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = schoolbook
    
    # Try ranks from lower bound up to schoolbook - 1
    # Lower bound is max(n*m, m*p, p*n) = 15. 
    # But we know 3x3x3 is 21, so 3x3x5 is likely > 21.
    # Let's try from 25 to 44.
    for r in range(25, best_rank):
        if time.time() > deadline - 10:
            break
            
        found = False
        # Try a few random seeds for gradient descent
        for seed in range(5):
            if time.time() > deadline - 5:
                break
                
            # 1. Continuous search
            U, V, W, loss = _gradient_descent(n, m, p, r, max_iters=1000, seed=seed)
            
            if loss < 1e-4:
                # 2. Discrete polish
                err, state = _discrete_polish(U, V, W, n, m, p, r, iters=200, seed=seed)
                if err < 1e-9:
                    U2, V2, W2 = state
                    # Final verification
                    if _validate(U2.tolist(), V2.tolist(), W2.tolist(), n, m, p):
                        best_rank = r
                        best_result = {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
                        found = True
                        break
        
        if found:
            # If we found a rank r, we could try lower ranks, but 
            # usually if r works, r-1 might not. 
            # The loop goes from low to high, so the first one found is minimal.
            return best_result
            
    return best_result

# EVOLVE-BLOCK-END
