"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    # Expected: T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(T[a, b, c] - 1.0) > tol:
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
    if np.max(np.abs(T_check)) > tol:
        return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition."""
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
    return U, V, W

def _build_target_tensor(n, m, p):
    """Build the target tensor for matrix multiplication."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _gradient_descent(U, V, W, T_target, n, m, p, max_iter=2000, tol=1e-10, seed=42):
    """Gradient descent with adaptive learning rate and restarts."""
    rng = np.random.RandomState(seed)
    U = U.copy().astype(float)
    V = V.copy().astype(float)
    W = W.copy().astype(float)
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    lr = 0.05
    patience = 0
    best_patience_err = float('inf')
    
    for it in range(max_iter):
        # Compute current tensor
        T_curr = np.einsum('ar,br,cr->abc', U, V, W)
        residual = T_curr - T_target
        err = np.sum(residual ** 2)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if err < best_patience_err - 1e-6:
                best_patience_err = err
                patience = 0
            else:
                patience += 1
        
        if best_err < tol:
            break
        
        if patience > 100:
            # Restart with perturbation
            U = best_U + rng.randn(*best_U.shape) * 0.1
            V = best_V + rng.randn(*best_V.shape) * 0.1
            W = best_W + rng.randn(*best_W.shape) * 0.1
            patience = 0
            lr = max(0.01, lr * 0.5)
            continue
        
        # Gradient computation
        grad_U = np.einsum('abc,br,cr->ar', residual, V, W)
        grad_V = np.einsum('abc,ar,cr->br', residual, U, W)
        grad_W = np.einsum('abc,ar,br->cr', residual, U, V)
        
        # Scale gradients by the "magnitude" of each factor
        U_norm = np.linalg.norm(U, axis=0, keepdims=True) + 1e-12
        V_norm = np.linalg.norm(V, axis=0, keepdims=True) + 1e-12
        W_norm = np.linalg.norm(W, axis=0, keepdims=True) + 1e-12
        
        grad_U = grad_U / (V_norm * W_norm)
        grad_V = grad_V / (U_norm * W_norm)
        grad_W = grad_W / (U_norm * V_norm)
        
        # Update
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Decay learning rate
        lr *= 0.999
    
    return best_U, best_V, best_W, best_err

def _structured_init(n, m, p, target_rank, rng):
    """Create structured initialization exploiting Kronecker/block structure."""
    # For 2x3 * 3x7, we can try to exploit the fact that the 3x7 multiplication
    # can be seen as combining smaller multiplications
    
    # Strategy: Use a block structure where we split the 3x7 into blocks
    # and use Strassen-like ideas for rectangular matrices
    
    U = np.zeros((n * m, target_rank))
    V = np.zeros((m * p, target_rank))
    W = np.zeros((p * n, target_rank))
    
    # Try different structured patterns
    pattern = rng.randint(0, 4)
    
    if pattern == 0:
        # Kronecker-like: split into 2x3 and 1x7 parts
        # Use rank-3 for the 2x3 part and rank-7 for the 1x7 part
        r1 = min(6, target_rank // 2)  # for 2x3
        r2 = min(14, target_rank - r1)  # for 1x7
        
        # Initialize first block (2x3 multiplication structure)
        for r in range(r1):
            i = rng.randint(0, n)
            j = rng.randint(0, m)
            U[i * m + j, r] = 1.0
            V[j * p + rng.randint(0, p), r] = 1.0
            W[rng.randint(0, p) * n + i, r] = 1.0
        
        # Initialize second block (1x7 multiplication structure)
        for r in range(r1, r1 + r2):
            i = rng.randint(0, n)
            j = rng.randint(0, m)
            k = rng.randint(0, p)
            U[i * m + j, r] = 1.0
            V[j * p + k, r] = 1.0
            W[k * n + i, r] = 1.0
        
        # Add random components for the rest
        for r in range(r1 + r2, target_rank):
            idx_u = rng.randint(0, n * m)
            idx_v = rng.randint(0, m * p)
            idx_w = rng.randint(0, p * n)
            U[idx_u, r] = 1.0
            V[idx_v, r] = 1.0
            W[idx_w, r] = 1.0
            
    elif pattern == 1:
        # Block diagonal structure
        block_size = max(1, target_rank // 7)
        for block in range(7):
            start = block * block_size
            end = min((block + 1) * block_size, target_rank)
            for r in range(start, end):
                i = rng.randint(0, n)
                j = rng.randint(0, m)
                k = (block % p)
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
    elif pattern == 2:
        # Strassen-inspired for rectangular
        # Split 3x7 into two parts: 3x3 and 3x4
        r_3x3 = min(9, target_rank // 2)
        r_3x4 = min(12, target_rank - r_3x3)
        
        for r in range(r_3x3):
            i = rng.randint(0, n)
            j = rng.randint(0, m)
            k = rng.randint(0, 3)
            U[i * m + j, r] = 1.0
            V[j * p + k, r] = 1.0
            W[k * n + i, r] = 1.0
        
        for r in range(r_3x3, r_3x3 + r_3x4):
            i = rng.randint(0, n)
            j = rng.randint(0, m)
            k = 3 + rng.randint(0, 4)
            U[i * m + j, r] = 1.0
            V[j * p + k, r] = 1.0
            W[k * n + i, r] = 1.0
    else:
        # Random sparse initialization
        for r in range(target_rank):
            idx_u = rng.randint(0, n * m)
            idx_v = rng.randint(0, m * p)
            idx_w = rng.randint(0, p * n)
            U[idx_u, r] = 1.0
            V[idx_v, r] = 1.0
            W[idx_w, r] = 1.0
    
    # Add small random noise to break symmetry
    U += rng.randn(n * m, target_rank) * 0.01
    V += rng.randn(m * p, target_rank) * 0.01
    W += rng.randn(p * n, target_rank) * 0.01
    
    return U, V, W

def _try_rank(n, m, p, target_rank, T_target, max_seeds=5, max_iter=1500, tol=1e-10, deadline=None):
    """Try to find a decomposition of given rank."""
    rng = np.random.RandomState(12345)
    
    for seed in range(max_seeds):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        # Try different initialization strategies
        if seed % 2 == 0:
            # Structured initialization
            U, V, W = _structured_init(n, m, p, target_rank, rng)
        else:
            # Random initialization
            scale = 1.0 / (target_rank ** 0.5)
            U = rng.randn(n * m, target_rank) * scale
            V = rng.randn(m * p, target_rank) * scale
            W = rng.randn(p * n, target_rank) * scale
        
        U, V, W, err = _gradient_descent(U, V, W, T_target, n, m, p, 
                                          max_iter=max_iter, tol=tol, seed=seed)
        
        if err < tol:
            # Verify
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=tol):
                return U, V, W, True
    
    return None, None, None, False

def _polish_decomposition(U, V, W, T_target, n, m, p, tol=1e-12, max_iter=500):
    """High-precision polish pass using Newton-like iterations."""
    U = U.copy().astype(float)
    V = V.copy().astype(float)
    W = W.copy().astype(float)
    rank = U.shape[1]
    
    for it in range(max_iter):
        T_curr = np.einsum('ar,br,cr->abc', U, V, W)
        residual = T_curr - T_target
        err = np.sum(residual ** 2)
        
        if err < tol:
            break
        
        # Compute gradients
        grad_U = np.einsum('abc,br,cr->ar', residual, V, W)
        grad_V = np.einsum('abc,ar,cr->br', residual, U, W)
        grad_W = np.einsum('abc,ar,br->cr', residual, U, V)
        
        # Compute approximate Hessian diagonal for scaling
        # Use norm-based scaling
        U_norm = np.linalg.norm(U, axis=0, keepdims=True) + 1e-12
        V_norm = np.linalg.norm(V, axis=0, keepdims=True) + 1e-12
        W_norm = np.linalg.norm(W, axis=0, keepdims=True) + 1e-12
        
        # Adaptive learning rate based on error
        lr = min(0.1, max(0.001, 0.1 * np.sqrt(tol / max(err, tol))))
        
        grad_U = grad_U / (V_norm * W_norm)
        grad_V = grad_V / (U_norm * W_norm)
        grad_W = grad_W / (U_norm * V_norm)
        
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = n * m * p  # 42
    
    # Build target tensor once
    T_target = _build_target_tensor(n, m, p)
    
    # Try to find better decompositions using local search
    # Start from lower ranks and go up
    for target_rank in range(30, 42):
        if time.time() > deadline - 20:
            break
        
        U, V, W, success = _try_rank(n, m, p, target_rank, T_target, 
                                       max_seeds=4, max_iter=1200, 
                                       tol=1e-10, deadline=deadline)
        
        if success:
            # Polish the solution
            U_pol, V_pol, W_pol = _polish_decomposition(U, V, W, T_target, n, m, p)
            if _verify(U_pol.tolist(), V_pol.tolist(), W_pol.tolist(), n, m, p, tol=1e-9):
                best_result = {"U": U_pol.tolist(), "V": V_pol.tolist(), "W": W_pol.tolist()}
                best_rank = target_rank
            else:
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = target_rank
            # Continue trying lower ranks
            continue
    
    # If we found a good rank, try to polish it further
    if best_rank < n * m * p and time.time() < deadline - 10:
        U_curr = np.array(best_result["U"])
        V_curr = np.array(best_result["V"])
        W_curr = np.array(best_result["W"])
        curr_rank = U_curr.shape[1]
        
        # Try to reduce rank by 1
        for target_rank in range(max(20, curr_rank - 5), curr_rank):
            if time.time() > deadline - 5:
                break
            
            U, V, W, success = _try_rank(n, m, p, target_rank, T_target,
                                           max_seeds=3, max_iter=800,
                                           tol=1e-10, deadline=deadline)
            if success:
                U_pol, V_pol, W_pol = _polish_decomposition(U, V, W, T_target, n, m, p)
                if _verify(U_pol.tolist(), V_pol.tolist(), W_pol.tolist(), n, m, p, tol=1e-9):
                    best_result = {"U": U_pol.tolist(), "V": V_pol.tolist(), "W": W_pol.tolist()}
                    best_rank = target_rank
                else:
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = target_rank
    
    return best_result
# EVOLVE-BLOCK-END
