# EVOLVE-BLOCK-START
import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def _validate(U, V, W, n, m, p, tol=1e-6):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    nm = n * m
    mp = m * p
    pn = p * n
    rank = U.shape[1]
    
    # Check the "on" positions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                if abs(val - 1.0) > tol:
                    return False, (i, j, k, val)
    
    # Check all "off" positions
    # Since nm*mp*pn is small (9*15*15 = 2025), we can check all
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                i_a = a // m
                j_a = a % m
                j_b = b // p
                k_b = b % p
                i_c = c % n
                k_c = c // n
                if j_a == j_b and i_a == i_c and k_b == k_c:
                    continue
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                if abs(val) > tol:
                    return False, (a, b, c, val)
    
    return True, None

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W

def _gradient_descent_search(n, m, p, target_rank, max_iters=1000, lr=0.01, seed=42, momentum=0.9):
    """
    Use gradient descent with momentum to find a low-rank decomposition.
    """
    rng = np.random.RandomState(seed)
    nm, mp, pn = n*m, m*p, p*n
    
    # Initialize
    U = rng.randn(nm, target_rank) * 0.1
    V = rng.randn(mp, target_rank) * 0.1
    W = rng.randn(pn, target_rank) * 0.1
    
    # Precompute target tensor entries
    target = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0
                
    # Momentum buffers
    vU = np.zeros_like(U)
    vV = np.zeros_like(V)
    vW = np.zeros_like(W)
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        # Compute S = sum_r U_r \otimes V_r \otimes W_r
        S = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Residual
        R = S - target
        loss = np.sum(R**2)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if best_loss < 1e-10:
            break
            
        # Gradients
        # dL/dU[a,r] = 2 * sum_{b,c} R[a,b,c] V[b,r] W[c,r]
        dU = 2 * np.einsum('abc,br,cr->ar', R, V, W)
        dV = 2 * np.einsum('abc,ar,cr->br', R, U, W)
        dW = 2 * np.einsum('abc,ar,br->cr', R, U, V)
        
        # Momentum update
        vU = momentum * vU + (1 - momentum) * dU
        vV = momentum * vV + (1 - momentum) * dV
        vW = momentum * vW + (1 - momentum) * dW
        
        U -= lr * vU
        V -= lr * vV
        W -= lr * vW
        
        # Optional: Decay learning rate
        # lr *= 0.999
        
    return best_U, best_V, best_W, best_loss

def _try_rounding(U, V, W, n, m, p, tol=1e-4):
    """
    Try to round U, V, W to integers or half-integers and check validity.
    """
    # Try rounding to nearest integer
    U_int = np.round(U)
    V_int = np.round(V)
    W_int = np.round(W)
    valid, _ = _validate(U_int, V_int, W_int, n, m, p, tol=1e-6)
    if valid:
        return U_int, V_int, W_int, True
    
    # Try rounding to nearest half-integer
    U_half = np.round(U * 2) / 2
    V_half = np.round(V * 2) / 2
    W_half = np.round(W * 2) / 2
    valid, _ = _validate(U_half, V_half, W_half, n, m, p, tol=1e-6)
    if valid:
        return U_half, V_half, W_half, True
        
    return None, None, None, False

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = (U_school, V_school, W_school)
    
    # Try ranks from 20 to 40
    # We can afford to try several ranks with GD
    for target_rank in range(20, 41):
        if time.time() > deadline - 10:
            break
            
        # Run GD with a few different seeds
        for seed in [42, 123, 456]:
            if time.time() > deadline - 5:
                break
                
            U, V, W, loss = _gradient_descent_search(n, m, p, target_rank, max_iters=500, lr=0.05, seed=seed, momentum=0.9)
            
            if loss < 1e-4:
                # Check if it's close enough to be rounded
                U_r, V_r, W_r, success = _try_rounding(U, V, W, n, m, p)
                if success:
                    best_rank = target_rank
                    best_result = (U_r, V_r, W_r)
                    break
        if best_rank == target_rank:
            break
            
    U, V, W = best_result
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
