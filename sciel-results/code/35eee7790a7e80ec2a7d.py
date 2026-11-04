import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def _build_target_tensor():
    n, m, p = DIMENSIONS
    # Target tensor: T[i,j, j',k, k',i'] = delta[i,i'] * delta[j,j'] * delta[k,k']
    # We'll represent it as a 3D array of shape (n*m, m*p, p*n)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _reconstruction_error(U, V, W, T):
    """Compute Frobenius norm of difference between reconstructed and target tensor."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    # Reconstruct: R[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # Use einsum for efficiency
    R = np.einsum('ar,br,cr->abc', U, V, W)
    return np.linalg.norm(R - T)

def _try_rank(rank, T, max_iters=5000, seed=0, time_limit=10):
    """Try to find a rank-r decomposition using gradient descent."""
    n, m, p = DIMENSIONS
    rng = np.random.default_rng(seed)
    
    best_error = np.inf
    best_U, best_V, best_W = None, None, None
    
    for attempt in range(3):
        # Random initialization
        U = rng.standard_normal((n * m, rank)) * 0.1
        V = rng.standard_normal((m * p, rank)) * 0.1
        W = rng.standard_normal((p * n, rank)) * 0.1
        
        # Use Adam-like optimization
        lr = 0.01
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        
        mU = np.zeros_like(U)
        mV = np.zeros_like(V)
        mW = np.zeros_like(W)
        vU = np.zeros_like(U)
        vV = np.zeros_like(V)
        vW = np.zeros_like(W)
        
        t = 0
        start_time = time.time()
        
        while t < max_iters:
            if time.time() - start_time > time_limit:
                break
            
            t += 1
            lr_t = lr * np.sqrt(1 - beta2**t) / (1 - beta1**t)
            
            # Compute gradient
            R = np.einsum('ar,br,cr->abc', U, V, W)
            E = R - T
            
            # dE/dU[a,r] = sum_{b,c} E[a,b,c] * V[b,r] * W[c,r]
            grad_U = np.einsum('abc,br,cr->ar', E, V, W)
            grad_V = np.einsum('abc,ar,cr->br', E, U, W)
            grad_W = np.einsum('abc,ar,br->cr', E, U, V)
            
            # Update moments
            mU = beta1 * mU + (1 - beta1) * grad_U
            mV = beta1 * mV + (1 - beta1) * grad_V
            mW = beta1 * mW + (1 - beta1) * grad_W
            vU = beta2 * vU + (1 - beta2) * grad_U**2
            vV = beta2 * vV + (1 - beta2) * grad_V**2
            vW = beta2 * vW + (1 - beta2) * grad_W**2
            
            # Update parameters
            U -= lr_t * mU / (np.sqrt(vU) + eps)
            V -= lr_t * mV / (np.sqrt(vV) + eps)
            W -= lr_t * mW / (np.sqrt(vW) + eps)
            
            err = _reconstruction_error(U, V, W, T)
            if err < best_error:
                best_error = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if err < 1e-10:
                break
        
        if best_error < 1e-10:
            break
    
    return best_error, best_U, best_V, best_W

def _snap_to_rational(M, denom_max=1000):
    """Snap matrix entries to rationals with small denominators."""
    from fractions import Fraction
    snapped = np.zeros_like(M)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            val = M[i, j]
            if abs(val) < 1e-12:
                snapped[i, j] = 0.0
                continue
            best_frac = None
            best_err = np.inf
            for d in range(1, denom_max + 1):
                num = round(val * d)
                if abs(num) > 10**6:
                    continue
                frac_val = num / d
                err = abs(val - frac_val)
                if err < best_err:
                    best_err = err
                    best_frac = frac_val
                if err < 1e-12:
                    break
            snapped[i, j] = best_frac if best_frac is not None else val
    return snapped

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that the decomposition exactly matches the target tensor."""
    R = np.einsum('ar,br,cr->abc', U, V, W)
    return np.linalg.norm(R - T) < tol

def construct():
    start_time = time.time()
    time_limit = 140  # Leave 10s margin
    
    n, m, p = DIMENSIONS
    T = _build_target_tensor()
    
    # Start from schoolbook rank 24
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Verify schoolbook works
    assert _verify_decomposition(U, V, W, T)
    
    best_rank = rank
    best_result = (U.copy(), V.copy(), W.copy())
    
    # Try to find lower rank decompositions
    for r in range(12, rank):
        if time.time() - start_time > time_limit - 10:
            break
        
        # Allocate time per rank attempt
        time_per_rank = min(15, (time_limit - 10 - (time.time() - start_time)) / max(1, rank - r))
        
        err, U_r, V_r, W_r = _try_rank(r, T, max_iters=3000, seed=r, time_limit=time_per_rank)
        
        if err < 1e-8:
            # Try to snap to rationals
            U_s = _snap_to_rational(U_r, denom_max=100)
            V_s = _snap_to_rational(V_r, denom_max=100)
            W_s = _snap_to_rational(W_r, denom_max=100)
            
            if _verify_decomposition(U_s, V_s, W_s, T):
                best_rank = r
                best_result = (U_s, V_s, W_s)
                break
            else:
                # Try with original (non-snapped) values
                if _verify_decomposition(U_r, V_r, W_r, T):
                    best_rank = r
                    best_result = (U_r, V_r, W_r)
                    break
    
    # Convert to lists
    U_out = best_result[0].tolist()
    V_out = best_result[1].tolist()
    W_out = best_result[2].tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
