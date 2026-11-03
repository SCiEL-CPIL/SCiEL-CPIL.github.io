"""Rank-minimal exact tensor decomposition for 3x4x4 matrix multiplication."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition of rank n*m*p."""
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

def _check_exact(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition is exact for all (i,j,k)."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _max_error(U, V, W, n, m, p):
    """Compute the maximum absolute error over all (i,j,k)."""
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                max_err = max(max_err, abs(val - 1.0))
    return max_err

def _snap_to_half_integer(arr, tol=1e-6):
    """Snap array entries to nearest half-integer if close enough."""
    snapped = np.round(arr * 2) / 2
    if np.max(np.abs(arr - snapped)) < tol:
        return snapped, True
    return arr, False

def _try_snap_and_check(U, V, W, n, m, p):
    """Try to snap to half-integers and check exactness."""
    for tol in [1e-5, 1e-4, 1e-3, 1e-2]:
        U_snapped, u_ok = _snap_to_half_integer(U, tol=tol)
        V_snapped, v_ok = _snap_to_half_integer(V, tol=tol)
        W_snapped, w_ok = _snap_to_half_integer(W, tol=tol)
        if u_ok and v_ok and w_ok:
            if _check_exact(U_snapped, V_snapped, W_snapped, n, m, p, tol=1e-9):
                return U_snapped, V_snapped, W_snapped
    return None

def _als_iterate_fast(U, V, W, n, m, p, max_iter=500, tol=1e-14):
    """Faster ALS with vectorized updates."""
    rank = U.shape[1]
    dim_u, dim_v, dim_w = n * m, m * p, p * n
    
    for it in range(max_iter):
        U_old = U.copy()
        V_old = V.copy()
        W_old = W.copy()
        
        # Update U: for each (i,j), solve least squares
        for idx in range(n * m):
            i, j = divmod(idx, m)
            # A[k, r] = V[j*p+k, r] * W[k*n+i, r] for k in range(p)
            A = V[j*p:(j+1)*p, :] * W[(0*n+i):(p*n+i):n, :]
            b = np.ones(p)
            try:
                sol = np.linalg.lstsq(A, b, rcond=None)[0]
                U[idx, :] = sol
            except:
                pass
        
        # Update V
        for idx in range(m * p):
            j, k = divmod(idx, p)
            # A[i, r] = U[i*m+j, r] * W[k*n+i, r] for i in range(n)
            A = U[j::m, :] * W[k*n:(k+1)*n, :]
            b = np.ones(n)
            try:
                sol = np.linalg.lstsq(A, b, rcond=None)[0]
                V[idx, :] = sol
            except:
                pass
        
        # Update W
        for idx in range(p * n):
            k, i = divmod(idx, n)
            # A[j, r] = U[i*m+j, r] * V[j*p+k, r] for j in range(m)
            A = U[i*m:(i+1)*m, :] * V[k::p, :]
            b = np.ones(m)
            try:
                sol = np.linalg.lstsq(A, b, rcond=None)[0]
                W[idx, :] = sol
            except:
                pass
        
        # Check convergence
        diff = (np.max(np.abs(U - U_old)) + 
                np.max(np.abs(V - V_old)) + 
                np.max(np.abs(W - W_old)))
        if diff < tol:
            break
    
    return U, V, W

def _structured_init_exploit_symmetry(n, m, p, rank, seed=0):
    """
    Initialization that exploits the structure of 3x4x4 matrix multiplication.
    The tensor has certain symmetries we can exploit.
    """
    rng = np.random.RandomState(seed)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Strategy: group terms by (i,k) pairs and use structured j-dependence
    # For each (i,k), we need sum_j U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = delta_{ik}
    
    # Use patterns that mix j values with signs
    patterns = [
        # Pattern 0: all positive
        lambda j: 1.0,
        # Pattern 1: alternating
        lambda j: 1.0 if j % 2 == 0 else -1.0,
        # Pattern 2: first two positive, last two negative
        lambda j: 1.0 if j < 2 else -1.0,
        # Pattern 3: first positive, rest negative
        lambda j: 1.0 if j == 0 else -1.0,
        # Pattern 4: last positive, rest negative
        lambda j: 1.0 if j == 3 else -1.0,
    ]
    
    for i in range(n):
        for k in range(p):
            # Try different patterns for each (i,k)
            for pat_idx in range(min(5, rank - col)):
                if col >= rank:
                    break
                pat = patterns[pat_idx]
                w_val = 1.0 if pat_idx % 2 == 0 else -1.0
                
                for j in range(m):
                    U[i*m+j, col] = pat(j)
                    V[j*p+k, col] = 1.0
                W[k*n+i, col] = w_val
                col += 1
    
    # Fill remaining with small random values
    while col < rank:
        U[:, col] = rng.randn(n*m) * 0.1
        V[:, col] = rng.randn(m*p) * 0.1
        W[:, col] = rng.randn(p*n) * 0.1
        col += 1
    
    return U, V, W

def _structured_init_pairwise(n, m, p, rank, seed=0):
    """
    Pairwise initialization: pair up j indices and use Strassen-like combinations.
    """
    rng = np.random.RandomState(seed)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    for i in range(n):
        for k in range(p):
            # Pair 1: j=0,1
            if col < rank:
                U[i*m+0, col] = 1.0
                U[i*m+1, col] = 1.0
                V[0*p+k, col] = 1.0
                V[1*p+k, col] = 1.0
                W[k*n+i, col] = 0.5
                col += 1
            
            # Pair 2: j=2,3
            if col < rank:
                U[i*m+2, col] = 1.0
                U[i*m+3, col] = 1.0
                V[2*p+k, col] = 1.0
                V[3*p+k, col] = 1.0
                W[k*n+i, col] = 0.5
                col += 1
            
            # Difference 1: j=0 - j=1
            if col < rank:
                U[i*m+0, col] = 1.0
                U[i*m+1, col] = -1.0
                V[0*p+k, col] = 1.0
                V[1*p+k, col] = 1.0
                W[k*n+i, col] = 0.0  # This term should vanish for correct (i,k)
                col += 1
            
            # Difference 2: j=2 - j=3
            if col < rank:
                U[i*m+2, col] = 1.0
                U[i*m+3, col] = -1.0
                V[2*p+k, col] = 1.0
                V[3*p+k, col] = 1.0
                W[k*n+i, col] = 0.0
                col += 1
    
    while col < rank:
        U[:, col] = rng.randn(n*m) * 0.2
        V[:, col] = rng.randn(m*p) * 0.2
        W[:, col] = rng.randn(p*n) * 0.2
        col += 1
    
    return U, V, W

def _structured_init_fourier(n, m, p, rank, seed=0):
    """
    Fourier-inspired initialization using roots of unity patterns.
    """
    rng = np.random.RandomState(seed)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Use 4th roots of unity for the j dimension
    omega = 1j  # i = exp(2*pi*i/4)
    roots = [1.0, 1j, -1.0, -1j]
    
    for i in range(n):
        for k in range(p):
            for r_idx in range(4):
                if col >= rank:
                    break
                # U varies with j using root r_idx
                for j in range(m):
                    U[i*m+j, col] = np.real(roots[r_idx] ** j)
                    V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0 / m
                col += 1
    
    while col < rank:
        U[:, col] = rng.randn(n*m) * 0.2
        V[:, col] = rng.randn(m*p) * 0.2
        W[:, col] = rng.randn(p*n) * 0.2
        col += 1
    
    return U, V, W

def _try_rank_als(n, m, p, rank, max_restarts=20, seed=None, deadline=None):
    """Try to find a rank-r exact decomposition using ALS."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    best_err = float('inf')
    best_U, best_V, best_W = None, None, None
    
    inits = [
        lambda s: _structured_init_exploit_symmetry(n, m, p, rank, seed=s),
        lambda s: _structured_init_pairwise(n, m, p, rank, seed=s),
        lambda s: _structured_init_fourier(n, m, p, rank, seed=s),
        lambda s: (np.random.randn(n*m, rank)*0.3, np.random.randn(m*p, rank)*0.3, np.random.randn(p*n, rank)*0.3),
    ]
    
    for restart in range(max_restarts):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        strat = restart % len(inits)
        U, V, W = inits[strat](seed + restart * 7 if seed is not None else restart * 7)
        
        # Run ALS with multiple max_iter settings
        for max_iter in [200, 400, 600]:
            if deadline is not None and time.time() > deadline - 5:
                break
            U, V, W = _als_iterate_fast(U, V, W, n, m, p, max_iter=max_iter)
            max_err = _max_error(U, V, W, n, m, p)
            
            if max_err < best_err:
                best_err = max_err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if max_err < 1e-7:
                result = _try_snap_and_check(U, V, W, n, m, p)
                if result is not None:
                    return result
    
    return None

def _try_rank_gradient(n, m, p, rank, max_restarts=10, seed=None, deadline=None):
    """Try to find a rank-r exact decomposition using gradient descent with momentum."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    dim_u, dim_v, dim_w = n*m, m*p, p*n
    
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    inits = [
        lambda s: _structured_init_exploit_symmetry(n, m, p, rank, seed=s),
        lambda s: _structured_init_pairwise(n, m, p, rank, seed=s),
        lambda s: _structured_init_fourier(n, m, p, rank, seed=s),
    ]
    
    for restart in range(max_restarts):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        strat = restart % len(inits)
        U, V, W = inits[strat](seed + restart * 17 if seed is not None else restart * 17)
        
        lr = 0.01
        momentum = 0.9
        vU = np.zeros_like(U)
        vV = np.zeros_like(V)
        vW = np.zeros_like(W)
        prev_loss = float('inf')
        
        for step in range(2000):
            T_rec = np.zeros((dim_u, dim_v, dim_w))
            for r in range(rank):
                T_rec += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            
            err = T_rec - T
            loss = np.sum(err**2)
            
            if loss < 1e-12:
                break
            if loss > 1e8:
                break
            
            dU = 2 * np.einsum('abc,br,cr->ar', err, V, W)
            dV = 2 * np.einsum('abc,ar,cr->br', err, U, W)
            dW = 2 * np.einsum('abc,ar,br->cr', err, U, V)
            
            vU = momentum * vU + (1 - momentum) * dU
            vV = momentum * vV + (1 - momentum) * dV
            vW = momentum * vW + (1 - momentum) * dW
            
            U -= lr * vU
            V -= lr * vV
            W -= lr * vW
            
            if loss < prev_loss * 0.98:
                lr *= 1.05
            elif loss > prev_loss * 1.02:
                lr *= 0.95
            prev_loss = loss
        
        max_err = _max_error(U, V, W, n, m, p)
        
        if max_err < 1e-7:
            result = _try_snap_and_check(U, V, W, n, m, p)
            if result is not None:
                return result
    
    return None

def _try_rank_l1_sparsity(n, m, p, rank, max_restarts=10, seed=None, deadline=None):
    """
    Try to find a rank-r decomposition that is sparse (many zeros).
    This uses L1 regularization as a proxy for minimizing effective rank.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    dim_u, dim_v, dim_w = n*m, m*p, p*n
    
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    inits = [
        lambda s: _structured_init_exploit_symmetry(n, m, p, rank, seed=s),
        lambda s: _structured_init_pairwise(n, m, p, rank, seed=s),
    ]
    
    for restart in range(max_restarts):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        strat = restart % len(inits)
        U, V, W = inits[strat](seed + restart * 23 if seed is not None else restart * 23)
        
        lr = 0.01
        l1_lambda = 0.01
        prev_loss = float('inf')
        
        for step in range(1500):
            T_rec = np.zeros((dim_u, dim_v, dim_w))
            for r in range(rank):
                T_rec += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            
            err = T_rec - T
            recon_loss = np.sum(err**2)
            l1_loss = l1_lambda * (np.sum(np.abs(U)) + np.sum(np.abs(V)) + np.sum(np.abs(W)))
            loss = recon_loss + l1_loss
            
            if loss < 1e-12:
                break
            if loss > 1e8:
                break
            
            dU = 2 * np.einsum('abc,br,cr->ar', err, V, W) + l1_lambda * np.sign(U)
            dV = 2 * np.einsum('abc,ar,cr->br', err, U, W) + l1_lambda * np.sign(V)
            dW = 2 * np.einsum('abc,ar,br->cr', err, U, V) + l1_lambda * np.sign(W)
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            if loss < prev_loss * 0.98:
                lr *= 1.05
            elif loss > prev_loss * 1.02:
                lr *= 0.95
            prev_loss = loss
        
        max_err = _max_error(U, V, W, n, m, p)
        
        if max_err < 1e-7:
            result = _try_snap_and_check(U, V, W, n, m, p)
            if result is not None:
                return result
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    U_sb, V_sb, W_sb = _build_schoolbook(n, m, p)
    
    best_result = None
    best_rank = schoolbook_rank
    
    # Phase 1: Try ALS with various ranks and initializations
    for rank in range(42, 33, -1):
        if time.time() > deadline - 25:
            break
        
        result = _try_rank_als(n, m, p, rank, max_restarts=12, seed=rank * 1000, deadline=deadline)
        if result is not None:
            best_result = result
            best_rank = rank
            break
    
    # Phase 2: Gradient descent with momentum
    if best_result is None:
        for rank in range(40, 33, -1):
            if time.time() > deadline - 20:
                break
            
            result = _try_rank_gradient(n, m, p, rank, max_restarts=8, seed=rank * 2000, deadline=deadline)
            if result is not None:
                best_result = result
                best_rank = rank
                break
    
    # Phase 3: L1 sparsity approach
    if best_result is None:
        for rank in range(38, 33, -1):
            if time.time() > deadline - 15:
                break
            
            result = _try_rank_l1_sparsity(n, m, p, rank, max_restarts=6, seed=rank * 3000, deadline=deadline)
            if result is not None:
                best_result = result
                best_rank = rank
                break
    
    if best_result is None:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
