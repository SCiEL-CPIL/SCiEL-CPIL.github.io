"""Matrix multiplication 2,3,3 - rank-minimal tensor decomposition."""
import numpy as np
from fractions import Fraction
import time
import scipy.optimize as opt

DIMENSIONS = [2, 3, 3]

def _to_rational_approx_fast(matrix, max_den=1000):
    """Convert a float matrix to a rational matrix using limit_denominator."""
    result = []
    for row in matrix:
        r = []
        for val in row:
            # Handle zero explicitly to avoid Fraction(0.0) issues or slow paths
            if abs(val) < 1e-12:
                r.append(Fraction(0))
            else:
                # limit_denominator is much faster than manual search
                frac = Fraction(val).limit_denominator(max_den)
                r.append(frac)
        result.append(r)
    return result

def _verify_decomposition_exact(U, V, W, n, m, p):
    """Verify with exact rational arithmetic for ALL entries."""
    rank = len(U[0])
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Precompute target values for all (u, v, w)
    # u in [0, nm), v in [0, mp), w in [0, pn)
    # Target is 1 if there exist i,j,k such that:
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    # Else 0.
    
    targets = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                targets[(u, v, w)] = 1

    for u in range(nm):
        for v in range(mp):
            for w in range(pn):
                target = targets.get((u, v, w), 0)
                s = Fraction(0)
                for c in range(rank):
                    s += U[u][c] * V[v][c] * W[w][c]
                if s != target:
                    return False
    return True

def _tensor_residual(x, n, m, p, rank):
    """
    Calculate the residual for the tensor decomposition.
    x is a flattened array of length 21*rank (U, V, W concatenated).
    """
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Unpack
    U = x[:nm*rank].reshape((nm, rank))
    V = x[nm*rank:nm*rank+mp*rank].reshape((mp, rank))
    W = x[nm*rank+mp*rank:].reshape((pn, rank))
    
    # Compute sum_{c} U[:,c] outer V[:,c] outer W[:,c]
    # This is equivalent to np.einsum('uc,vc,wc->uvw', U, V, W)
    tensor = np.einsum('uc,vc,wc->uvw', U, V, W)
    
    # Target tensor
    # T[u,v,w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i
    # We can build the target tensor once and pass it, but for simplicity in closure, 
    # we can reconstruct or store it. 
    # To be efficient, let's assume we have a global target or compute it.
    # Computing it every time is expensive. Let's make it a global.
    
    global _TARGET_TENSOR
    if _TARGET_TENSOR is None:
        _TARGET_TENSOR = np.zeros((nm, mp, pn))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    _TARGET_TENSOR[u, v, w] = 1.0
    
    diff = tensor - _TARGET_TENSOR
    # Return flattened difference for least_squares
    return diff.flatten()

_TARGET_TENSOR = None

def _try_rank(rank, n, m, p, max_attempts=20, time_limit=None):
    """Try to find a rank-r decomposition using local optimization."""
    rng = np.random.default_rng(42 + rank)
    nm = n * m
    mp = m * p
    pn = p * n
    num_vars = (nm + mp + pn) * rank
    
    # Initialize global target if not done
    global _TARGET_TENSOR
    if _TARGET_TENSOR is None:
        _TARGET_TENSOR = np.zeros((nm, mp, pn))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    _TARGET_TENSOR[u, v, w] = 1.0

    for attempt in range(max_attempts):
        if time_limit is not None and time.time() > time_limit:
            break
        
        # Random initialization
        # Scale small to avoid initial large gradients
        x0 = rng.random(num_vars) * 0.1 - 0.05
        
        try:
            # Use least_squares to minimize norm of residual
            # Method 'lm' is Levenberg-Marquardt, good for small problems
            # But requires len(residual) >= len(params). 
            # Residual size: nm*mp*pn = 324.
            # Param size: 21*rank. For rank=18, 378. 
            # If rank is high, params > residuals, LM might fail or need 'trf'/'dogbox'.
            # 'trf' (Trust Region Reflective) handles overdetermined/underdetermined.
            res = opt.least_squares(_tensor_residual, x0, args=(n, m, p, rank), 
                                    method='trf', max_nfev=1000)
            
            if res.success or res.cost < 1e-10:
                # Check if the solution is valid
                # Reconstruct U, V, W
                U = res.x[:nm*rank].reshape((nm, rank))
                V = res.x[nm*rank:nm*rank+mp*rank].reshape((mp, rank))
                W = res.x[nm*rank+mp*rank:].reshape((pn, rank))
                
                # Try to rationalize
                found_rational = False
                for max_den in [10, 50, 100, 500, 1000, 5000]:
                    U_r = _to_rational_approx_fast(U, max_den)
                    V_r = _to_rational_approx_fast(V, max_den)
                    W_r = _to_rational_approx_fast(W, max_den)
                    
                    # Verify exact
                    if _verify_decomposition_exact(U_r, V_r, W_r, n, m, p):
                        return U_r, V_r, W_r
                        
        except Exception:
            continue
            
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Try ranks from 17 down to 9
    # Schoolbook is 18.
    # If we find 17, score is 18/17 = 1.058
    # If we find 16, score is 18/16 = 1.125
    
    for rank in range(17, 8, -1):
        if time.time() > deadline:
            break
            
        result = _try_rank(rank, n, m, p, max_attempts=5, time_limit=deadline)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
            
    # Fallback to schoolbook
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
