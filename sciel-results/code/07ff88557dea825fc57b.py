"""
Rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication.

Strategy:
- Start from the schoolbook rank-36 decomposition.
- Use a structured block approach: the 2x3x6 tensor can be viewed as
  two 2x3x3 blocks (since p=6=2*3).
- Search for low-rank decompositions using:
  1. Known results for small matrix multiplication tensors.
  2. Integer/half-integer constrained optimization via structured patterns.
  3. A final verification pass.

Known bounds: For 2x3 * 3x6, the tensor is T_{2,3,6}.
The schoolbook rank is 36.

Key insight: T_{2,3,6} can be related to T_{2,3,3}. If we can find a rank-r
decomposition for 2x3x3, we might extend it. But 2x3x6 is not simply two copies
of 2x3x3 in a useful way for rank reduction via simple tensor product.

However, we can try:
1. Direct search for rank < 36 of T_{2,3,6} using structured initialization.
2. Search for rank < 18 of T_{2,3,3} and extend via a valid construction.
3. Use scipy.optimize.least_squares as a refinement tool.

The decomposition condition: for all (i,j,k) in [2]x[3]x[6]:
    sum_r U[i*3+j, r] * V[j*6+k, r] * W[k*2+i, r] = 1

And all entries must be integers or half-integers.
"""

import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the decomposition exactly reproduces the matrix multiplication tensor.
    For all (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _snap_to_half_integers(mat, tol=1e-6):
    """Snap values to nearest half-integer if close."""
    mat = np.array(mat, dtype=float)
    snapped = np.round(mat * 2) / 2.0
    if np.max(np.abs(mat - snapped)) < tol:
        return snapped
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
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

def _residuals_233(params, n=2, m=3, p=3, r=None):
    """Compute residuals for T_{2,3,3} decomposition."""
    if r is None:
        r = int((len(params) / (n*m + m*p + p*n)))
    nm = n * m
    mp = m * p
    pn = p * n
    
    U = params[:nm*r].reshape(nm, r)
    V = params[nm*r:nm*r+mp*r].reshape(mp, r)
    W = params[nm*r+mp*r:].reshape(pn, r)
    
    res = np.zeros(n*m*p)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                res[idx] = s - 1.0
                idx += 1
    return res

def _residuals_236(params, n=2, m=3, p=6, r=None):
    """Compute residuals for T_{2,3,6} decomposition."""
    if r is None:
        r = int((len(params) / (n*m + m*p + p*n)))
    nm = n * m
    mp = m * p
    pn = p * n
    
    U = params[:nm*r].reshape(nm, r)
    V = params[nm*r:nm*r+mp*r].reshape(mp, r)
    W = params[nm*r+mp*r:].reshape(pn, r)
    
    res = np.zeros(n*m*p)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                res[idx] = s - 1.0
                idx += 1
    return res

def _unpack_233(params, r):
    n, m, p = 2, 3, 3
    U = params[:6*r].reshape(6, r)
    V = params[6*r:6*r+9*r].reshape(9, r)
    W = params[6*r+9*r:].reshape(6, r)
    return U, V, W

def _unpack_236(params, r):
    n, m, p = 2, 3, 6
    U = params[:6*r].reshape(6, r)
    V = params[6*r:6*r+18*r].reshape(18, r)
    W = params[6*r+18*r:].reshape(12, r)
    return U, V, W

def _expand_233_to_236(U_sub, V_sub, W_sub):
    """
    Expand a 2x3x3 decomposition to a 2x3x6 decomposition.
    This uses the block structure: p=6 is split into two blocks of 3.
    The resulting rank is 2 * r_sub.
    """
    r_sub = U_sub.shape[1]
    r_big = 2 * r_sub
    
    U_big = np.zeros((6, r_big))
    V_big = np.zeros((18, r_big))
    W_big = np.zeros((12, r_big))
    
    # First block (k=0,1,2)
    U_big[:, :r_sub] = U_sub
    for j in range(3):
        for k in range(3):
            V_big[j * 6 + k, :r_sub] = V_sub[j * 3 + k]
    for k in range(3):
        for i in range(2):
            W_big[k * 2 + i, :r_sub] = W_sub[k * 2 + i]
    
    # Second block (k=3,4,5)
    U_big[:, r_sub:] = U_sub
    for j in range(3):
        for k in range(3):
            V_big[j * 6 + (k + 3), r_sub:] = V_sub[j * 3 + k]
    for k in range(3):
        for i in range(2):
            W_big[(k + 3) * 2 + i, r_sub:] = W_sub[k * 2 + i]
    
    return U_big, V_big, W_big

def _try_233_rank(r, deadline, max_trials=20):
    """
    Try to find a rank-r decomposition of T_{2,3,3} using scipy.optimize.least_squares.
    Returns (U, V, W) if found with half-integer entries, else None.
    """
    from scipy.optimize import least_squares
    
    n, m, p = 2, 3, 3
    nm, mp, pn = n*m, m*p, p*n
    total_params = (nm + mp + pn) * r
    
    for trial in range(max_trials):
        if time.time() > deadline:
            return None
        
        # Different random seeds for each trial
        rng = np.random.RandomState(42 + trial)
        
        # Initialize: mix of random and structured
        if trial < 5:
            # Small random init
            x0 = rng.randn(total_params) * 0.3
        elif trial < 10:
            # Initialize near schoolbook-like structure (sparse)
            x0 = rng.randn(total_params) * 0.1
            # Add some 1s at structured positions
            for c in range(min(r, nm)):
                x0[c * r + c] = 1.0
        else:
            # Half-integer random init
            x0 = (rng.randint(-4, 5, size=total_params) * 0.5)
        
        try:
            result = least_squares(
                _residuals_233, x0, args=(n, m, p, r),
                method='trf', max_nfev=5000, ftol=1e-14, xtol=1e-14, gtol=1e-14
            )
        except Exception:
            continue
        
        if time.time() > deadline:
            return None
        
        if result.cost < 1e-12:
            U, V, W = _unpack_233(result.x, r)
            
            # Try snapping to half-integers
            U_snap = _snap_to_half_integers(U, tol=1e-8)
            V_snap = _snap_to_half_integers(V, tol=1e-8)
            W_snap = _snap_to_half_integers(W, tol=1e-8)
            
            if U_snap is not None and V_snap is not None and W_snap is not None:
                if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p):
                    return U_snap, V_snap, W_snap
            
            # Try integer snapping
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            if _verify_decomposition(U_int, V_int, W_int, n, m, p):
                return U_int, V_int, W_int
            
            # If the solution is exact but not snapped, check if it's already half-integer
            if _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
                # Check if entries are close to half-integers
                if (np.all(np.abs(U * 2 - np.round(U * 2)) < 1e-8) and
                    np.all(np.abs(V * 2 - np.round(V * 2)) < 1e-8) and
                    np.all(np.abs(W * 2 - np.round(W * 2)) < 1e-8)):
                    return np.round(U*2)/2, np.round(V*2)/2, np.round(W*2)/2
    
    return None

def _try_236_rank_direct(r, deadline, max_trials=10):
    """
    Try to find a rank-r decomposition of T_{2,3,6} directly using scipy.optimize.least_squares.
    """
    from scipy.optimize import least_squares
    
    n, m, p = 2, 3, 6
    nm, mp, pn = n*m, m*p, p*n
    total_params = (nm + mp + pn) * r
    
    for trial in range(max_trials):
        if time.time() > deadline:
            return None
        
        rng = np.random.RandomState(100 + trial)
        
        if trial < 3:
            x0 = rng.randn(total_params) * 0.2
        elif trial < 6:
            # Structured init: replicate schoolbook pattern sparsely
            x0 = rng.randn(total_params) * 0.1
            # Add identity-like structure in U
            for c in range(min(r, nm)):
                x0[c * r + c] = 1.0
        else:
            x0 = (rng.randint(-4, 5, size=total_params) * 0.5)
        
        try:
            result = least_squares(
                _residuals_236, x0, args=(n, m, p, r),
                method='trf', max_nfev=3000, ftol=1e-14, xtol=1e-14, gtol=1e-14
            )
        except Exception:
            continue
        
        if time.time() > deadline:
            return None
        
        if result.cost < 1e-12:
            U, V, W = _unpack_236(result.x, r)
            
            U_snap = _snap_to_half_integers(U, tol=1e-8)
            V_snap = _snap_to_half_integers(V, tol=1e-8)
            W_snap = _snap_to_half_integers(W, tol=1e-8)
            
            if U_snap is not None and V_snap is not None and W_snap is not None:
                if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p):
                    return U_snap, V_snap, W_snap
            
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            if _verify_decomposition(U_int, V_int, W_int, n, m, p):
                return U_int, V_int, W_int
    
    return None

def construct():
    """
    Construct the rank-minimal exact tensor decomposition for 2x3 * 3x6.
    Returns {'U': U, 'V': V, 'W': W} with integer/half-integer entries.
    """
    start_time = time.time()
    deadline = start_time + 135.0  # Leave 15s margin for 150s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # Strategy 1: Try to find rank < 36 for T_{2,3,6} directly
    # Start with ranks that might be achievable
    # The schoolbook is 36. Try 35, 34, ... down to some minimum.
    # But first, try the 2x3x3 approach which might give better structure.
    
    # Strategy 2: Find rank < 18 for T_{2,3,3}, then expand to T_{2,3,6}
    # If rank(T_{2,3,3}) = r, then rank(T_{2,3,6}) <= 2r
    # So if we find r=17 for 2x3x3, we get rank 34 for 2x3x6.
    # If we find r=16, we get rank 32.
    
    best_result = None
    best_rank = 36
    
    # First, try the 2x3x3 sub-problem for various ranks
    # This is more tractable than direct 2x3x6 search
    for r_sub in [17, 16, 15]:
        if time.time() > deadline - 30:
            break
        
        result = _try_233_rank(r_sub, deadline)
        if result is not None:
            U_sub, V_sub, W_sub = result
            U_big, V_big, W_big = _expand_233_to_236(U_sub, V_sub, W_sub)
            
            if _verify_decomposition(U_big, V_big, W_big, n, m, p):
                rank_big = U_big.shape[1]
                if rank_big < best_rank:
                    best_rank = rank_big
                    best_result = (U_big, V_big, W_big)
                break  # Found the best from this family, stop
    
    # Strategy 3: Direct search for T_{2,3,6} with rank < best_rank
    if best_rank == 36:
        # No improvement from 2x3x3, try direct search
        for r_direct in [35, 34, 33]:
            if time.time() > deadline - 20:
                break
            
            result = _try_236_rank_direct(r_direct, deadline)
            if result is not None:
                U_d, V_d, W_d = result
                if _verify_decomposition(U_d, V_d, W_d, n, m, p):
                    if r_direct < best_rank:
                        best_rank = r_direct
                        best_result = (U_d, V_d, W_d)
                    break
    
    # Return best found or schoolbook
    if best_result is not None:
        U, V, W = best_result
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Final snap to ensure clean half-integers
        U_snap = _snap_to_half_integers(U, tol=1e-8)
        V_snap = _snap_to_half_integers(V, tol=1e-8)
        W_snap = _snap_to_half_integers(W, tol=1e-8)
        
        if U_snap is not None and V_snap is not None and W_snap is not None:
            if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p):
                U, V, W = U_snap, V_snap, W_snap
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook (rank 36)
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
