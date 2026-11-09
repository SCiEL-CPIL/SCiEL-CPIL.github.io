"""
Improved algorithm for 3x3x8 matrix multiplication tensor decomposition.

Strategy:
- The current seed uses schoolbook rank-72 (27+27+18).
- We can improve by using a sub-schoolbook decomposition for the 3x3x3 blocks.
- A known rank-24 decomposition for 3x3x3 exists (one less than schoolbook 27).
- Even better: rank-23 is the known minimum for 3x3x3.
- For 3x3x2, schoolbook is rank 18, which is likely optimal.

Approach:
1. Implement a verified rank-24 decomposition for 3x3x3 using a known construction.
2. If that's not sufficient, try rank-23.
3. Combine: 2*24 + 18 = 66, or 2*23 + 18 = 64.

Let me implement a concrete rank-24 decomposition for 3x3x3.

For 3x3x3, one approach is to use the fact that we can "merge" two of the schoolbook
terms. The schoolbook has 27 terms, one for each (i,j,k) triple. We can sometimes
combine terms that share structure.

Actually, a well-known rank-24 decomposition uses the following idea:
- 24 terms with entries from {-2, -1, 0, 1, 2}

Let me construct a valid rank-24 decomposition. I'll use a systematic approach:
Start from schoolbook and try to merge terms.

Actually, let me use a different well-known approach. For 3x3x3, there's a
decomposition based on the 3-point FFT with rational coefficients that achieves
rank 24.

Here's a concrete construction I'll verify:

The key insight for reducing rank: if two terms have the same U and V entries
but different W, or some other structural similarity, they might be mergeable.

Let me try a different approach: use a local search starting from schoolbook
to find a rank-24 or rank-25 decomposition.

Given the time constraints, let me implement:
1. A verified rank-25 decomposition (easier to find) for 3x3x3.
2. Total: 2*25 + 18 = 68.

Actually, let me try to find a rank-24 decomposition by solving a system of equations.

For rank r, we need to find U (9 x r), V (9 x r), W (9 x r) such that:
sum_t U[i*3+j, t] * V[j*3+k, t] * W[k*3+i, t] = 1 for all valid (i,j,k)
and = 0 for all invalid combinations.

For 3x3x3: 9*9*9 = 729 equations, but only 27 are "active" (one for each (i,j,k)).
The other 702 must be zero.

This is a system of 729 polynomial equations in 9*r*3 = 27r variables.

For r=24: 648 unknowns, 729 equations. Overdetermined but might have solutions.

Let me use a numerical approach: random initialization + gradient descent / Newton's method.

Given the 150s budget, I'll allocate time wisely.

Let me implement a local search for the 3x3x3 tensor:
- Start with a random sparse initialization
- Use scipy.optimize.minimize to minimize the sum of squared residuals
- If we find a valid decomposition, great. If not, fall back to schoolbook.

Actually, for reliability, let me use a known explicit decomposition.

Here's a rank-24 decomposition for 3x3x3 that I'll construct and verify:

I'll use the following approach based on "tensor rank reduction":
Start with schoolbook (27 terms) and try to eliminate 3 terms by adjusting others.

Let me just implement a robust numerical search with a time budget.
"""

import time
import numpy as np
import sys

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that U, V, W form a valid tensor decomposition for n x m times m x p.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    if U.ndim != 2 or V.ndim != 2 or W.ndim != 2:
        return False
        
    r = U.shape[1]
    if V.shape[1] != r or W.shape[1] != r:
        return False
        
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False

    # Create target tensor
    target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0

    # Compute decomposition tensor
    T = np.einsum('at,bt,ct->abc', U, V, W)
    
    if not np.allclose(T, target, atol=tol, rtol=0):
        return False
        
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition for n x m times m x p."""
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

def _local_search_3x3x3(max_time=60.0, target_rank=24, seed=42):
    """
    Try to find a rank-target_rank decomposition for 3x3x3 using local search.
    Returns (U, V, W) if found, else None.
    """
    n, m, p = 3, 3, 3
    r = target_rank
    
    # Create target tensor
    target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m + j, j * p + k, k * n + i] = 1.0
    
    rng = np.random.RandomState(seed)
    
    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    deadline = time.time() + max_time
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Random sparse initialization
        U = rng.randn(n * m, r) * 0.5
        V = rng.randn(m * p, r) * 0.5
        W = rng.randn(p * n, r) * 0.5
        
        # Make it sparse: zero out many entries
        for _ in range(3):
            mask_u = rng.rand(n * m, r) < 0.3
            mask_v = rng.rand(m * p, r) < 0.3
            mask_w = rng.rand(p * n, r) < 0.3
            U[mask_u] = 0
            V[mask_v] = 0
            W[mask_w] = 0
        
        # Use scipy to minimize
        try:
            from scipy.optimize import minimize
            
            def flatten(U, V, W):
                return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def unflatten(x):
                U = x[:n*m*r].reshape(n*m, r)
                V = x[n*m*r:2*n*m*r].reshape(m*p, r)
                W = x[2*n*m*r:3*n*m*r].reshape(p*n, r)
                return U, V, W
            
            def objective(x):
                U, V, W = unflatten(x)
                T = np.einsum('at,bt,ct->abc', U, V, W)
                return np.sum((T - target) ** 2)
            
            x0 = flatten(U, V, W)
            
            res = minimize(objective, x0, method='L-BFGS-B', 
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12})
            
            U, V, W = unflatten(res.x)
            err = objective(res.x)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_err < 1e-9:
                return best_U, best_V, best_W
                
        except Exception:
            continue
    
    if best_err < 1e-9 and best_U is not None:
        return best_U, best_V, best_W
    
    return None

def _combine_decompositions(decomp_list, n=3, m=3, total_p=8):
    """
    Combine multiple decompositions for different p-blocks into one for the full problem.
    Each decomp is (U, V, W) as numpy arrays for 3x3x3 or 3x3x2.
    Returns (U_full, V_full, W_full) for 3x3x8 as lists.
    """
    total_rank = 0
    for U, V, W in decomp_list:
        total_rank += U.shape[1]
    
    U_full = np.zeros((n * m, total_rank))
    V_full = np.zeros((m * total_p, total_rank))
    W_full = np.zeros((total_p * n, total_rank))
    
    offset = 0
    p_offset = 0
    for U, V, W in decomp_list:
        p = V.shape[0] // m
        r = U.shape[1]
        
        U_full[:, offset:offset+r] = U
        
        for j in range(m):
            for k_local in range(p):
                row_in_block = j * p + k_local
                row_in_full = j * total_p + (p_offset + k_local)
                V_full[row_in_full, offset:offset+r] = V[row_in_block, :]
        
        for k_local in range(p):
            for i in range(n):
                row_in_block = k_local * n + i
                row_in_full = (p_offset + k_local) * n + i
                W_full[row_in_full, offset:offset+r] = W[row_in_block, :]
        
        offset += r
        p_offset += p
    
    return U_full.tolist(), V_full.tolist(), W_full.tolist()

def construct():
    """
    Construct the decomposition for 3x3x8 matrix multiplication.
    Returns dict with keys 'U', 'V', 'W'.
    """
    n, m, p = 3, 3, 8
    start_time = time.time()
    
    # Start with schoolbook as fallback
    U333_s, V333_s, W333_s = _schoolbook(3, 3, 3)
    U332_s, V332_s, W332_s = _schoolbook(3, 3, 2)
    
    # Try to find better decompositions for 3x3x3
    # Allocate time carefully
    time_for_search = 60.0
    search_deadline = start_time + time_for_search
    
    # Try rank 24 first (most impactful)
    U333_b, V333_b, W333_b = None, None, None
    if time.time() < search_deadline:
        result = _local_search_3x3x3(max_time=30.0, target_rank=24, seed=42)
        if result is not None:
            U333_b, V333_b, W333_b = result
            # Verify
            if _verify_decomposition(U333_b, V333_b, W333_b, 3, 3, 3):
                pass
            else:
                U333_b = None
    
    # If rank 24 didn't work, try rank 25
    if U333_b is None and time.time() < search_deadline:
        result = _local_search_3x3x3(max_time=20.0, target_rank=25, seed=123)
        if result is not None:
            U333_b, V333_b, W333_b = result
            if _verify_decomposition(U333_b, V333_b, W333_b, 3, 3, 3):
                pass
            else:
                U333_b = None
    
    # Decide which 3x3x3 decomposition to use
    if U333_b is not None:
        U333, V333, W333 = U333_b, V333_b, W333_b
    else:
        U333, V333, W333 = U333_s, V333_s, W333_s
    
    # Combine
    U_full, V_full, W_full = _combine_decompositions([
        (U333, V333, W333),
        (U333, V333, W333),
        (U332_s, V332_s, W332_s)
    ])
    
    # Verify the full decomposition
    if not _verify_decomposition(U_full, V_full, W_full, 3, 3, 8):
        # Fallback to pure schoolbook
        U_full, V_full, W_full = _combine_decompositions([
            (U333_s, V333_s, W333_s),
            (U333_s, V333_s, W333_s),
            (U332_s, V332_s, W332_s)
        ])
    
    return {"U": U_full, "V": V_full, "W": W_full}

# EVOLVE-BLOCK-END
