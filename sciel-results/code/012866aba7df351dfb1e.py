"""Program for rank-minimal exact tensor decomposition of 3x3x5 matrix multiplication."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product

_START_TIME = time.time()
_DEADLINE = 140.0  # Safety margin before 150s kill

def _remaining():
    return _DEADLINE - (time.time() - _START_TIME)

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

def _is_valid(U, V, W, n, m, p):
    """Check if the decomposition exactly satisfies the tensor equation."""
    rank = len(U[0]) if U else 0
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
                if abs(s - 1.0) > 1e-6:
                    return False
    return True

def _error_count(U, V, W, n, m, p):
    """Count how many (i,j,k) triples have error > 1e-6."""
    rank = len(U[0]) if U else 0
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    errs = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
                if abs(s - 1.0) > 1e-6:
                    errs += 1
    return errs

def _alt_minimize(U, V, W, n, m, p, max_iters=50, tol=1e-8):
    """
    Alternating minimization: fix two matrices, solve for the third.
    For fixed U, V: W[k*n+i, r] appears in equation for (i,j,k) as:
      sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    This is linear in W. We can set up a linear system.
    
    Similarly for U and V.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for it in range(max_iters):
        if _remaining() < 5.0:
            break
        
        # Fix U, V; solve for W
        # For each (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        # Group by (k,i) index in W: w_idx = k*n+i
        # For each w_idx, we have multiple equations (one per j)
        # Let's build the system: for each (k,i), variables are W[k*n+i, 0..rank-1]
        # Equations: for each j: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        
        # Build linear system for W
        # Variables: rank * p * n = 15*rank
        # Equations: n*m*p = 45
        # This is underdetermined if rank is small, overdetermined if large.
        # Use least squares.
        
        A_w = np.zeros((n*m*p, p*n*rank))
        b_w = np.ones(n*m*p)
        eq = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    w_row = k*n + i
                    for r in range(rank):
                        A_w[eq, w_row*rank + r] = U[i*m+j, r] * V[j*p+k, r]
                    eq += 1
        
        # Solve least squares
        try:
            W_sol, _, _, _ = np.linalg.lstsq(A_w, b_w, rcond=None)
            W = W_sol.reshape(p*n, rank)
        except:
            pass
        
        # Fix U, W; solve for V
        # For each (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        # Variables: V[j*p+k, r] for j in [0,m), k in [0,p), r in [0,rank)
        A_v = np.zeros((n*m*p, m*p*rank))
        b_v = np.ones(n*m*p)
        eq = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    v_row = j*p + k
                    for r in range(rank):
                        A_v[eq, v_row*rank + r] = U[i*m+j, r] * W[k*n+i, r]
                    eq += 1
        
        try:
            V_sol, _, _, _ = np.linalg.lstsq(A_v, b_v, rcond=None)
            V = V_sol.reshape(m*p, rank)
        except:
            pass
        
        # Fix V, W; solve for U
        # For each (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        A_u = np.zeros((n*m*p, n*m*rank))
        b_u = np.ones(n*m*p)
        eq = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = i*m + j
                    for r in range(rank):
                        A_u[eq, u_row*rank + r] = V[j*p+k, r] * W[k*n+i, r]
                    eq += 1
        
        try:
            U_sol, _, _, _ = np.linalg.lstsq(A_u, b_u, rcond=None)
            U = U_sol.reshape(n*m, rank)
        except:
            pass
        
        # Check convergence
        err = _error_count(U.tolist(), V.tolist(), W.tolist(), n, m, p)
        if err == 0:
            break
    
    return U, V, W

def _try_rank(n, m, p, rank, num_trials=100, seed=42):
    """Try to find a valid decomposition of given rank."""
    rng = np.random.RandomState(seed)
    
    for trial in range(num_trials):
        if _remaining() < 10.0:
            break
        
        # Structured initialization: use block structure
        # Initialize with small random perturbations of a structured pattern
        U = rng.randn(n*m, rank) * 0.1
        V = rng.randn(m*p, rank) * 0.1
        W = rng.randn(p*n, rank) * 0.1
        
        # Run alternating minimization
        U, V, W = _alt_minimize(U, V, W, n, m, p, max_iters=30)
        
        # Check if valid
        if _is_valid(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return U.tolist(), V.tolist(), W.tolist()
        
        # Try with different initialization: sparse-like
        U = rng.rand(n*m, rank) * 2 - 1
        V = rng.rand(m*p, rank) * 2 - 1
        W = rng.rand(p*n, rank) * 2 - 1
        U, V, W = _alt_minimize(U, V, W, n, m, p, max_iters=30)
        
        if _is_valid(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def _try_strassen_based(n, m, p):
    """
    Try to construct a decomposition using Strassen-like ideas.
    For 3x3x5, we can think of it as combining 3x3x3 and 3x3x2 blocks.
    """
    # This is a placeholder for a more sophisticated approach
    # For now, return None to indicate no improvement found
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as guaranteed baseline
    best = _schoolbook(n, m, p)
    best_rank = n * m * p  # 45
    
    start_time = time.time()
    
    # Try to find lower-rank decompositions
    # Lower bound: max(n*m, m*p, p*n) = max(9, 15, 15) = 15
    # But tensor rank lower bounds are more subtle.
    # Known: rank(2,2,2)=7, rank(3,3,3)<=23
    # For 3,3,5: try ranks from 20 upward
    
    # Try structured ranks
    candidate_ranks = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 
                       31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]
    
    for rank in candidate_ranks:
        if _remaining() < 15.0:
            break
        
        result = _try_rank(n, m, p, rank, num_trials=50, seed=rank)
        if result is not None:
            U, V, W = result
            best = {"U": U, "V": V, "W": W}
            best_rank = rank
            # Found a valid decomposition; try even lower ranks
            continue
    
    # Also try the Strassen-based approach
    if _remaining() > 20.0:
        result = _try_strassen_based(n, m, p)
        if result is not None:
            U, V, W = result
            if len(U[0]) < best_rank:
                best = {"U": U, "V": V, "W": W}
                best_rank = len(U[0])
    
    # Return the best found
    return best

# EVOLVE-BLOCK-END
