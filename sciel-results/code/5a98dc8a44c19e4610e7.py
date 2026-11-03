import numpy as np
import time
import random

DIMENSIONS = [4, 4, 5]
_start_time = None

def _time_left(deadline):
    return deadline - time.time()

def _target_tensor(n, m, p):
    """Return the 4x4x5 matrix multiplication tensor as a flat array."""
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1
    return T

def _is_valid(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W exactly decompose the multiplication tensor."""
    rank = len(U[0]) if U else 0
    T = _target_tensor(n, m, p)
    # Reconstruct tensor
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] -= U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
    return np.allclose(T, 0, atol=tol)

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_rank_r(n, m, p, r, deadline, max_iters=100000):
    """
    Try to find a rank-r decomposition using a greedy/search approach.
    We search over integer/half-integer coefficients in a small range.
    """
    T = _target_tensor(n, m, p)
    
    # We need to find U (n*m x r), V (m*p x r), W (p*n x r)
    # such that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # This is a non-linear system. Let's try a different approach:
    # Fix U and V, then solve for W linearly.
    # Or use a randomized approach.
    
    # Strategy: Try random sparse U, V with entries in {-1, 0, 1, 0.5, -0.5}
    # and solve for W.
    
    best = None
    for it in range(max_iters):
        if time.time() > deadline - 5:
            break
        
        # Generate random sparse U, V
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        
        # Make them sparse: each column has a few non-zero entries
        for col in range(r):
            # Choose a few random positions for U
            num_nonzero_u = random.randint(2, min(6, n * m))
            idx_u = random.sample(range(n * m), num_nonzero_u)
            for idx in idx_u:
                U[idx, col] = random.choice([-1, 0.5, 1, 1.5, 2])
            
            num_nonzero_v = random.randint(2, min(6, m * p))
            idx_v = random.sample(range(m * p), num_nonzero_v)
            for idx in idx_v:
                V[idx, col] = random.choice([-1, 0.5, 1, 1.5, 2])
        
        # For each (i,k), we need: sum_j T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This is linear in W for fixed U, V.
        # For each k, i: W[k*n+i] appears in terms where V[j*p+k] is non-zero.
        
        # Let's set up a linear system for W.
        # For each (i, j, k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # This is not linear in W because W appears multiplied by U and V per term.
        
        # Actually, it IS linear in W if we fix U and V:
        # For each (i, k): sum_r [sum_j U[i*m+j, r] * V[j*p+k, r]] * W[k*n+i, r] = 1
        # Wait no, that's not right either. Let me re-examine.
        
        # The constraint is: for all i, j, k:
        #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # 
        # This is linear in W[k*n+i, r] for each r, but coupled across r.
        # Actually, it's a system of n*m*p equations in p*n*r unknowns (W entries).
        # Each equation involves r terms (one per r).
        
        # Let's write it as: for each (i,j,k):
        #   sum_r (U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r] = 1
        #
        # This is linear in the W variables! We have n*m*p equations and p*n*r unknowns.
        # For n=4, m=4, p=5: 80 equations and 5*4*r = 20r unknowns.
        # For r >= 4, we have more unknowns than equations (20r >= 80 when r >= 4).
        
        # Let's set up the linear system.
        # Variables: W[k*n+i, r] for k in 0..p-1, i in 0..n-1, r in 0..r-1
        # Total: p * n * r variables
        
        num_vars = p * n * r
        num_eqs = n * m * p
        
        # Build the system A * w = b
        # where w is the vector of W entries
        
        A = np.zeros((num_eqs, num_vars))
        b = np.ones(num_eqs)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i * m * p + j * p + k
                    for r_idx in range(r):
                        var_idx = (k * n + i) * r + r_idx
                        A[eq_idx, var_idx] = U[i * m + j, r_idx] * V[j * p + k, r_idx]
        
        # Solve A * w = b
        try:
            w, residuals, rank_sys, sv = np.linalg.lstsq(A, b, rcond=None)
            # Check if it's an exact solution
            if np.allclose(A @ w, b, atol=1e-9):
                # Check if w entries are close to integer or half-integer
                w_rounded = np.round(w * 2) / 2
                if np.allclose(A @ w_rounded, b, atol=1e-9):
                    # Success! Convert to list format
                    W_list = [[0] * r for _ in range(p * n)]
                    for idx in range(p * n):
                        for r_idx in range(r):
                            W_list[idx][r_idx] = w_rounded[idx * r + r_idx]
                    
                    # Convert U, V to list format
                    U_list = [[0] * r for _ in range(n * m)]
                    V_list = [[0] * r for _ in range(m * p)]
                    for idx in range(n * m):
                        for r_idx in range(r):
                            U_list[idx][r_idx] = U[idx, r_idx]
                    for idx in range(m * p):
                        for r_idx in range(r):
                            V_list[idx][r_idx] = V[idx, r_idx]
                    
                    # Final validation
                    if _is_valid(U_list, V_list, W_list, n, m, p):
                        return {"U": U_list, "V": V_list, "W": W_list}
        except Exception:
            pass
    
    return None

def construct():
    global _start_time
    _start_time = time.time()
    n, m, p = DIMENSIONS
    deadline = _start_time + 140  # Safety margin
    
    # Try ranks from low to high
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(16, 20, 20) = 20
    # Known upper bound: 80 (schoolbook)
    # Expected: ~23 based on 4x4x4 (rank 19) + 4 for 5th column
    
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try ranks from 20 upward
    for r in range(20, min(best_rank, 40)):
        if time.time() > deadline - 10:
            break
        
        result = _try_rank_r(n, m, p, r, deadline, max_iters=50000)
        if result is not None:
            best_result = result
            best_rank = r
            break  # Found the minimum rank
    
    return best_result
