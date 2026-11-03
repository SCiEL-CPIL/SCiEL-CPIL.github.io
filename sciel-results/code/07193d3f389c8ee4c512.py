"""Structural change: use integer/half-integer structured search with exact verification.
For 3x5x5 matrix multiplication, we search for low-rank decompositions using
structured patterns and local refinement, with exact rational verification."""

DIMENSIONS = [3, 5, 5]

import numpy as np
import time
import random
import sys
from itertools import product

# EVOLVE-BLOCK-START

def _verify_exact(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition is exact: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(len(U[0])):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_exact_fast(U, V, W, n, m, p, tol=1e-10):
    """Vectorized verification."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m + j], V[j*p + k] * W[k*n + i])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
    return {"U": U, "V": V, "W": W}

def _compute_error(U, V, W, n, m, p):
    """Compute sum of squared errors."""
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(len(U[0])):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                err += (s - 1.0) ** 2
    return err

def _compute_error_fast(U, V, W, n, m, p):
    """Vectorized error computation."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m + j], V[j*p + k] * W[k*n + i])
                err += (s - 1.0) ** 2
    return err

def _random_half_integer_matrix(rows, cols, scale=3):
    """Generate a random matrix with half-integer entries in [-scale, scale]."""
    M = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            val = random.choice([-1, 1]) * random.randint(0, scale)
            if random.random() < 0.3:
                val += 0.5 * random.choice([-1, 1])
            M[r, c] = val
    return M

def _try_local_refinement(U, V, W, n, m, p, time_limit, max_iters=5000):
    """Try to refine a near-exact decomposition to exact half-integer values."""
    start_time = time.time()
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    best_err = _compute_error_fast(U, V, W, n, m, p)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if time.time() > start_time + time_limit:
            break
        
        err = _compute_error_fast(U, V, W, n, m, p)
        if err < 1e-12:
            # Snap to half-integers
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            if _verify_exact_fast(U_snap, V_snap, W_snap, n, m, p):
                return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
            # Try integer snap
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            if _verify_exact_fast(U_int, V_int, W_int, n, m, p):
                return U_int.tolist(), V_int.tolist(), W_int.tolist()
            return U.tolist(), V.tolist(), W.tolist()
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Random perturbation of one entry
        mat = random.choice(['U', 'V', 'W'])
        if mat == 'U':
            r, c = random.randint(0, U.shape[0]-1), random.randint(0, U.shape[1]-1)
            U[r, c] += random.gauss(0, 0.1)
        elif mat == 'V':
            r, c = random.randint(0, V.shape[0]-1), random.randint(0, V.shape[1]-1)
            V[r, c] += random.gauss(0, 0.1)
        else:
            r, c = random.randint(0, W.shape[0]-1), random.randint(0, W.shape[1]-1)
            W[r, c] += random.gauss(0, 0.1)
    
    # Final check on best
    if best_err < 1e-10:
        U_snap = np.round(best_U * 2) / 2
        V_snap = np.round(best_V * 2) / 2
        W_snap = np.round(best_W * 2) / 2
        if _verify_exact_fast(U_snap, V_snap, W_snap, n, m, p):
            return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
        U_int = np.round(best_U)
        V_int = np.round(best_V)
        W_int = np.round(best_W)
        if _verify_exact_fast(U_int, V_int, W_int, n, m, p):
            return U_int.tolist(), V_int.tolist(), W_int.tolist()
    
    return None

def _structured_search_rank(n, m, p, target_rank, time_limit):
    """Search for a decomposition of given rank using structured patterns."""
    start_time = time.time()
    
    # Try various structured initializations
    for attempt in range(20):
        if time.time() > start_time + time_limit:
            break
        
        # Random half-integer initialization
        U = _random_half_integer_matrix(n*m, target_rank, scale=2)
        V = _random_half_integer_matrix(m*p, target_rank, scale=2)
        W = _random_half_integer_matrix(p*n, target_rank, scale=2)
        
        err = _compute_error_fast(U, V, W, n, m, p)
        
        # If error is small, try to refine
        if err < 1.0:
            result = _try_local_refinement(U, V, W, n, m, p, time_limit - (time.time() - start_time))
            if result is not None:
                U_r, V_r, W_r = result
                if _verify_exact_fast(U_r, V_r, W_r, n, m, p):
                    return U_r, V_r, W_r
    
    return None

def _gradient_descent_search(n, m, p, target_rank, time_limit):
    """Use gradient descent to find a decomposition, then snap to half-integers."""
    start_time = time.time()
    
    for trial in range(5):
        if time.time() > start_time + time_limit:
            break
        
        # Random initialization
        U = np.random.randn(n*m, target_rank) * 0.5
        V = np.random.randn(m*p, target_rank) * 0.5
        W = np.random.randn(p*n, target_rank) * 0.5
        
        # Gradient descent with adaptive learning rate
        lr = 0.1
        best_err = float('inf')
        
        for step in range(2000):
            if time.time() > start_time + time_limit:
                break
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_row = U[i*m + j]
                        v_row = V[j*p + k]
                        w_row = W[k*n + i]
                        s = np.dot(u_row, v_row * w_row)
                        err_term = 2 * (s - 1.0)
                        dU[i*m + j] += err_term * v_row * w_row
                        dV[j*p + k] += err_term * u_row * w_row
                        dW[k*n + i] += err_term * u_row * v_row
            
            # Update with clipping
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            err = _compute_error_fast(U, V, W, n, m, p)
            if err < best_err:
                best_err = err
            
            if err < 1e-8:
                break
            
            # Adaptive learning rate
            if err > best_err * 1.5:
                lr *= 0.5
            elif step % 100 == 0:
                lr *= 0.95
        
        # Try to snap to half-integers
        if best_err < 1e-5:
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            if _verify_exact_fast(U_snap, V_snap, W_snap, n, m, p):
                return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
            
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            if _verify_exact_fast(U_int, V_int, W_int, n, m, p):
                return U_int.tolist(), V_int.tolist(), W_int.tolist()
            
            # Try local refinement from the GD solution
            result = _try_local_refinement(U, V, W, n, m, p, max(0, time_limit - (time.time() - start_time)))
            if result is not None:
                U_r, V_r, W_r = result
                if _verify_exact_fast(U_r, V_r, W_r, n, m, p):
                    return U_r, V_r, W_r
    
    return None

def _try_rank(n, m, p, target_rank, time_limit):
    """Try multiple methods to find a decomposition of target_rank."""
    start_time = time.time()
    
    # Method 1: Structured random search
    remaining = time_limit - (time.time() - start_time)
    if remaining > 5:
        result = _structured_search_rank(n, m, p, target_rank, remaining * 0.4)
        if result is not None:
            return result
    
    # Method 2: Gradient descent
    remaining = time_limit - (time.time() - start_time)
    if remaining > 5:
        result = _gradient_descent_search(n, m, p, target_rank, remaining * 0.6)
        if result is not None:
            return result
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Schoolbook fallback (rank 75)
    schoolbook = _schoolbook_decomposition(n, m, p)
    schoolbook_rank = n * m * p
    
    # Known lower bounds: for 3x5x5, the rank is at least max(n*m, m*p, p*n) = 25
    # But actually, matrix multiplication rank is known to be less than n*m*p.
    # For 2x2x2, rank 7 is known. For larger matrices, we need to search.
    
    # Try ranks from just below schoolbook down
    # For 3x5x5 = 75, try 74, 73, ... down to some limit
    min_rank_to_try = max(25, schoolbook_rank - 30)
    
    best_result = None
    best_rank = schoolbook_rank
    
    for target_rank in range(schoolbook_rank - 1, min_rank_to_try - 1, -1):
        if time.time() > deadline - 15.0:
            break
        
        # Allocate time: more time for lower ranks (harder)
        ranks_remaining = target_rank - min_rank_to_try + 1
        time_per_rank = min(15.0, (deadline - time.time() - 10.0) / max(1, ranks_remaining))
        
        if time_per_rank < 2.0:
            break
        
        result = _try_rank(n, m, p, target_rank, time_per_rank)
        
        if result is not None:
            U_r, V_r, W_r = result
            # Final verification
            if _verify_exact_fast(U_r, V_r, W_r, n, m, p):
                return {"U": U_r, "V": V_r, "W": W_r}
    
    # Return schoolbook if nothing better found
    return schoolbook

# EVOLVE-BLOCK-END
