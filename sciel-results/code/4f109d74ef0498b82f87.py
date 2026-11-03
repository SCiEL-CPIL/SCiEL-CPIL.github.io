import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 6]

def _verify_exact(U, V, W, n, m, p, R):
    """Verify the decomposition exactly using Fractions."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = Fraction(0, 1)
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                for r in range(R):
                    u_val = Fraction(str(U[idx_u, r]))
                    v_val = Fraction(str(V[idx_v, r]))
                    w_val = Fraction(str(W[idx_w, r]))
                    val += u_val * v_val * w_val
                if val != 1:
                    return False
    return True

def _snap_to_half_integers(mat, tol=0.01):
    """Snap values to the nearest half-integer if within tolerance."""
    snapped = np.zeros_like(mat)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            nearest = round(val * 2) / 2.0
            if abs(val - nearest) < tol:
                snapped[i, j] = nearest
            else:
                snapped[i, j] = nearest
    return snapped

def _als_step_U(U, V, W, n, m, p, R):
    """Update U using least squares."""
    new_U = np.zeros_like(U)
    for i in range(n):
        for j in range(m):
            idx_u = i * m + j
            A = np.zeros((p, R))
            for k in range(p):
                idx_v = j * p + k
                idx_w = k * n + i
                A[k, :] = V[idx_v, :] * W[idx_w, :]
            b = np.ones(p)
            x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            new_U[idx_u, :] = x
    return new_U

def _als_step_V(U, V, W, n, m, p, R):
    """Update V using least squares."""
    new_V = np.zeros_like(V)
    for j in range(m):
        for k in range(p):
            idx_v = j * p + k
            A = np.zeros((n, R))
            for i in range(n):
                idx_u = i * m + j
                idx_w = k * n + i
                A[i, :] = U[idx_u, :] * W[idx_w, :]
            b = np.ones(n)
            x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            new_V[idx_v, :] = x
    return new_V

def _als_step_W(U, V, W, n, m, p, R):
    """Update W using least squares."""
    new_W = np.zeros_like(W)
    for k in range(p):
        for i in range(n):
            idx_w = k * n + i
            A = np.zeros((m, R))
            for j in range(m):
                idx_u = i * m + j
                idx_v = j * p + k
                A[j, :] = U[idx_u, :] * V[idx_v, :]
            b = np.ones(m)
            x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            new_W[idx_w, :] = x
    return new_W

def _calculate_error(U, V, W, n, m, p, R):
    """Calculate the Frobenius norm of the error."""
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                for r in range(R):
                    val += U[idx_u, r] * V[idx_v, r] * W[idx_w, r]
                err += (val - 1.0) ** 2
    return np.sqrt(err)

def _als_solve(n, m, p, R, max_time, max_trials=15, max_iters=200):
    """
    Try to find a rank-R decomposition for M_{n,m,p} using ALS.
    Returns (U, V, W) as numpy arrays or None.
    """
    start_time = time.time()
    
    for trial in range(max_trials):
        if time.time() - start_time > max_time:
            break
        
        # Different initialization strategies
        if trial % 5 == 0:
            np.random.seed(42 + trial)
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
        elif trial % 5 == 1:
            np.random.seed(100 + trial)
            U = np.random.randint(-3, 4, size=(n * m, R)).astype(float)
            V = np.random.randint(-3, 4, size=(m * p, R)).astype(float)
            W = np.random.randint(-3, 4, size=(p * n, R)).astype(float)
        elif trial % 5 == 2:
            np.random.seed(200 + trial)
            U = np.zeros((n * m, R))
            V = np.zeros((m * p, R))
            W = np.zeros((p * n, R))
            for i in range(n * m):
                U[i, i % R] = 1.0
            for j in range(m * p):
                V[j, j % R] = 1.0
            for k in range(p * n):
                W[k, k % R] = 1.0
        elif trial % 5 == 3:
            np.random.seed(300 + trial)
            U = np.zeros((n * m, R))
            V = np.zeros((m * p, R))
            W = np.zeros((p * n, R))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col < R:
                            U[i * m + j, col] = 1.0
                            V[j * p + k, col] = 1.0
                            W[k * n + i, col] = 1.0
                        col += 1
            U += 0.1 * np.random.randn(n * m, R)
            V += 0.1 * np.random.randn(m * p, R)
            W += 0.1 * np.random.randn(p * n, R)
        else:
            np.random.seed(400 + trial)
            # Half-integer random initialization
            U = np.random.randint(-4, 5, size=(n * m, R)) / 2.0
            V = np.random.randint(-4, 5, size=(m * p, R)) / 2.0
            W = np.random.randint(-4, 5, size=(p * n, R)) / 2.0
        
        prev_err = float('inf')
        for it in range(max_iters):
            if time.time() - start_time > max_time:
                break
                
            U = _als_step_U(U, V, W, n, m, p, R)
            V = _als_step_V(U, V, W, n, m, p, R)
            W = _als_step_W(U, V, W, n, m, p, R)
            
            if it % 2 == 0:
                err = _calculate_error(U, V, W, n, m, p, R)
                if err < 1e-9:
                    U_snap = _snap_to_half_integers(U, tol=0.05)
                    V_snap = _snap_to_half_integers(V, tol=0.05)
                    W_snap = _snap_to_half_integers(W, tol=0.05)
                    
                    if _verify_exact(U_snap, V_snap, W_snap, n, m, p, R):
                        return U_snap, V_snap, W_snap
                elif err > prev_err * 1.05 and it > 10:
                    break
                prev_err = err
                
    return None

def _try_direct_half_integer_search(n, m, p, R, max_time, num_samples=500):
    """
    Try to find a rank-R decomposition by randomly sampling half-integer matrices
    and checking if they exactly decompose the tensor.
    """
    start_time = time.time()
    
    for sample in range(num_samples):
        if time.time() - start_time > max_time:
            break
        
        np.random.seed(1000 + sample * 7)
        
        # Random half-integer matrices
        U = np.random.randint(-4, 5, size=(n * m, R)) / 2.0
        V = np.random.randint(-4, 5, size=(m * p, R)) / 2.0
        W = np.random.randint(-4, 5, size=(p * n, R)) / 2.0
        
        # Quick check: compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    for r in range(R):
                        val += U[idx_u, r] * V[idx_v, r] * W[idx_w, r]
                    err += (val - 1.0) ** 2
        
        if err < 1e-12:
            if _verify_exact(U, V, W, n, m, p, R):
                return U, V, W
        
        # Try a few ALS iterations from this starting point
        for it in range(50):
            if time.time() - start_time > max_time:
                break
            U = _als_step_U(U, V, W, n, m, p, R)
            V = _als_step_V(U, V, W, n, m, p, R)
            W = _als_step_W(U, V, W, n, m, p, R)
            
            if it % 10 == 0:
                err = _calculate_error(U, V, W, n, m, p, R)
                if err < 1e-9:
                    U_snap = _snap_to_half_integers(U, tol=0.05)
                    V_snap = _snap_to_half_integers(V, tol=0.05)
                    W_snap = _snap_to_half_integers(W, tol=0.05)
                    
                    if _verify_exact(U_snap, V_snap, W_snap, n, m, p, R):
                        return U_snap, V_snap, W_snap
        
    return None

def _schoolbook(n, m, p):
    """Return the standard schoolbook decomposition."""
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

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Lower bound for rank
    min_possible = max(n * m, m * p, p * n)
    
    # Strategy 1: Try direct ALS on M_{2,3,6} at various ranks
    # Start from the lower bound and go up
    for target_rank in range(min_possible, n * m * p + 1):
        if time.time() >= deadline - 5:
            break
        
        remaining_time = deadline - time.time()
        
        # Allocate time based on rank
        if target_rank <= 20:
            rank_time = min(20, remaining_time * 0.25)
        elif target_rank <= 25:
            rank_time = min(15, remaining_time * 0.2)
        elif target_rank <= 30:
            rank_time = min(10, remaining_time * 0.15)
        else:
            rank_time = min(5, remaining_time * 0.1)
        
        if rank_time < 3:
            continue
        
        # First try ALS
        result = _als_solve(n, m, p, target_rank, max_time=rank_time * 0.6,
                           max_trials=15, max_iters=200)
        
        if result is not None:
            U_np, V_np, W_np = result
            if _verify_exact(U_np, V_np, W_np, n, m, p, target_rank):
                return {"U": U_np.tolist(), "V": V_np.tolist(), "W": W_np.tolist()}
        
        # Then try half-integer random search
        remaining_for_search = rank_time * 0.4
        if remaining_for_search > 2:
            result = _try_direct_half_integer_search(n, m, p, target_rank,
                                                     max_time=remaining_for_search,
                                                     num_samples=200)
            if result is not None:
                U_np, V_np, W_np = result
                if _verify_exact(U_np, V_np, W_np, n, m, p, target_rank):
                    return {"U": U_np.tolist(), "V": V_np.tolist(), "W": W_np.tolist()}
        
        remaining_time = deadline - time.time()
        if remaining_time < 10:
            break
    
    # Fallback: schoolbook
    return _schoolbook(n, m, p)
