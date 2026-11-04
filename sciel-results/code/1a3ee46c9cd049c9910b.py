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

def _build_linear_system(U, V, n, m, p, r):
    """Build the linear system A*w = b for solving W given U, V."""
    num_vars = p * n * r
    num_eqs = n * m * p
    A = np.zeros((num_eqs, num_vars))
    b = np.ones(num_eqs)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                eq_idx = i * m * p + j * p + k
                for r_idx in range(r):
                    var_idx = (k * n + i) * r + r_idx
                    A[eq_idx, var_idx] = U[i * m + j, r_idx] * V[j * p + k, r_idx]
    return A, b

def _try_rank_structured(n, m, p, r, deadline, max_iters=20000):
    """
    Try to find a rank-r decomposition using structured patterns.
    Strategy: Use block-diagonal structure where each rank term handles
    a subset of (i,j,k) triples, then solve for W.
    """
    T = _target_tensor(n, m, p)
    best = None
    
    for it in range(max_iters):
        if time.time() > deadline - 5:
            break
        
        # Generate structured U, V
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        
        # Pattern 1: Each column handles a small block
        for col in range(r):
            # Choose which (i,j) pairs this column affects in U
            num_u = random.randint(1, min(4, n * m))
            idx_u = random.sample(range(n * m), num_u)
            for idx in idx_u:
                U[idx, col] = random.choice([-2, -1, -0.5, 0.5, 1, 2])
            
            # Choose which (j,k) pairs this column affects in V
            num_v = random.randint(1, min(4, m * p))
            idx_v = random.sample(range(m * p), num_v)
            for idx in idx_v:
                V[idx, col] = random.choice([-2, -1, -0.5, 0.5, 1, 2])
        
        # Build and solve linear system
        A, b = _build_linear_system(U, V, n, m, p, r)
        
        try:
            w, residuals, rank_sys, sv = np.linalg.lstsq(A, b, rcond=None)
            if np.allclose(A @ w, b, atol=1e-9):
                # Try rounding to half-integers
                for scale in [2, 4, 8]:
                    w_rounded = np.round(w * scale) / scale
                    if np.allclose(A @ w_rounded, b, atol=1e-9):
                        # Check entries are "nice"
                        W_list = [[0] * r for _ in range(p * n)]
                        for idx in range(p * n):
                            for r_idx in range(r):
                                W_list[idx][r_idx] = float(w_rounded[idx * r + r_idx])
                        
                        U_list = [[0] * r for _ in range(n * m)]
                        V_list = [[0] * r for _ in range(m * p)]
                        for idx in range(n * m):
                            for r_idx in range(r):
                                U_list[idx][r_idx] = float(U[idx, r_idx])
                        for idx in range(m * p):
                            for r_idx in range(r):
                                V_list[idx][r_idx] = float(V[idx, r_idx])
                        
                        if _is_valid(U_list, V_list, W_list, n, m, p):
                            return {"U": U_list, "V": V_list, "W": W_list}
        except Exception:
            pass
    
    return None

def _try_rank_greedy_merge(n, m, p, r, deadline, max_iters=50000):
    """
    Start from a known valid decomposition at higher rank and try to merge terms.
    """
    # Start from schoolbook
    base = _schoolbook(n, m, p)
    rank = n * m * p
    
    if rank <= r:
        if _is_valid(base["U"], base["V"], base["W"], n, m, p):
            return base
    
    # Try to reduce rank by merging pairs of terms
    U = [row[:] for row in base["U"]]
    V = [row[:] for row in base["V"]]
    W = [row[:] for row in base["W"]]
    current_rank = rank
    
    for merge_attempt in range(1000):
        if time.time() > deadline - 10:
            break
        if current_rank <= r:
            break
        
        # Try to merge two random terms
        if current_rank < 2:
            break
        
        idx1, idx2 = random.sample(range(current_rank), 2)
        
        # Try to combine terms idx1 and idx2
        # We want to find a new term that replaces both
        # This is complex; instead, try zeroing out one term if possible
        
        # Simple approach: try setting one term to zero and see if we can adjust others
        # This is a heuristic - skip for now and use random perturbation
        
        # Alternative: randomly perturb and re-solve
        # For now, just break if we can't find a simple merge
        pass
    
    return None

def _try_rank_r(n, m, p, r, deadline, max_iters=30000):
    """
    Try to find a rank-r decomposition using multiple strategies.
    """
    # Strategy 1: Structured random
    result = _try_rank_structured(n, m, p, r, deadline, max_iters=max_iters // 3)
    if result is not None:
        return result
    
    # Strategy 2: More random with different coefficient ranges
    T = _target_tensor(n, m, p)
    
    for it in range(max_iters):
        if time.time() > deadline - 5:
            break
        
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        
        for col in range(r):
            # Sparse with varying density
            density = random.choice([0.1, 0.2, 0.3])
            for idx in range(n * m):
                if random.random() < density:
                    U[idx, col] = random.choice([-3, -2, -1, -0.5, 0.5, 1, 2, 3])
            for idx in range(m * p):
                if random.random() < density:
                    V[idx, col] = random.choice([-3, -2, -1, -0.5, 0.5, 1, 2, 3])
        
        A, b = _build_linear_system(U, V, n, m, p, r)
        
        try:
            w, residuals, rank_sys, sv = np.linalg.lstsq(A, b, rcond=None)
            if np.allclose(A @ w, b, atol=1e-9):
                # Try multiple rounding scales
                for scale in [1, 2, 4, 8, 16]:
                    w_rounded = np.round(w * scale) / scale
                    if np.allclose(A @ w_rounded, b, atol=1e-9):
                        W_list = [[0] * r for _ in range(p * n)]
                        for idx in range(p * n):
                            for r_idx in range(r):
                                W_list[idx][r_idx] = float(w_rounded[idx * r + r_idx])
                        
                        U_list = [[0] * r for _ in range(n * m)]
                        V_list = [[0] * r for _ in range(m * p)]
                        for idx in range(n * m):
                            for r_idx in range(r):
                                U_list[idx][r_idx] = float(U[idx, r_idx])
                        for idx in range(m * p):
                            for r_idx in range(r):
                                V_list[idx][r_idx] = float(V[idx, r_idx])
                        
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
    
    # Theoretical lower bound
    lb = max(n * m, m * p, p * n)
    # For 4x4x5: max(16, 20, 20) = 20
    
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try ranks from lower bound upward
    for r in range(lb, min(best_rank, 45)):
        if time.time() > deadline - 15:
            break
        
        # Use more iterations for lower ranks (harder to find)
        iters = max(5000, 20000 - (r - lb) * 200)
        
        result = _try_rank_r(n, m, p, r, deadline, max_iters=iters)
        if result is not None:
            best_result = result
            best_rank = r
            # Don't break - try to find even lower rank? No, we're going up.
            # We found rank r, so we can stop since we're searching from low to high.
            break
    
    return best_result
