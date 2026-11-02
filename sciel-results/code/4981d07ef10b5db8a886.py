"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def _is_half_int(val, tol=1e-9):
    """Check if val is a half-integer."""
    d = val * 2
    return abs(d - round(d)) < tol

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify exact decomposition."""
    r = len(U[0]) if U else 0
    # Check half-integer entries
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if not _is_half_int(val):
                    return False
    # Verify trilinear form
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_decomposition(n, m, p, rank, max_iters, seed):
    """Try to find a rank decomposition via random search."""
    rng = random.Random(seed)
    U_size = n * m
    V_size = m * p
    W_size = p * n
    
    for it in range(max_iters):
        # Random initialization
        U = [[rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0]) for _ in range(rank)] for _ in range(U_size)]
        V = [[rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0]) for _ in range(rank)] for _ in range(V_size)]
        W = [[rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0]) for _ in range(rank)] for _ in range(W_size)]
        
        if _verify(U, V, W, n, m, p):
            return U, V, W
    
    return None

def _try_fixed_UV_solve_W(n, m, p, rank, max_iters, seed):
    """Fix U and V randomly, solve linear system for W."""
    import numpy as np
    rng = random.Random(seed)
    U_size = n * m
    V_size = m * p
    W_size = p * n
    
    for it in range(max_iters):
        U = [[rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0]) for _ in range(rank)] for _ in range(U_size)]
        V = [[rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0]) for _ in range(rank)] for _ in range(V_size)]
        
        # Build linear system for W
        # For each (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # Variables: W[k*n+i, c], total W_size*rank variables
        # Equations: n*m*p
        
        num_eqs = n * m * p
        num_vars = W_size * rank
        
        # A: num_eqs x num_vars
        A = np.zeros((num_eqs, num_vars))
        b = np.ones(num_eqs)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(rank):
                        var_idx = (k * n + i) * rank + c
                        A[eq_idx, var_idx] = U[i * m + j][c] * V[j * p + k][c]
                    eq_idx += 1
        
        # Solve least squares (underdetermined)
        try:
            x, residuals, rank_np, sv = np.linalg.lstsq(A, b, rcond=None)
        except Exception:
            continue
        
        # Check if residual is small
        if len(residuals) > 0 and residuals[0] > 1e-6:
            continue
        
        # Check if solution has half-integer entries
        valid = True
        for val in x:
            if not _is_half_int(val):
                valid = False
                break
        
        if valid:
            W = [[0.0] * rank for _ in range(W_size)]
            for var_idx in range(num_vars):
                ki = var_idx // rank
                c = var_idx % rank
                W[ki][c] = x[var_idx]
            
            if _verify(U, V, W, n, m, p):
                return U, V, W
    
    return None

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Try ranks from 21 upward
    for rank in [21, 22, 23, 24]:
        if time.time() > deadline:
            break
        
        # Try fixed UV solve W approach
        max_iters = 200
        result = _try_fixed_UV_solve_W(n, m, p, rank, max_iters, seed=42)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
        
        if time.time() > deadline:
            break
        
        # Try pure random
        max_iters = 500
        result = _try_decomposition(n, m, p, rank, max_iters, seed=123)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
    rank = n * m * p
    U_size = n * m
    V_size = m * p
    W_size = p * n
    U = [[0.0] * rank for _ in range(U_size)]
    V = [[0.0] * rank for _ in range(V_size)]
    W = [[0.0] * rank for _ in range(W_size)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
