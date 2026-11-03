"""Find rank-minimal tensor decomposition for 3x3 * 3x4 matrix multiplication."""
import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that the decomposition is exact."""
    n, m, p = n, m, p
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_exact(U, V, W, n, m, p):
    """Verify with exact arithmetic using Fractions."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j, c]) * Fraction(V[j*p+k, c]) * Fraction(W[k*n+i, c])
                if s != 1:
                    return False
    return True

def _round_to_half(x):
    """Round to nearest half-integer."""
    return round(2*x) / 2.0

def _try_round(U, V, W, n, m, p):
    """Try rounding to half-integers and verify."""
    Ur = np.array([[_round_to_half(v) for v in row] for row in U])
    Vr = np.array([[_round_to_half(v) for v in row] for row in V])
    Wr = np.array([[_round_to_half(v) for v in row] for row in W])
    if _verify_exact(Ur, Vr, Wr, n, m, p):
        return Ur, Vr, Wr
    return None

def _als_decompose(r, n, m, p, init_seed=42, max_iter=2000, tol=1e-10):
    """Alternating least squares for tensor decomposition."""
    rng = np.random.RandomState(init_seed)
    U = rng.randn(n*m, r) * 0.1
    V = rng.randn(m*p, r) * 0.1
    W = rng.randn(p*n, r) * 0.1
    
    # Precompute index mappings
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    for it in range(max_iter):
        # Update U: for each column c, U[:,c] = argmin ||T - sum_c' ...||
        # Actually, for fixed V and W, the problem for U is:
        # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # This is linear in U for fixed V,W.
        
        # Build linear system for U
        # Variables: U[i*m+j, c] for all i,j,c
        # Equations: for each (i,j,k): sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1
        
        # For each (i,j), we have r variables U[i*m+j, 0..r-1]
        # And the equation for (i,j,k) involves U[i*m+j, :] dotted with (V[j*p+k,:] * W[k*n+i,:])
        
        # So for fixed (i,j), we have p equations (one for each k) in r unknowns.
        # If p < r, this is underdetermined. We solve least squares.
        
        new_U = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                # Build A matrix: p x r
                A = np.zeros((p, r))
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                # Solve A @ U[i*m+j, :] = ones(p)
                sol, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                new_U[i*m+j, :] = sol
        
        # Update V
        new_V = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, r))
                for i in range(n):
                    A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                sol, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(n), rcond=None)
                new_V[j*p+k, :] = sol
        
        # Update W
        new_W = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, r))
                for j in range(m):
                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                sol, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                new_W[k*n+i, :] = sol
        
        U, V, W = new_U, new_V, new_W
        
        # Check convergence
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err += (s - 1.0)**2
        if err < tol:
            break
    
    return U, V, W, err

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    best_result = None
    best_rank = n * m * p  # schoolbook
    
    # Get schoolbook as fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_result = (U_sb, V_sb, W_sb, best_rank)
    
    # Try ranks from low to high
    for r in range(15, best_rank + 1):
        if time.time() > deadline - 10:
            break
        
        found = False
        # Try multiple random initializations
        max_trials = 50
        for trial in range(max_trials):
            if time.time() > deadline - 5:
                break
            
            seed = 42 + trial * 1000 + r
            try:
                U, V, W, err = _als_decompose(r, n, m, p, init_seed=seed, max_iter=500)
            except Exception:
                continue
            
            if err < 1e-8:
                # Try to round to half-integers
                rounded = _try_round(U, V, W, n, m, p)
                if rounded is not None:
                    Ur, Vr, Wr = rounded
                    best_result = (Ur, Vr, Wr, r)
                    found = True
                    break
                else:
                    # Try integer rounding
                    Ui = np.round(U).astype(float)
                    Vi = np.round(V).astype(float)
                    Wi = np.round(W).astype(float)
                    if _verify_exact(Ui, Vi, Wi, n, m, p):
                        best_result = (Ui, Vi, Wi, r)
                        found = True
                        break
        
        if found:
            break
    
    U, V, W, rank = best_result
    # Convert to lists of lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
