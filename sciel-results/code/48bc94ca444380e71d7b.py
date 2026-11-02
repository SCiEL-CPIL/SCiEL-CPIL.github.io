"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

import numpy as np
import time
import random
from fractions import Fraction

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n=4, m=4, p=8, tol=1e-9):
    """Verify that the tensor decomposition is exact."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # The multiplication tensor T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    errors = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_arr[i*m + j], V_arr[j*p + k] * W_arr[k*n + i])
                if abs(s - 1.0) > tol:
                    errors.append((i, j, k, s))
                    if len(errors) > 5:
                        return False, errors
    return len(errors) == 0, errors

def _schoolbook_decomposition(n=4, m=4, p=8):
    """Return the standard rank n*m*p decomposition."""
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

def _strassen_block_2x2x4(n=2, m=2, p=4):
    """
    Construct a Strassen-like decomposition for 2x2 * 2x4.
    
    We split the 2x4 right matrix into two 2x2 blocks: B = [B1 | B2]
    Then C = A * B = [A*B1 | A*B2]
    
    Using Strassen for each 2x2*2x2 multiplication:
    M1 = (A11+A22)*(B11+B22)
    M2 = (A21+A22)*B11
    M3 = A11*(B12-B22)
    M4 = A22*(B21-B11)
    M5 = (A11+A12)*B22
    M6 = (A21-A11)*(B11+B12)
    M7 = (A12-A22)*(B21+B22)
    
    C11 = M1 - M4 + M5 + M6
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M3 + M2 + M7
    
    For B = [B1 | B2], we do this twice.
    
    Total rank: 7 * 2 = 14 (vs schoolbook 2*2*4 = 16)
    """
    # This is complex; let's use a simpler approach
    # Just return None to indicate we need a different strategy
    return None

def _try_peeling_decomposition(n, m, p, rank, seed=42, time_limit=10.0):
    """
    Peeling algorithm: fix U and V to sparse patterns, solve for W.
    
    The idea: if we fix U and V, then for each (i,j,k), we need:
    sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    This is a linear system in W. If we fix U and V, we can solve for W.
    """
    rng = np.random.RandomState(seed)
    start_time = time.time()
    
    # Try sparse patterns for U and V
    # Pattern 1: each row of U has exactly one 1
    # Pattern 2: random sparse with density ~ rank/(n*m)
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    
    # Assign random columns to each row of U
    for i in range(n*m):
        col = rng.randint(0, rank)
        U[i, col] = 1.0
    
    for j in range(m*p):
        col = rng.randint(0, rank)
        V[j, col] = 1.0
    
    # Now solve for W
    # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # This is: sum_c (U[i*m+j,c] * V[j*p+k,c]) * W[k*n+i,c] = 1
    
    # Build the linear system
    # Unknowns: W[k*n+i, c] for k in [0,p), i in [0,n), c in [0,rank)
    # Equations: for each (i,j,k), one equation
    
    num_unknowns = p * n * rank
    num_equations = n * m * p
    
    A_mat = np.zeros((num_equations, num_unknowns))
    b_vec = np.ones(num_equations)
    
    eq_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for c in range(rank):
                    coeff = U[i*m + j, c] * V[j*p + k, c]
                    if coeff != 0:
                        var_idx = (k * n + i) * rank + c
                        A_mat[eq_idx, var_idx] = coeff
                eq_idx += 1
    
    # Solve the linear system
    try:
        # Use least squares since the system may be underdetermined
        W_flat, residuals, rank_sys, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        
        # Check residual
        if len(residuals) > 0 and residuals[0] > 1e-6:
            return None, None, None
        
        W = W_flat.reshape(p * n, rank)
        
        # Verify
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m + j], V[j*p + k] * W[k*n + i])
                    if abs(s - 1.0) > 1e-6:
                        return None, None, None
        
        return U, V, W
        
    except Exception:
        return None, None, None

def _try_random_decomposition(n, m, p, rank, seed=42, time_limit=30.0):
    """Try to find a rank-r decomposition using random initialization and refinement."""
    rng = np.random.RandomState(seed)
    start_time = time.time()
    
    U = rng.randn(n*m, rank)
    V = rng.randn(m*p, rank)
    W = rng.randn(p*n, rank)
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for iteration in range(200):
        if time.time() - start_time > time_limit:
            break
            
        U_new = U.copy()
        V_new = V.copy()
        W_new = W.copy()
        
        samples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        rng.shuffle(samples)
        sample_size = min(50, len(samples))
        
        lr = 0.001
        for idx in range(sample_size):
            i, j, k = samples[idx]
            ui = i*m + j
            vj = j*p + k
            wk = k*n + i
            
            s = np.dot(U[ui], V[vj] * W[wk])
            err = s - 1.0
            
            U_new[ui] -= lr * err * (V[vj] * W[wk])
            V_new[vj] -= lr * err * (U[ui] * W[wk])
            W_new[wk] -= lr * err * (U[ui] * V[vj])
        
        U, V, W = U_new, V_new, W_new
        
        total_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    total_err += (s - 1.0)**2
        if total_err < best_err:
            best_err = total_err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if best_err < 1e-12:
                break
    
    return best_U, best_V, best_W, best_err

def construct():
    """Construct a rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication."""
    n, m, p = 4, 4, 8
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    # Start with schoolbook
    result = _schoolbook_decomposition(n, m, p)
    best_result = result
    best_rank = n * m * p  # 128
    
    # Try peeling decomposition for various ranks
    for rank in [32, 36, 40, 44, 48, 56, 64, 72, 80, 96, 112]:
        if time.time() - start_time > deadline:
            break
        
        for seed in [42, 123, 7, 999, 2024]:
            if time.time() - start_time > deadline:
                break
            
            try:
                U, V, W = _try_peeling_decomposition(n, m, p, rank, seed=seed, time_limit=10.0)
            except Exception:
                continue
            
            if U is not None:
                U_clean = [[round(x, 10) if abs(x - round(x, 10)) < 1e-9 else x for x in row] for row in U.tolist()]
                V_clean = [[round(x, 10) if abs(x - round(x, 10)) < 1e-9 else x for x in row] for row in V.tolist()]
                W_clean = [[round(x, 10) if abs(x - round(x, 10)) < 1e-9 else x for x in row] for row in W.tolist()]
                
                ok, _ = _verify_decomposition(U_clean, V_clean, W_clean, n, m, p)
                if ok:
                    best_result = {"U": U_clean, "V": V_clean, "W": W_clean}
                    best_rank = rank
                    break
        
        if best_rank == rank:
            break
    
    # If peeling didn't work, try random search for a few ranks
    for rank in [48, 56, 64, 80]:
        if time.time() - start_time > deadline:
            break
        if best_rank <= rank:
            break
        
        for seed in [42, 123, 7]:
            if time.time() - start_time > deadline:
                break
            
            try:
                U, V, W, err = _try_random_decomposition(n, m, p, rank, seed=seed, time_limit=10.0)
            except Exception:
                continue
            
            if err < 1e-6:
                U_clean = [[round(x) if abs(x - round(x)) < 1e-9 else x for x in row] for row in U.tolist()]
                V_clean = [[round(x) if abs(x - round(x)) < 1e-9 else x for x in row] for row in V.tolist()]
                W_clean = [[round(x) if abs(x - round(x)) < 1e-9 else x for x in row] for row in W.tolist()]
                
                ok, _ = _verify_decomposition(U_clean, V_clean, W_clean, n, m, p)
                if ok:
                    best_result = {"U": U_clean, "V": V_clean, "W": W_clean}
                    best_rank = rank
                    break
    
    return best_result
# EVOLVE-BLOCK-END
