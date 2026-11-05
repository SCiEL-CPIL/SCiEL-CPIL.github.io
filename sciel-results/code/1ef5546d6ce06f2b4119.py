"""Matrix multiplication 3,5,6 - rank-minimal tensor decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [3, 5, 6]
DEADLINE = 150.0
START_TIME = time.time()

def _elapsed():
    return time.time() - START_TIME

def _remaining():
    return DEADLINE - _elapsed() - 5.0

def _snap_to_half(x):
    return round(2 * x) / 2.0

def _verify_fast(U, V, W, n, m, p):
    if not U or not V or not W:
        return False
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    if Ua.shape != (n*m, rank) or Va.shape != (m*p, rank) or Wa.shape != (p*n, rank):
        return False
    
    for arr in [Ua, Va, Wa]:
        if not np.allclose(2*arr, np.round(2*arr), atol=1e-9):
            return False
    
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    if not np.allclose(T, np.ones((n, m, p)), atol=1e-9):
        return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _als_refine(U, V, W, n, m, p, max_iter=200, tol=1e-10):
    """Alternating least squares refinement."""
    Ua = np.array(U, dtype=float).copy()
    Va = np.array(V, dtype=float).copy()
    Wa = np.array(W, dtype=float).copy()
    rank = Ua.shape[1]
    
    for it in range(max_iter):
        if _remaining() < 3.0:
            break
        
        # Update U
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, rank))
                for k in range(p):
                    M[k, :] = Va[j*p + k] * Wa[k*n + i]
                try:
                    Ua[i*m + j] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, rank))
                for i in range(n):
                    M[i, :] = Ua[i*m + j] * Wa[k*n + i]
                try:
                    Va[j*p + k] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = Ua[i*m + j] * Va[j*p + k]
                try:
                    Wa[k*n + i] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        U3 = Ua.reshape(n, m, rank)
        V3 = Va.reshape(m, p, rank)
        W3 = Wa.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        error = np.max(np.abs(T - 1.0))
        
        if error < tol:
            break
    
    return Ua, Va, Wa

def _svd_init(n, m, p, rank, seed=None):
    """
    Initialize using SVD of the tensor's mode-3 unfolding.
    The multiplication tensor T[i,j,k] = delta_{i,k} (identity in appropriate sense)
    For matmul, T[i,j,k] = 1 if we think of it as the "identity" tensor.
    Actually for matmul tensor, T[i,j,k] represents the coefficient such that
    C[i,k] = sum_j A[i,j]*B[j,k], so the tensor is T[i,j,k] = 1 for all valid (i,j,k)
    in the sense that it's the structure tensor.
    
    Better: use random init with structured patterns based on Strassen.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    # Use a structured initialization: each rank component gets a "pattern"
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for r in range(rank):
        # Each component: pick a random (i,j,k) and set non-zeros in a structured way
        pattern = rng.randint(0, 4)
        
        if pattern == 0:
            # Single element
            i = rng.randint(0, n-1)
            j = rng.randint(0, m-1)
            k = rng.randint(0, p-1)
            U[i*m+j, r] = rng.choice([0.5, 1.0, 1.5])
            V[j*p+k, r] = rng.choice([0.5, 1.0, 1.5])
            W[k*n+i, r] = rng.choice([0.5, 1.0, 1.5])
        elif pattern == 1:
            # Row i in U, column j in V, row k in W
            i = rng.randint(0, n-1)
            j = rng.randint(0, m-1)
            k = rng.randint(0, p-1)
            for jj in range(m):
                U[i*m+jj, r] = rng.choice([0.5, 1.0])
            for kk in range(p):
                V[j*p+kk, r] = rng.choice([0.5, 1.0])
            for ii in range(n):
                W[k*n+ii, r] = rng.choice([0.5, 1.0])
        elif pattern == 2:
            # Strassen-like: two elements in each
            i1 = rng.randint(0, n-1)
            i2 = rng.randint(0, n-1)
            j1 = rng.randint(0, m-1)
            j2 = rng.randint(0, m-1)
            k1 = rng.randint(0, p-1)
            k2 = rng.randint(0, p-1)
            U[i1*m+j1, r] = rng.choice([0.5, 1.0])
            U[i2*m+j2, r] = rng.choice([0.5, 1.0])
            V[j1*p+k1, r] = rng.choice([0.5, 1.0])
            V[j2*p+k2, r] = rng.choice([0.5, 1.0])
            W[k1*n+i1, r] = rng.choice([0.5, 1.0])
            W[k2*n+i2, r] = rng.choice([0.5, 1.0])
        elif pattern == 3:
            # Block: all j for fixed i, all k for fixed j, all i for fixed k
            i = rng.randint(0, n-1)
            j = rng.randint(0, m-1)
            k = rng.randint(0, p-1)
            for jj in range(m):
                U[i*m+jj, r] = 1.0
            for kk in range(p):
                V[j*p+kk, r] = 1.0
            for ii in range(n):
                W[k*n+ii, r] = 1.0
        else:
            # Random sparse
            for _ in range(rng.randint(1, 3)):
                i = rng.randint(0, n-1)
                j = rng.randint(0, m-1)
                k = rng.randint(0, p-1)
                U[i*m+j, r] += rng.choice([0.5, 1.0, -0.5, -1.0])
                V[j*p+k, r] += rng.choice([0.5, 1.0, -0.5, -1.0])
                W[k*n+i, r] += rng.choice([0.5, 1.0, -0.5, -1.0])
    
    return U, V, W

def _try_rank(n, m, p, rank, num_inits=10, seed_base=0):
    """Try to find a rank-rank decomposition using multiple initializations + ALS."""
    best_result = None
    best_error = np.inf
    
    for init_num in range(num_inits):
        if _remaining() < 10.0:
            break
        
        seed = seed_base + init_num * 7919
        
        U, V, W = _svd_init(n, m, p, rank, seed=seed)
        
        # Refine with ALS
        U_ref, V_ref, W_ref = _als_refine(U, V, W, n, m, p, max_iter=150)
        
        U3 = U_ref.reshape(n, m, rank)
        V3 = V_ref.reshape(m, p, rank)
        W3 = W_ref.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        error = np.max(np.abs(T - 1.0))
        
        if error < best_error:
            best_error = error
            best_result = (U_ref, V_ref, W_ref)
        
        if error < 1e-10:
            # Try to snap to half-integers
            U_snapped = np.vectorize(_snap_to_half)(U_ref)
            V_snapped = np.vectorize(_snap_to_half)(V_ref)
            W_snapped = np.vectorize(_snap_to_half)(W_ref)
            
            U_out = U_snapped.tolist()
            V_out = V_snapped.tolist()
            W_out = W_snapped.tolist()
            
            if _verify_fast(U_out, V_out, W_out, n, m, p):
                return U_out, V_out, W_out
            
            # Also try raw
            U_out_raw = U_ref.tolist()
            V_out_raw = V_ref.tolist()
            W_out_raw = W_ref.tolist()
            
            if _verify_fast(U_out_raw, V_out_raw, W_out_raw, n, m, p):
                return U_out_raw, V_out_raw, W_out_raw
    
    return None

def _strassen_2x2x2():
    """Returns U, V, W for 2x2x2 multiplication with rank 7."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00 + A11) * (B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1; W[1, 0] = -1; W[2, 0] = -1
    
    # M2 = (A10 + A11) * B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[0, 1] = -1; W[2, 1] = 1
    
    # M3 = A00 * (B01 - B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[0, 2] = 1; W[2, 2] = -1
    
    # M4 = A11 * (B10 - B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[1, 3] = 1; W[3, 3] = -1
    
    # M5 = (A00 + A01) * B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = -1; W[3, 4] = 1
    
    # M6 = (A10 - A00) * (B00 + B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[0, 5] = 1; W[1, 5] = 1
    
    # M7 = (A01 - A11) * (B10 + B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[2, 6] = 1; W[3, 6] = 1
    
    return U, V, W

def _strassen_3x3x3():
    """
    Known rank-23 decomposition for 3x3x3 (Winograd).
    Returns U, V, W of shape (9, 23).
    """
    # Winograd's rank-23 algorithm for 3x3x3
    # This is a known result. Let me use a simplified version.
    # Actually, let me just return None and rely on other methods.
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook = _schoolbook(n, m, p)
    best = schoolbook
    best_rank = n * m * p
    
    min_rank = max(n*m, m*p, n*p)  # = 30
    
    # Try ranks from min_rank upward
    # Known: r(3,5,6) is between 30 and 90
    # Let's try a focused search
    
    for rank in range(min_rank, min(90, min_rank + 30)):
        if _remaining() < 15.0:
            break
        
        result = _try_rank(n, m, p, rank, num_inits=8, seed_base=rank * 1000)
        if result is not None:
            U, V, W = result
            if _verify_fast(U, V, W, n, m, p):
                best = {"U": U, "V": V, "W": W}
                best_rank = rank
                break
    
    if not _verify_fast(best["U"], best["V"], best["W"], n, m, p):
        best = schoolbook
    
    return best
