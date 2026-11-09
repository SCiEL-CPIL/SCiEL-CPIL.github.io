"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from scipy.optimize import minimize
from itertools import product

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """
    Build the matrix multiplication tensor T[a, b, c] where:
    a = i*m + j (flattened A index), b = j*p + k (flattened B index), c = k*n + i (flattened C index)
    T[a, b, c] = 1 if indices are consistent (same i, j, k), else 0.
    """
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify_exact(T, U, V, W, tol=1e-8):
    """
    Verify that the decomposition is exact (within tolerance).
    T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    """
    r = U.shape[1]
    recon = np.zeros(T.shape, dtype=np.float64)
    for rr in range(r):
        recon += np.einsum('a,b,c->abc', U[:, rr], V[:, rr], W[:, rr])
    residual = T - recon
    return np.max(np.abs(residual)) < tol

def _als_step(T, U, V, W, rank, time_limit, start_time, max_iter=100, tol=1e-10):
    """
    Perform ALS (Alternating Least Squares) iterations to refine the decomposition.
    Returns updated (U, V, W).
    """
    a_size = T.shape[0]
    b_size = T.shape[1]
    c_size = T.shape[2]
    prev_residual = np.inf
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Fix V and W, solve for U
        M_U = np.einsum('br,cr->bcr', V, W).reshape(b_size * c_size, rank)
        T_flat = T.reshape(a_size, b_size * c_size)
        
        try:
            U_new, _, _, _ = np.linalg.lstsq(M_U, T_flat.T, rcond=None)
            U = U_new.T
        except:
            pass
        
        # Fix U and W, solve for V
        M_V = np.einsum('ar,cr->acr', U, W).reshape(a_size * c_size, rank)
        T_flat_v = T.reshape(a_size, c_size, b_size).transpose(0, 2, 1).reshape(b_size, a_size * c_size)
        
        try:
            V_new, _, _, _ = np.linalg.lstsq(M_V, T_flat_v.T, rcond=None)
            V = V_new.T
        except:
            pass
        
        # Fix U and V, solve for W
        M_W = np.einsum('ar,br->abr', U, V).reshape(a_size * b_size, rank)
        T_flat_w = T.reshape(a_size, b_size, c_size).transpose(2, 0, 1).reshape(c_size, a_size * b_size)
        
        try:
            W_new, _, _, _ = np.linalg.lstsq(M_W, T_flat_w.T, rcond=None)
            W = W_new.T
        except:
            pass
        
        # Check convergence
        recon = np.zeros(T.shape, dtype=np.float64)
        for rr in range(rank):
            recon += np.einsum('a,b,c->abc', U[:, rr], V[:, rr], W[:, rr])
        residual = np.max(np.abs(T - recon))
        
        if abs(residual - prev_residual) < tol:
            break
        prev_residual = residual
    
    return U, V, W

def _try_rank_als(T, n, m, p, rank, time_limit, seed=0):
    """
    Try to find a rank-`rank` decomposition of T using ALS with multiple restarts.
    Returns (U, V, W) if found, else None.
    """
    np.random.seed(seed)
    
    a_size = n * m
    b_size = m * p
    c_size = p * n
    
    start_time = time.time()
    
    # Try multiple random initializations
    for attempt in range(25):
        if time.time() - start_time > time_limit:
            break
        
        np.random.seed(seed + attempt)
        
        # Different initialization strategies
        if attempt % 5 == 0:
            # Random Gaussian
            U = np.random.randn(a_size, rank) * 0.1
            V = np.random.randn(b_size, rank) * 0.1
            W = np.random.randn(c_size, rank) * 0.1
        elif attempt % 5 == 1:
            # Sparse random
            U = np.zeros((a_size, rank))
            V = np.zeros((b_size, rank))
            W = np.zeros((c_size, rank))
            for r in range(rank):
                idx_u = np.random.choice(a_size, size=min(rank//2, a_size), replace=False)
                idx_v = np.random.choice(b_size, size=min(rank//2, b_size), replace=False)
                idx_w = np.random.choice(c_size, size=min(rank//2, c_size), replace=False)
                U[idx_u, r] = 1.0
                V[idx_v, r] = 1.0
                W[idx_w, r] = 1.0
        elif attempt % 5 == 2:
            # Structured: assign each (i,j,k) to a column
            col_assignment = np.random.randint(0, rank, size=(n, m, p))
            U = np.zeros((a_size, rank))
            V = np.zeros((b_size, rank))
            W = np.zeros((c_size, rank))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        r = col_assignment[i, j, k]
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        U[a, r] += 1.0
                        V[b, r] += 1.0
                        W[c, r] += 1.0
        elif attempt % 5 == 3:
            # Half-integer structured
            col_assignment = np.random.randint(0, rank, size=(n, m, p))
            U = np.zeros((a_size, rank))
            V = np.zeros((b_size, rank))
            W = np.zeros((c_size, rank))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        r = col_assignment[i, j, k]
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        U[a, r] += 0.5
                        V[b, r] += 0.5
                        W[c, r] += 0.5
        else:
            # Mixed: some columns structured, some random
            U = np.zeros((a_size, rank))
            V = np.zeros((b_size, rank))
            W = np.zeros((c_size, rank))
            for r in range(rank):
                if r % 3 == 0:
                    # Structured
                    i = r % n
                    j = (r // n) % m
                    k = (r // (n * m)) % p
                    U[i * m + j, r] = 1.0
                    V[j * p + k, r] = 1.0
                    W[k * n + i, r] = 1.0
                else:
                    # Random
                    U[np.random.randint(0, a_size), r] = 1.0
                    V[np.random.randint(0, b_size), r] = 1.0
                    W[np.random.randint(0, c_size), r] = 1.0
        
        # Run ALS
        U, V, W = _als_step(T, U, V, W, rank, time_limit, start_time, max_iter=80)
        
        # Check if exact
        if _verify_exact(T, U, V, W, tol=1e-8):
            return U, V, W
    
    return None

def _try_rank_integer_search(T, n, m, p, rank, time_limit, seed=0):
    """
    Try to find a rank-`rank` decomposition with integer/half-integer entries
    using a structured search approach.
    """
    a_size = n * m
    b_size = m * p
    c_size = p * n
    
    start_time = time.time()
    np.random.seed(seed)
    
    for attempt in range(15):
        if time.time() - start_time > time_limit:
            break
        
        np.random.seed(seed + 1000 + attempt)
        
        # Different structured initialization strategies
        if attempt % 3 == 0:
            # Random assignment of (i,j,k) to columns
            col_assignment = np.random.randint(0, rank, size=(n, m, p))
            U = np.zeros((a_size, rank))
            V = np.zeros((b_size, rank))
            W = np.zeros((c_size, rank))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        r = col_assignment[i, j, k]
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        U[a, r] += 1.0
                        V[b, r] += 1.0
                        W[c, r] += 1.0
        elif attempt % 3 == 1:
            # Block structure: group by i
            U = np.zeros((a_size, rank))
            V = np.zeros((b_size, rank))
            W = np.zeros((c_size, rank))
            cols_per_i = rank // n
            for i in range(n):
                base = i * cols_per_i
                for j in range(m):
                    for k in range(p):
                        r = base + (j * p + k) % cols_per_i
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        U[a, r] += 1.0
                        V[b, r] += 1.0
                        W[c, r] += 1.0
        else:
            # Striped pattern
            U = np.zeros((a_size, rank))
            V = np.zeros((b_size, rank))
            W = np.zeros((c_size, rank))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        r = (i * m + j + k) % rank
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        U[a, r] += 1.0
                        V[b, r] += 1.0
                        W[c, r] += 1.0
        
        # Run ALS to refine
        U, V, W = _als_step(T, U, V, W, rank, time_limit, start_time, max_iter=100)
        
        if _verify_exact(T, U, V, W, tol=1e-8):
            return U, V, W
        
        # Try with half-integer adjustments
        if attempt % 2 == 1:
            U2 = U.copy()
            V2 = V.copy()
            W2 = W.copy()
            U2 += np.random.randn(a_size, rank) * 0.01
            V2 += np.random.randn(b_size, rank) * 0.01
            W2 += np.random.randn(c_size, rank) * 0.01
            
            U2, V2, W2 = _als_step(T, U2, V2, W2, rank, time_limit, start_time, max_iter=100)
            
            if _verify_exact(T, U2, V2, W2, tol=1e-8):
                return U2, V2, W2
    
    return None

def _to_list(mat):
    """Convert numpy array to list, rounding to nearest integer or half-integer if close."""
    result = []
    for row in mat:
        row_list = []
        for val in row:
            nearest_int = round(val)
            nearest_half = round(2 * val) / 2.0
            if abs(val - nearest_int) < 1e-6:
                row_list.append(int(nearest_int))
            elif abs(val - nearest_half) < 1e-6:
                row_list.append(nearest_half)
            else:
                row_list.append(float(val))
        result.append(row_list)
    return result

def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    schoolbook_rank = n * m * p  # 48
    
    time_start = time.time()
    time_limit_total = 135  # Leave 15s margin for 150s limit
    
    best_rank = schoolbook_rank
    best_decomp = None
    
    # Build schoolbook decomposition as fallback
    U_sb = np.zeros((n * m, schoolbook_rank))
    V_sb = np.zeros((m * p, schoolbook_rank))
    W_sb = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    # Lower bound: max(n*p, m*p, n*m) = max(16, 24, 6) = 24
    lower_bound = max(n * p, m * p, n * m)
    
    # For 2x3x8, known results suggest rank around 30-40
    # Try ranks in a smart order: start with likely achievable ranks
    # Focus on ranks 28-42 first, then expand if time permits
    likely_ranks = list(range(30, 43))
    other_ranks = [r for r in range(lower_bound, best_rank) if r not in likely_ranks]
    
    # Sort other ranks by distance from 35 (middle of likely range)
    other_ranks.sort(key=lambda r: abs(r - 35))
    
    ranks_to_try = likely_ranks + other_ranks
    
    for rank in ranks_to_try:
        elapsed = time.time() - time_start
        if elapsed > time_limit_total:
            break
        
        remaining_time = time_limit_total - elapsed
        rank_idx = ranks_to_try.index(rank)
        per_try_time = min(remaining_time / max(1, len(ranks_to_try) - rank_idx), 10.0)
        
        # First try integer search (structured)
        result = _try_rank_integer_search(T, n, m, p, rank, per_try_time * 0.4, seed=rank)
        if result is not None:
            U, V, W = result
            best_rank = rank
            best_decomp = (U, V, W)
            break
        
        # Then try ALS
        result = _try_rank_als(T, n, m, p, rank, per_try_time * 0.6, seed=rank)
        if result is not None:
            U, V, W = result
            best_rank = rank
            best_decomp = (U, V, W)
            break
    
    if best_decomp is not None:
        U, V, W = best_decomp
        return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
    else:
        # Schoolbook fallback
        return {"U": _to_list(U_sb), "V": _to_list(V_sb), "W": _to_list(W_sb)}

# EVOLVE-BLOCK-END
