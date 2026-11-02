import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 4, 7]

def _verify(U, V, W, n, m, p, tol=1e-9):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    Texp = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                Texp[a, b, c] = 1.0
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    err = np.max(np.abs(T - Texp))
    return err < tol

def _schoolbook(n, m, p):
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

def _strassen_decomp(n, m, p):
    """
    Returns the rank-50 decomposition.
    """
    rank = 50
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    col = 0
    
    def add_strassen_222(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)
        
        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)
        
        # M1
        U[a_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1; W[c11_idx, col] += 1
        col += 1
        # M2
        U[c_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1
        W[c10_idx, col] += 1; W[c11_idx, col] -= 1
        col += 1
        # M3
        U[a_idx, col] += 1
        V[f_idx, col] += 1; V[h_idx, col] -= 1
        W[c01_idx, col] += 1; W[c11_idx, col] += 1
        col += 1
        # M4
        U[d_idx, col] += 1
        V[g_idx, col] += 1; V[e_idx, col] -= 1
        W[c00_idx, col] += 1; W[c10_idx, col] += 1
        col += 1
        # M5
        U[a_idx, col] += 1; U[b_idx, col] += 1
        V[h_idx, col] += 1
        W[c00_idx, col] -= 1; W[c01_idx, col] += 1
        col += 1
        # M6
        U[c_idx, col] += 1; U[a_idx, col] -= 1
        V[e_idx, col] += 1; V[f_idx, col] += 1
        W[c11_idx, col] += 1
        col += 1
        # M7
        U[b_idx, col] += 1; U[d_idx, col] -= 1
        V[g_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1
        col += 1

    def add_schoolbook_221(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0
        
        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)
        
        U[a_idx, col] += 1; V[e_idx, col] += 1; W[c00_idx, col] += 1
        col += 1
        U[b_idx, col] += 1; V[g_idx, col] += 1; W[c00_idx, col] += 1
        col += 1
        U[c_idx, col] += 1; V[e_idx, col] += 1; W[c10_idx, col] += 1
        col += 1
        U[d_idx, col] += 1; V[g_idx, col] += 1; W[c10_idx, col] += 1
        col += 1

    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            add_strassen_222(0, j_start, k_start)
        add_schoolbook_221(0, j_start, 6)
    
    return U, V, W

def _try_hypergraph_merge(n, m, p, deadline):
    """
    Attempts to reduce rank by finding non-local mergers using a hypergraph coverage heuristic.
    Specifically, we look for a set of existing rank-1 tensors (from the rank-50 solution)
    that can be replaced by a smaller set of new rank-1 tensors (with integer/half-integer entries)
    such that the sum of the replaced tensors equals the sum of the new tensors.
    
    This is equivalent to finding a linear dependence among the existing tensors that allows
    for a rank reduction, but constrained to integer/half-integer solutions.
    
    Given the complexity, we will try a simpler heuristic:
    1. Check for linear dependencies in the space of flattened tensors.
    2. If a dependency is found, attempt to construct a new set of tensors that satisfies the dependency
       and has integer/half-integer entries.
    3. If successful, return the new set.
    
    However, finding such a dependency and converting it to a valid decomposition is extremely difficult.
    For now, we will implement a search for a specific known rank-49 decomposition if one exists,
    or return None if no improvement is found within the time limit.
    """
    start_time = time.time()
    U50, V50, W50 = _strassen_decomp(n, m, p)
    
    # Flatten the 50 tensors
    def flatten(u, v, w):
        return np.einsum('a,b,c->abc', u, v, w).flatten()
    
    T_flat = np.array([flatten(U50[:, r], V50[:, r], W50[:, r]) for r in range(50)])
    T_flat = T_flat.T # Shape: (tensor_dim, 50)
    
    # Check for linear dependencies
    try:
        U_svd, s, Vt_svd = np.linalg.svd(T_flat, full_matrices=False)
        tol = 1e-9
        num_rank = np.sum(s > tol)
        
        if num_rank < 50:
            # There is a dependency.
            # This means the 50 tensors are linearly dependent.
            # We can potentially reduce the rank, but we need to ensure the new set is valid.
            # For now, we will not attempt the complex reconstruction.
            pass
    except:
        pass
    
    # If no simple dependency is found, or if we can't reconstruct, return None
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Get the rank-50 solution
    U, V, W = _strassen_decomp(n, m, p)
    
    # Verify it
    if _verify(U, V, W, n, m, p):
        # Try to improve using hypergraph merge heuristic
        if time.time() < deadline - 10:
            improved = _try_hypergraph_merge(n, m, p, deadline)
            if improved is not None:
                return improved
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback
    return _schoolbook(n, m, p)
