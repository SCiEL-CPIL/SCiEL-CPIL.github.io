"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Check if the tensor decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    max_err = 0.0
    for i in range(n * m):
        i_A = i // m
        j_A = i % m
        for j in range(m * p):
            j_B = j // p
            k_B = j % p
            if j_A != j_B:
                continue
            for k in range(p * n):
                k_C = k // n
                i_C = k % n
                expected = 1.0 if (i_A == i_C and k_B == k_C) else 0.0
                actual = 0.0
                for c in range(rank):
                    actual += U[i, c] * V[j, c] * W[k, c]
                err = abs(actual - expected)
                if err > max_err:
                    max_err = err
    
    return max_err < 1e-9

def _compute_error(U, V, W, n, m, p):
    """Compute the maximum error of the decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    max_err = 0.0
    for i in range(n * m):
        i_A = i // m
        j_A = i % m
        for j in range(m * p):
            j_B = j // p
            k_B = j % p
            if j_A != j_B:
                continue
            for k in range(p * n):
                k_C = k // n
                i_C = k % n
                expected = 1.0 if (i_A == i_C and k_B == k_C) else 0.0
                actual = 0.0
                for c in range(rank):
                    actual += U[i, c] * V[j, c] * W[k, c]
                err = abs(actual - expected)
                if err > max_err:
                    max_err = err
    
    return max_err

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return U, V, W

def _try_rank(rank, n, m, p, time_limit=10, seed=42):
    """Try to find a decomposition of given rank using random initialization and refinement."""
    start_time = time.time()
    np.random.seed(seed)
    
    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    # Precompute index mappings for speed
    i_map = [(i // m, i % m) for i in range(n * m)]
    j_map = [(j // p, j % p) for j in range(m * p)]
    k_map = [(k // n, k % n) for k in range(p * n)]
    
    # Build compatibility lists
    compat = {}
    for i in range(n * m):
        i_A, j_A = i_map[i]
        for j in range(m * p):
            j_B, k_B = j_map[j]
            if j_A == j_B:
                for k in range(p * n):
                    k_C, i_C = k_map[k]
                    if i_A == i_C and k_B == k_C:
                        if i not in compat:
                            compat[i] = []
                        compat[i].append((j, k))
    
    for attempt in range(20):
        if time.time() - start_time > time_limit:
            break
        
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        # Iterative refinement with alternating updates
        for iteration in range(100):
            if time.time() - start_time > time_limit:
                break
            
            # Update U
            for i in range(n * m):
                if i not in compat:
                    continue
                vw_sum = np.zeros(rank)
                vw_sq_sum = np.zeros(rank)
                for j, k in compat[i]:
                    vw = V[j] * W[k]
                    vw_sum += vw
                    vw_sq_sum += vw * vw
                denom = np.sum(vw_sq_sum)
                if denom > 1e-12:
                    U[i] = vw_sum / denom
            
            # Update V
            for j in range(m * p):
                j_B, k_B = j_map[j]
                uw_sum = np.zeros(rank)
                uw_sq_sum = np.zeros(rank)
                for i in range(n * m):
                    i_A, j_A = i_map[i]
                    if j_A != j_B:
                        continue
                    for k in range(p * n):
                        k_C, i_C = k_map[k]
                        if i_A == i_C and k_B == k_C:
                            uw = U[i] * W[k]
                            uw_sum += uw
                            uw_sq_sum += uw * uw
                denom = np.sum(uw_sq_sum)
                if denom > 1e-12:
                    V[j] = uw_sum / denom
            
            # Update W
            for k in range(p * n):
                k_C, i_C = k_map[k]
                uv_sum = np.zeros(rank)
                uv_sq_sum = np.zeros(rank)
                for i in range(n * m):
                    i_A, j_A = i_map[i]
                    if i_A != i_C:
                        continue
                    for j in range(m * p):
                        j_B, k_B = j_map[j]
                        if j_B == j_A and k_B == k_C:
                            uv = U[i] * V[j]
                            uv_sum += uv
                            uv_sq_sum += uv * uv
                denom = np.sum(uv_sq_sum)
                if denom > 1e-12:
                    W[k] = uv_sum / denom
            
            # Compute error
            err = _compute_error(U, V, W, n, m, p)
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if best_err < 1e-9:
                break
    
    if best_err < 1e-9:
        return best_U, best_V, best_W, best_err
    return None, None, None, best_err

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Try to find lower-rank decompositions
    # For 2x4x6, the theoretical minimum is at least max(n*m, m*p, p*n) = max(8, 24, 12) = 24
    # But Strassen-like methods might achieve lower rank
    # Try ranks from 24 up to schoolbook rank
    for rank in range(24, n * m * p + 1):
        if time.time() > deadline:
            break
        
        time_remaining = deadline - time.time()
        if time_remaining < 5:
            break
        
        time_limit = min(time_remaining * 0.5, 15)
        
        U, V, W, err = _try_rank(rank, n, m, p, time_limit=time_limit)
        
        if U is not None and err < 1e-9:
            best_rank = rank
            best_U = U
            best_V = V
            best_W = W
            break
    
    # Convert to lists for return
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
