"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Check if the tensor decomposition is exact using vectorized operations."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
    T = np.einsum('ic,jc,kc->ijk', U, V, W)
    
    # Build expected tensor
    i_idx = np.arange(n * m)
    j_idx = np.arange(m * p)
    k_idx = np.arange(p * n)
    
    i_A = i_idx // m
    j_A = i_idx % m
    j_B = j_idx // p
    k_B = j_idx % p
    k_C = k_idx // n
    i_C = k_idx % n
    
    cond1 = (j_A[:, None] == j_B[None, :])
    cond2 = (i_A[:, None] == i_C[None, :])
    cond3 = (k_B[:, None] == k_C[None, :])
    
    expected = cond1[:, :, None] * cond2[:, None, :] * cond3[None, :, :]
    expected = expected.astype(float)
    
    max_err = np.max(np.abs(T - expected))
    return max_err < 1e-9

def _try_rank(rank, n, m, p, time_limit=10):
    """Try to find a decomposition of given rank using random initialization and refinement."""
    start_time = time.time()
    np.random.seed(42)
    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    for attempt in range(5):
        if time.time() - start_time > time_limit:
            break
        
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        for iteration in range(50):
            if time.time() - start_time > time_limit:
                break
            
            # Update U
            for i in range(n * m):
                i_A = i // m
                j_A = i % m
                for j in range(m * p):
                    j_B = j // p
                    k_B = j % p
                    if j_A == j_B:
                        for k in range(p * n):
                            k_C = k // n
                            i_C = k % n
                            if i_A == i_C and k_B == k_C:
                                vw = V[j] * W[k]
                                denom = np.dot(vw, vw)
                                if denom > 1e-12:
                                    U[i] = 1.0 * vw / denom
            
            # Update V
            for j in range(m * p):
                j_B = j // p
                k_B = j % p
                for i in range(n * m):
                    i_A = i // m
                    j_A = i % m
                    if j_A == j_B:
                        for k in range(p * n):
                            k_C = k // n
                            i_C = k % n
                            if i_A == i_C and k_B == k_C:
                                uw = U[i] * W[k]
                                denom = np.dot(uw, uw)
                                if denom > 1e-12:
                                    V[j] = 1.0 * uw / denom
            
            # Update W
            for k in range(p * n):
                k_C = k // n
                i_C = k % n
                for i in range(n * m):
                    i_A = i // m
                    j_A = i % m
                    for j in range(m * p):
                        j_B = j // p
                        k_B = j % p
                        if i_A == i_C and j_A == j_B and k_B == k_C:
                            uv = U[i] * V[j]
                            denom = np.dot(uv, uv)
                            if denom > 1e-12:
                                W[k] = 1.0 * uv / denom
            
            # Check error
            err = 0.0
            for i in range(n * m):
                i_A = i // m
                j_A = i % m
                for j in range(m * p):
                    j_B = j // p
                    k_B = j % p
                    for k in range(p * n):
                        k_C = k // n
                        i_C = k % n
                        expected = 1.0 if (j_A == j_B and i_A == i_C and k_B == k_C) else 0.0
                        actual = np.dot(U[i], V[j] * W[k])
                        err = max(err, abs(actual - expected))
            
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if best_err < 1e-9:
                break
    
    if best_err < 1e-9 and best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return None

def _strassen_like_2x4x6():
    """
    Attempt a Strassen-like decomposition for 2x4 * 4x6.
    
    Strategy: Split the 2x4 matrix into two 2x2 blocks, and the 4x6 matrix into two 4x3 blocks.
    We can use Strassen's 7-multiplication algorithm for the 2x2 * 2x2 part, but here
    we need 2x4 * 4x6. 
    
    Actually, let's think differently. The standard rank for 2x4*4x6 is 48.
    Can we do better?
    
    One approach: Use the fact that we can multiply 2x4 * 4x6 by splitting into:
    A = [A1 A2] where A1, A2 are 2x2
    B = [B1; B2] where B1, B2 are 2x6
    
    C = A1*B1 + A2*B2
    
    If we use Strassen for 2x2*2x6? No, Strassen is for square.
    
    Let's try a different decomposition. The tensor rank of <2,4,6> is known to be 48 in general,
    but there might be special structure.
    
    Actually, for this specific problem, the known minimum rank is 48 (schoolbook).
    Let's just return schoolbook to be safe, but try a quick verification.
    """
    pass

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition."""
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
    
    # Try to find a lower rank decomposition if time permits
    # The schoolbook rank is 48. Let's see if we can find something lower.
    # Known bounds suggest rank is at least 48 for this size, but let's try.
    
    # Try ranks from 47 down to 30, with a time budget
    best_result = None
    time_budget = 100  # Leave 50s margin for safety
    
    for rank in range(47, 29, -1):
        if time.time() - start_time > time_budget:
            break
        
        result = _try_rank(rank, n, m, p, time_limit=10)
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        return best_result
    
    # Fall back to schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
