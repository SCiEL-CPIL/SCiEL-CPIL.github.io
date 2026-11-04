"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def _build_tensor(n, m, p):
    """Build the multiplication tensor as a 3D array T[u_idx, v_idx, w_idx]."""
    T = np.zeros((n*m, m*p, p*n), dtype=np.int32)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_r U[:,r] outer V[:,r] outer W[:,r] equals the MM tensor."""
    T_recon = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for r in range(len(U[0])):
        u_col = np.array([U[i][r] for i in range(n*m)], dtype=np.float64)
        v_col = np.array([V[i][r] for i in range(m*p)], dtype=np.float64)
        w_col = np.array([W[i][r] for i in range(p*n)], dtype=np.float64)
        T_recon += np.einsum('a,b,c->abc', u_col, v_col, w_col)
    
    T_true = _build_tensor(n, m, p).astype(np.float64)
    return np.allclose(T_recon, T_true, atol=tol)

def _try_rank(n, m, p, rank, max_iters=5000, seed=42, deadline=None):
    """Try to find a rank-`rank` decomposition with integer entries in {-1,0,1}."""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    
    best = None
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Random integer entries in {-1, 0, 1}
        U = np.zeros((n*m, rank), dtype=np.int32)
        V = np.zeros((m*p, rank), dtype=np.int32)
        W = np.zeros((p*n, rank), dtype=np.int32)
        
        # Fill with random -1, 0, 1
        U = np_rng.integers(-1, 2, size=(n*m, rank))
        V = np_rng.integers(-1, 2, size=(m*p, rank))
        W = np_rng.integers(-1, 2, size=(p*n, rank))
        
        # Quick check: compute T_recon and compare
        T_recon = np.zeros((n*m, m*p, p*n), dtype=np.int32)
        for r in range(rank):
            T_recon += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        
        T_true = _build_tensor(n, m, p)
        if np.array_equal(T_recon, T_true):
            # Convert to list format
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    """
    Construct a tensor decomposition for matrix multiplication M_{n,m,p}.
    """
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    deadline = time.time() + 130  # 130s deadline, leave 20s margin
    
    # First, try to find a better decomposition via random search
    # Start from lower ranks and go up
    
    # For 3x3x8, known bounds: schoolbook is 72.
    # Let's try ranks from 40 to 71
    best_result = None
    
    # Try a few random searches at various ranks
    for rank in range(40, min(72, schoolbook_rank)):
        if time.time() > deadline:
            break
        result = _try_rank(n, m, p, rank, max_iters=200, seed=rank*1000, deadline=deadline)
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        return best_result
    
    # Fall back to schoolbook
    rank = schoolbook_rank
    U = [[0.0 for _ in range(rank)] for _ in range(n * m)]
    V = [[0.0 for _ in range(rank)] for _ in range(m * p)]
    W = [[0.0 for _ in range(rank)] for _ in range(p * n)]
    
    col_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][col_idx] = 1.0
                V[v_idx][col_idx] = 1.0
                W[w_idx][col_idx] = 1.0
                col_idx += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    # Simple self-check logic (not called by evaluator)
    start_time = time.time()
    res = construct()
    n, m, p = DIMENSIONS
    # Basic shape check
    assert len(res["U"]) == n * m
    assert len(res["V"]) == m * p
    assert len(res["W"]) == p * n
    rank = len(res["U"][0]) if n * m > 0 else 0
    assert len(res["V"][0]) == rank
    assert len(res["W"][0]) == rank
    
    # Time check
    elapsed = time.time() - start_time
    print(f"Constructed in {elapsed:.4}s")
