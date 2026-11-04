"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import random

def _build_tensor(n, m, p):
    """Build the multiplication tensor as a 3D array T[u_idx, v_idx, w_idx]."""
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    return T

def _try_rank_search(n, m, p, rank, max_iters=500, seed=42, deadline=None):
    """Try to find a rank-`rank` decomposition using random search with refinement."""
    rng = np.random.default_rng(seed)
    T = _build_tensor(n, m, p)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Initialize with random values in {-1, 0, 1}
        U = rng.integers(-1, 2, size=(n*m, rank))
        V = rng.integers(-1, 2, size=(m*p, rank))
        W = rng.integers(-1, 2, size=(p*n, rank))
        
        # Compute residual
        T_recon = np.zeros_like(T)
        for r in range(rank):
            T_recon += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        residual = T - T_recon
        
        if np.allclose(residual, 0, atol=1e-9):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
        
        # Try to fix the residual by adjusting individual terms
        for r in range(rank):
            if deadline is not None and time.time() > deadline:
                break
            
            # Compute the current contribution of term r
            term_r = np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            
            # The target for term r is: T - (sum of all other terms)
            other_terms = T_recon - term_r
            target = T - other_terms
            
            # Try a random perturbation
            old_U = U[:,r].copy()
            old_V = V[:,r].copy()
            old_W = W[:,r].copy()
            
            # Perturb
            U[:,r] = rng.integers(-1, 2, size=(n*m,))
            V[:,r] = rng.integers(-1, 2, size=(m*p,))
            W[:,r] = rng.integers(-1, 2, size=(p*n,))
            
            # Compute new residual
            new_term_r = np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            new_T_recon = other_terms + new_term_r
            new_residual = T - new_T_recon
            
            if np.linalg.norm(new_residual) < np.linalg.norm(residual):
                T_recon = new_T_recon
                residual = new_residual
            else:
                # Revert
                U[:,r] = old_U
                V[:,r] = old_V
                W[:,r] = old_W
        
        # Check if we found a valid decomposition
        if np.allclose(residual, 0, atol=1e-9):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    return None

def construct():
    """
    Construct a tensor decomposition for matrix multiplication M_{n,m,p}.
    """
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    deadline = time.time() + 130  # 130s deadline, leave 20s margin
    
    # Try to find a better decomposition via random search
    best_result = None
    
    # Try ranks from 50 to 71
    for rank in range(50, min(72, schoolbook_rank)):
        if time.time() > deadline:
            break
        result = _try_rank_search(n, m, p, rank, max_iters=200, seed=rank*1000, deadline=deadline)
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
    start_time = time.time()
    res = construct()
    n, m, p = DIMENSIONS
    assert len(res["U"]) == n * m
    assert len(res["V"]) == m * p
    assert len(res["W"]) == p * n
    rank = len(res["U"][0]) if n * m > 0 else 0
    assert len(res["V"][0]) == rank
    assert len(res["W"][0]) == rank
    elapsed = time.time() - start_time
    print(f"Constructed in {elapsed:.4f}s")
