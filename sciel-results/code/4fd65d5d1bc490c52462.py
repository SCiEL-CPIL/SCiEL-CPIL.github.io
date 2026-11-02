"""Program for 3x3x3 matrix multiplication tensor decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication.
    
    Strategy: 
    1. Try to find rank 23, 24, 25 decompositions via structured numerical search
    2. Fall back to known rank-26 decomposition if search fails
    
    The tensor identity: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    def idx_u(i, j): return i * m + j
    def idx_v(j, k): return j * p + k
    def idx_w(k, i): return k * n + i
    
    # Try ranks from 23 upward
    for R in [23, 24, 25]:
        if time.time() > deadline - 30:
            break
        result = _search_rank(R, n, m, p, deadline)
        if result is not None:
            return result
    
    # Fallback: rank-26 decomposition (Strassen + border)
    return _rank26_fallback(n, m, p)

def _search_rank(R, n, m, p, deadline):
    """
    Try to find a rank-R decomposition via iterative optimization.
    
    Approach: Initialize U, V, W randomly, then iteratively update one factor
    while keeping others fixed (alternating minimization style), but since the
    constraint is exact (not least-squares), we use a different strategy:
    
    For a fixed V and W, the constraint on U is:
      For each (i,j,k): sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    
    This is linear in U. Similarly for V and W. So we can do:
    1. Initialize V, W randomly
    2. Solve for U from linear system
    3. Solve for V from linear system  
    4. Solve for W from linear system
    5. Check if solution is consistent (rank constraint)
    
    Actually, a better approach: use the fact that for a rank-R decomposition,
    we can parameterize and use Newton-type methods. But that's complex.
    
    Simpler: try many random initializations and use coordinate descent.
    """
    n, m, p = 3, 3, 3
    
    # The constraint: for all i,j,k: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    # This is 27 scalar equations.
    
    # Strategy: Fix two factors, solve for the third via least squares, iterate.
    # If we can get all residuals to zero, we have a valid decomposition.
    
    max_iters = 2000
    num_restarts = 50
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Random initialization
        U = np.random.randn(n * m, R)
        V = np.random.randn(m * p, R)
        W = np.random.randn(p * n, R)
        
        # Normalize to avoid scaling issues
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        best_residual = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Compute residual
            U3 = U.reshape(n, m, R)
            V3 = V.reshape(m, p, R)
            W3 = W.reshape(p, n, R)
            T_recon = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
            residual = np.linalg.norm(T_recon - 1.0)
            
            if residual < best_residual:
                best_residual = residual
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if residual < 1e-9:
                break
            
            # Update U: for each (i,j), solve linear system
            # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all k
            # This is: U[i,j,:] @ (V[j,:,r] * W[:,i,r]) = ones(p)
            # Let M[i,j] be a p x R matrix where M[i,j][k,r] = V[j,k,r]*W[k,i,r]
            # Then U[i,j,:] @ M[i,j].T = ones(p)
            # If p <= R, this is underdetermined; use least squares or Moore-Penrose
            
            for i in range(n):
                for j in range(m):
                    # Build p x R matrix
                    M = np.zeros((p, R))
                    for k in range(p):
                        for r in range(R):
                            M[k, r] = V[j * p + k, r] * W[k * n + i, r]
                    # Solve U[i,j,:] @ M.T = ones(p)
                    # i.e., M @ U[i,j,:].T = ones(p)
                    if p <= R:
                        # Underdetermined: use minimum norm solution
                        U[i * m + j, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                    else:
                        # Overdetermined: least squares
                        U[i * m + j, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
            
            # Update V
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, R))
                    for i in range(n):
                        for r in range(R):
                            M[i, r] = U[i * m + j, r] * W[k * n + i, r]
                    V[j * p + k, :] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
            
            # Update W
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, R))
                    for j in range(m):
                        for r in range(R):
                            M[j, r] = U[i * m + j, r] * V[j * p + k, r]
                    W[k * n + i, :] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
        
        # Check if we found a valid solution
        U3 = best_U.reshape(n, m, R)
        V3 = best_V.reshape(m, p, R)
        W3 = best_W.reshape(p, n, R)
        T_recon = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        if np.allclose(T_recon, 1.0, atol=1e-8):
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return None

def _rank26_fallback(n, m, p):
    """Known rank-26 decomposition: Strassen for 2x2 core + 19 border terms."""
    R = 26
    U = np.zeros((n * m, R), dtype=np.int64)
    V = np.zeros((m * p, R), dtype=np.int64)
    W = np.zeros((p * n, R), dtype=np.int64)
    
    def idx_u(i, j): return i * m + j
    def idx_v(j, k): return j * p + k
    def idx_w(k, i): return k * n + i
    
    # Strassen 7 terms for top-left 2x2
    r = 0
    U[idx_u(0, 0), r] = 1; U[idx_u(1, 1), r] = 1
    V[idx_v(0, 0), r] = 1; V[idx_v(1, 1), r] = 1
    W[idx_w(0, 0), r] = 1; W[idx_w(1, 1), r] = 1
    
    r = 1
    U[idx_u(1, 0), r] = 1; U[idx_u(1, 1), r] = 1
    V[idx_v(0, 0), r] = 1
    W[idx_w(0, 1), r] = 1; W[idx_w(1, 1), r] = -1
    
    r = 2
    U[idx_u(0, 0), r] = 1
    V[idx_v(0, 1), r] = 1; V[idx_v(1, 1), r] = -1
    W[idx_w(1, 0), r] = 1; W[idx_w(1, 1), r] = 1
    
    r = 3
    U[idx_u(1, 1), r] = 1
    V[idx_v(1, 0), r] = 1; V[idx_v(0, 0), r] = -1
    W[idx_w(0, 0), r] = 1; W[idx_w(0, 1), r] = 1
    
    r = 4
    U[idx_u(0, 0), r] = 1; U[idx_u(0, 1), r] = 1
    V[idx_v(1, 1), r] = 1
    W[idx_w(0, 0), r] = -1; W[idx_w(1, 0), r] = 1
    
    r = 5
    U[idx_u(1, 0), r] = 1; U[idx_u(0, 0), r] = -1
    V[idx_v(0, 0), r] = 1; V[idx_v(0, 1), r] = 1
    W[idx_w(1, 1), r] = 1
    
    r = 6
    U[idx_u(0, 1), r] = 1; U[idx_u(1, 1), r] = -1
    V[idx_v(1, 0), r] = 1; V[idx_v(1, 1), r] = 1
    W[idx_w(0, 0), r] = 1
    
    # 19 border terms
    border_indices = [
        (2, 0, 0), (2, 0, 1), (2, 0, 2),
        (2, 1, 0), (2, 1, 1), (2, 1, 2),
        (2, 2, 0), (2, 2, 1), (2, 2, 2),
        (0, 2, 0), (0, 2, 1), (0, 2, 2),
        (1, 2, 0), (1, 2, 1), (1, 2, 2),
        (0, 0, 2), (0, 1, 2),
        (1, 0, 2), (1, 1, 2)
    ]
    
    col = 7
    for i, j, k in border_indices:
        U[idx_u(i, j), col] = 1
        V[idx_v(j, k), col] = 1
        W[idx_w(k, i), col] = 1
        col += 1
    
    assert col == R
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
