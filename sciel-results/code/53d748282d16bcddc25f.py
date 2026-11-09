"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm = n * m  # 12
    mp = m * p  # 32
    pn = p * n  # 24
    rank_schoolbook = n * m * p  # 96

    def schoolbook():
        U = np.zeros((nm, rank_schoolbook))
        V = np.zeros((mp, rank_schoolbook))
        W = np.zeros((pn, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W

    def verify(U, V, W):
        """Check if U, V, W decompose the matrix multiplication tensor."""
        r = U.shape[1]
        if V.shape[1] != r or W.shape[1] != r:
            return False
        # T[(i,j),(j,k),(k,i)] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # Should equal 1 if indices match, 0 otherwise
        # Actually for matrix mult: C[i,k] = sum_j A[i,j]*B[j,k]
        # So T[(i,j),(j,k),(i,k)] = 1
        # Let's check: for each (i,j,k), the contribution should be 1 only when 
        # we're computing C[i,k] from A[i,j] and B[j,k]
        # So sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should = 1 for all valid (i,j,k)
        # And for invalid combinations it should be 0, but there are no invalid ones in this indexing.
        # Actually every (i,j,k) triple is valid.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True

    def try_rank(r, max_iter=5000, seed=42):
        """Try to find a rank-r decomposition using iterative refinement."""
        if r < max(nm, mp, pn):
            return None  # Can't go below this
        
        rng = np.random.RandomState(seed)
        
        # Initialize with structured pattern
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        
        # Use a structured initialization based on grouping
        # Group by i, j, k patterns
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if idx < r:
                        U[i*m + j, idx] = 1
                        V[j*p + k, idx] = 1
                        W[k*n + i, idx] = 1
                        idx += 1
        
        if idx < r:
            # Fill remaining with small random values
            for c in range(idx, r):
                U[:, c] = rng.randn(nm) * 0.1
                V[:, c] = rng.randn(mp) * 0.1
                W[:, c] = rng.randn(pn) * 0.1
        
        # Iterative refinement: alternate solving for U, V, W
        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Solve for U: for each row i*m+j, U[i*m+j,:] should satisfy
            # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
            # This is a linear system in U[i*m+j,:]
            for i in range(n):
                for j in range(m):
                    # Build system: for each k, sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1
                    # Let b_k = V[j*p+k,:] * W[k*n+i,:] (element-wise product)
                    # Then sum_c U[i*m+j,c] * b_k[c] = 1 for all k
                    # This is: B @ U[i*m+j,:] = 1 where B[k,c] = V[j*p+k,c]*W[k*n+i,c]
                    B = np.zeros((p, r))
                    for k in range(p):
                        B[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    # Solve B @ x = ones(p)
                    try:
                        x, residuals, rank_b, sv = np.linalg.lstsq(B, np.ones(p), rcond=None)
                        U[i*m+j, :] = x
                    except:
                        pass
            
            # Solve for V
            for j in range(m):
                for k in range(p):
                    B = np.zeros((n, r))
                    for i in range(n):
                        B[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        x, residuals, rank_b, sv = np.linalg.lstsq(B, np.ones(n), rcond=None)
                        V[j*p+k, :] = x
                    except:
                        pass
            
            # Solve for W
            for i in range(n):
                for k in range(p):
                    B = np.zeros((m, r))
                    for j in range(m):
                        B[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        x, residuals, rank_b, sv = np.linalg.lstsq(B, np.ones(m), rcond=None)
                        W[k*n+i, :] = x
                    except:
                        pass
            
            # Check convergence
            if verify(U, V, W):
                # Try to snap to integer/half-integer values
                U_snapped = np.round(U * 2) / 2
                V_snapped = np.round(V * 2) / 2
                W_snapped = np.round(W * 2) / 2
                if verify(U_snapped, V_snapped, W_snapped):
                    return U_snapped, V_snapped, W_snapped
                else:
                    return U, V, W
        
        return None

    # Try different ranks from low to high
    best_result = None
    
    # Lower bound considerations
    # Rank must be at least max(nm, mp, pn) = max(12, 32, 24) = 32
    # Also at least sqrt(n*m*p) ≈ 9.8, so >= 10
    # But more tightly, for matrix multiplication, rank >= max(n, m, p) and other bounds
    
    # Try ranks from 32 up to schoolbook
    for r in range(32, rank_schoolbook + 1):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(r, max_iter=2000, seed=r)
        if result is not None:
            U, V, W = result
            if verify(U, V, W):
                best_result = (r, U, V, W)
                break  # Found the lowest rank that works
    
    if best_result is not None:
        r, U, V, W = best_result
        # Convert to lists
        return {
            "U": U.tolist(),
            "V": V.tolist(), 
            "W": W.tolist()
        }
    
    # Fallback to schoolbook
    U, V, W = schoolbook()
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
