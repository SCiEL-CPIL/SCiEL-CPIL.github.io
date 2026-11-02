"""Matrix multiplication 2x3 x 3x4 rank-minimal decomposition search."""
import numpy as np
from itertools import product
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Tensor: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We need T[i,j,k] = 1 for all i,j,k (the multiplication tensor)
    
    # Start with schoolbook as fallback
    rank_school = n * m * p
    U_fallback = np.zeros((n * m, rank_school), dtype=float)
    V_fallback = np.zeros((m * p, rank_school), dtype=float)
    W_fallback = np.zeros((p * n, rank_school), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fallback[i * m + j, col] = 1
                V_fallback[j * p + k, col] = 1
                W_fallback[k * n + i, col] = 1
                col += 1
    
    deadline = time.time() + 140  # Safety margin
    
    def verify(U, V, W):
        """Check if U,V,W exactly decompose the multiplication tensor."""
        n, m, p = DIMENSIONS
        # T[i,j,k] should equal 1 for all i,j,k
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def snap_to_half_int(arr):
        """Snap values to nearest half-integer."""
        return np.round(arr * 2) / 2
    
    # Search strategy: try ranks from low to high
    # Known lower bounds: for (2,3,4), the rank is at least max(n*m, m*p, p*n) = 12
    # But more refined bounds exist. Let's try from 12 upward.
    
    best_result = None
    
    # Try structured decompositions based on algebraic identities
    # For matrix multiplication, there are known decompositions using
    # the fact that (A+B)(C+D) = AC + AD + BC + BD
    
    # Strategy 1: Try to find a decomposition of rank 16, 18, 20, 22, 24
    # Using randomized search with structured initialization
    
    for target_rank in range(12, 25):
        if time.time() > deadline - 20:
            break
        
        # Try multiple random initializations
        n_attempts = 20 if target_rank <= 18 else 10
        for attempt in range(n_attempts):
            if time.time() > deadline - 15:
                break
            
            # Initialize with small integer/half-integer values
            entries = [0, 0.5, 1, 1.5, 2, -0.5, -1, -1.5, -2]
            
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Random sparse initialization
            for c in range(target_rank):
                # Each column of U, V, W has a few non-zero entries
                nz_u = np.random.choice(n * m, size=np.random.randint(1, 4), replace=False)
                nz_v = np.random.choice(m * p, size=np.random.randint(1, 4), replace=False)
                nz_w = np.random.choice(p * n, size=np.random.randint(1, 4), replace=False)
                
                U[nz_u, c] = np.random.choice(entries, size=len(nz_u))
                V[nz_v, c] = np.random.choice(entries, size=len(nz_v))
                W[nz_w, c] = np.random.choice(entries, size=len(nz_w))
            
            # Verify
            if verify(U, V, W):
                # Convert to list format
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                return best_result
        
        # If we found nothing at this rank, continue to next
        if time.time() > deadline - 10:
            break
    
    # If no better decomposition found, return schoolbook
    return {
        "U": U_fallback.tolist(),
        "V": V_fallback.tolist(),
        "W": W_fallback.tolist()
    }
