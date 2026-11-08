"""Matrix multiplication 2,4,4 - rank-minimal tensor decomposition search."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 16
    pn = p * n  # 8
    
    # Target tensor: T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # We need U (nm x r), V (mp x r), W (pn x r) such that
    # for all (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Start with schoolbook rank 32
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Initialize with schoolbook
    U_sb = np.zeros((nm, best_rank))
    V_sb = np.zeros((mp, best_rank))
    W_sb = np.zeros((pn, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    # Try to find lower rank decompositions
    # Strategy: random initialization + local search / coordinate descent
    
    def check_decomposition(U, V, W, tol=1e-6):
        """Check if U, V, W form a valid decomposition."""
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def try_rank(r, max_iters=5000):
        """Try to find a valid decomposition of rank r."""
        if r >= best_rank:
            return None, None, None
        
        U = np.random.randint(0, 2, size=(nm, r)).astype(float)
        V = np.random.randint(0, 2, size=(mp, r)).astype(float)
        W = np.random.randint(0, 2, size=(pn, r)).astype(float)
        
        # Local search: try to adjust entries to satisfy constraints
        # This is a hard problem, so use a heuristic approach
        
        # Alternative: use the fact that we can permute and scale
        # Let's try a more structured approach
        
        # For now, just return the initial guess if it happens to work
        if check_decomposition(U, V, W):
            return U, V, W
        return None, None, W
    
    # Try known constructions for small matrix multiplication
    # For 2x4 * 4x4, we can potentially use block structure
    
    # Attempt: exploit that n=2 is small
    # A is 2x4, B is 4x4, result is 2x4
    # Each output element C[i,k] = sum_j A[i,j]*B[j,k]
    # There are 8 output elements, each is a dot product of length 4
    
    # Standard: 8*4 = 32 multiplications
    # Can we share? If we precompute some combinations...
    
    # Let me try a different approach: use the Strassen-like idea
    # But for rectangular matrices it's more complex
    
    # Try random search with half-integer values
    random.seed(42)
    for r in range(16, best_rank):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random initializations
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            # Use values from {-1, 0, 1, 1/2}
            U = np.random.choice([-1, 0, 0.5, 1], size=(nm, r))
            V = np.random.choice([-1, 0, 0.5, 1], size=(mp, r))
            W = np.random.choice([-1, 0, 0.5, 1], size=(pn, r))
            
            if check_decomposition(U, V, W):
                best_rank = r
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                break
    
    # Convert to lists for return
    def to_list(arr):
        return arr.tolist()
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
