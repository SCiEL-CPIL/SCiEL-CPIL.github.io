"""Matrix multiplication 2x5x5 rank-minimal exact tensor decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 135  # Safety margin before 150s limit
    
    # Build the target tensor
    target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def verify(U, V, W, rank):
        """Check if the decomposition is exact using einsum."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        # Compute T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # Using einsum: 'ar,br,cr->abc'
        try:
            computed = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr, optimize=True)
        except Exception:
            # Fallback to manual computation
            for a in range(n * m):
                for b in range(m * p):
                    for c in range(p * n):
                        val = np.dot(U_arr[a, :], V_arr[b, :] * W_arr[c, :])
                        if abs(val - target[a, b, c]) > 1e-9:
                            return False
            return True
        return np.allclose(computed, target, atol=1e-9)

    def try_rank(rank, max_iters=3000, seed=None):
        """Try to find a rank-r decomposition via random search with structured initialization."""
        if seed is not None:
            random.seed(seed)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                return None
            
            # Use values from {-2, -1, -0.5, 0, 0.5, 1, 2}
            vals = [-2, -1, -0.5, 0, 0.5, 1, 2]
            U = [[random.choice(vals) for _ in range(rank)] for _ in range(n * m)]
            V = [[random.choice(vals) for _ in range(rank)] for _ in range(m * p)]
            W = [[random.choice(vals) for _ in range(rank)] for _ in range(p * n)]
            
            if verify(U, V, W, rank):
                return {"U": U, "V": V, "W": W}
        
        return None

    # Try structured approach: exploit the block structure
    # For 2×5×5, think of it as two rows of 5×5 matrix multiplication
    # C[0,k] = sum_j A[0,j]*B[j,k]
    # C[1,k] = sum_j A[1,j]*B[j,k]
    
    # Try ranks from lower bound upward
    # Lower bound: max(n*m, m*p, n*p) = max(10, 25, 10) = 25
    # But we know schoolbook is 50, so try 25-50
    
    # First, try a few specific structured decompositions
    # Laderman-style for 2×2×2 blocks, then extend
    
    # Try random search for various ranks
    for rank in range(25, 51):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(rank, max_iters=2000)
        if result is not None:
            return result
    
    # If no improvement found, return schoolbook
    return schoolbook()

# EVOLVE-BLOCK-END
