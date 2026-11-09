import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x4 by 4x5 matrix.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 60
    
    def validate_decomposition(U, V, W, tol=1e-9):
        """Check if the decomposition is valid"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(U.shape[1]):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def to_integer_or_half_integer(arr, tol=1e-6):
        """Convert float array to integer or half-integer if possible"""
        result = np.zeros_like(arr)
        for idx in np.ndindex(arr.shape):
            val = arr[idx]
            # Try to find nearest integer or half-integer
            candidates = [round(val * 2) / 2.0]
            for c in candidates:
                if abs(val - c) < tol:
                    result[idx] = c
                    break
            else:
                # If no match, keep the value (shouldn't happen for valid decompositions)
                result[idx] = val
        return result
    
    # Try to find a decomposition with rank less than schoolbook
    # Use a randomized local search approach
    
    best_rank = schoolbook_rank
    best_U = None
    best_V = None
    best_W = None
    
    # Initialize with schoolbook
    U0 = np.zeros((n * m, schoolbook_rank), dtype=float)
    V0 = np.zeros((m * p, schoolbook_rank), dtype=float)
    W0 = np.zeros((p * n, schoolbook_rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1.0
                V0[j * p + k, col] = 1.0
                W0[k * n + i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    
    # Try to reduce rank by removing zero columns or finding dependencies
    # First, check if any column is all zeros (shouldn't be in schoolbook)
    
    # Try randomized perturbation and local optimization
    np.random.seed(42)
    
    for target_rank in range(50, 19, -1):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
            
            # Initialize with random values
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # Scale to make the sum of products close to 1
            # For each (i,j,k), we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # This is hard to satisfy directly, so use a heuristic
            
            # Try a structured initialization based on schoolbook
            # Pick target_rank random (i,j,k) triples and use them
            indices = list(range(n * m * p))
            np.random.shuffle(indices)
            selected = indices[:target_rank]
            
            U = np.zeros((n * m, target_rank), dtype=float)
            V = np.zeros((m * p, target_rank), dtype=float)
            W = np.zeros((p * n, target_rank), dtype=float)
            
            for c, idx in enumerate(selected):
                i = idx // (m * p)
                j = (idx // p) % m
                k = idx % p
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
            
            # Check if this partial decomposition is valid
            if validate_decomposition(U, V, W):
                best_rank = target_rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break
        
        if best_rank == target_rank:
            break
    
    # Convert to integer/half-integer if possible
    if best_U is not None:
        best_U = to_integer_or_half_integer(best_U)
        best_V = to_integer_or_half_integer(best_V)
        best_W = to_integer_or_half_integer(best_W)
    
    # Final validation
    if best_U is not None and validate_decomposition(best_U, best_V, best_W):
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        # Fall back to schoolbook
        return {
            "U": U0.tolist(),
            "V": V0.tolist(),
            "W": W0.tolist()
        }
# EVOLVE-BLOCK-END
