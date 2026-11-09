"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]
import time
import numpy as np
from fractions import Fraction

def _schoolbook_3x3x8():
    """Returns schoolbook decomposition for 3x3x8 with rank 72."""
    n, m, p = 3, 3, 8
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
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tolerance=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Build the tensor from the decomposition
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # The matrix multiplication tensor is: T[i,j,k] = 1 if this is a valid term, else 0
    # Actually, the tensor for matrix multiplication C = A*B is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] = 1 (it's a rank-1 tensor for each (i,j,k) triple)
    # Wait, no. The tensor is: sum_{i,j,k} A[i,j] * B[j,k] * C[i,k]
    # For the multiplication to be correct, we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
    # But that's not right either. Let me think again.
    
    # The matrix multiplication tensor M satisfies:
    # M[i,j,k] = 1 for all valid (i,j,k) where i in [0,n), j in [0,m), k in [0,p)
    # And the decomposition is: M = sum_r U_r ⊗ V_r ⊗ W_r
    # where (U_r ⊗ V_r ⊗ W_r)[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # So we need: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                if abs(val - 1.0) > tolerance:
                    return False
    return True

def _try_rank_r_decomposition(n, m, p, r, time_budget=10.0):
    """Try to find a rank-r decomposition using random search."""
    start_time = time.time()
    np.random.seed(42)
    
    # Coefficients to try: -2, -1, -0.5, 0, 0.5, 1, 2
    coeffs = [-2, -1, -0.5, 0, 0.5, 1, 2]
    
    U = np.zeros((n * m, r), dtype=float)
    V = np.zeros((m * p, r), dtype=float)
    W = np.zeros((p * n, r), dtype=float)
    
    # Initialize with random sparse patterns
    for attempt in range(1000):
        if time.time() - start_time > time_budget:
            break
            
        # Reset
        U[:] = 0
        V[:] = 0
        W[:] = 0
        
        # Each column gets a random sparse pattern
        for col in range(r):
            # Choose random non-zero positions
            n_nonzero_U = np.random.randint(1, 4)
            n_nonzero_V = np.random.randint(1, 4)
            n_nonzero_W = np.random.randint(1, 4)
            
            idx_U = np.random.choice(n * m, size=n_nonzero_U, replace=False)
            idx_V = np.random.choice(m * p, size=n_nonzero_V, replace=False)
            idx_W = np.random.choice(p * n, size=n_nonzero_W, replace=False)
            
            for idx in idx_U:
                U[idx, col] = np.random.choice(coeffs)
            for idx in idx_V:
                V[idx, col] = np.random.choice(coeffs)
            for idx in idx_W:
                W[idx, col] = np.random.choice(coeffs)
        
        # Verify
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for rr in range(r):
                        val += U[i * m + j, rr] * V[j * p + k, rr] * W[k * n + i, rr]
                    if abs(val - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def construct():
    """
    Construct the decomposition for 3x3x8 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Start with schoolbook (rank 72) as fallback
    best_U, best_V, best_W = _schoolbook_3x3x8()
    best_rank = 72
    
    # Try to find better decompositions
    # Strategy: Try ranks from 50 to 71, but this is computationally expensive
    # Instead, let's try a smarter approach: block decomposition
    
    # Block approach: split p=8 into blocks
    # Try 3+3+2: need rank for 3x3x3 and 3x3x2
    # 3x3x3 schoolbook: 27, 3x3x2 schoolbook: 18, total: 72 (no gain)
    # If we can find rank 23 for 3x3x3: 23+23+18 = 64
    
    # Let's try to construct a rank-23 decomposition for 3x3x3
    # Using a known structure from Winograd-like methods
    
    # Actually, let me try a different approach: use the fact that we can
    # combine blocks more cleverly.
    
    # For now, return schoolbook if we can't find better quickly
    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Try random search for rank 64
    result = _try_rank_r_decomposition(n, m, p, 64, time_budget=10.0)
    if result is not None:
        U, V, W = result
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Try rank 65
    result = _try_rank_r_decomposition(n, m, p, 65, time_budget=10.0)
    if result is not None:
        U, V, W = result
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Try rank 66
    result = _try_rank_r_decomposition(n, m, p, 66, time_budget=10.0)
    if result is not None:
        U, V, W = result
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # Return schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
