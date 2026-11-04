"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify that the tensor decomposition is correct."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T_computed = np.zeros((n, m, p))
    for c in range(rank):
        u_c = U_arr[:, c].reshape(n, m)
        v_c = V_arr[:, c].reshape(m, p)
        w_c = W_arr[:, c].reshape(p, n)
        T_computed += u_c[:, :, None] * v_c[:, None, :] * w_c[None, :, :]
    
    T_expected = np.ones((n, m, p))
    return np.allclose(T_computed, T_expected, atol=1e-10)

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Strategy: Use Strassen-like decomposition for 2x2 blocks
    # For 3x4x6, we can decompose into:
    # - Two 2x2x2 Strassen blocks (rank 7 each) = 14
    # - Remaining terms to handle the 3rd row/col and 6 columns
    
    # Let's try a structured approach:
    # Split A (3x4) into [A1; A2] where A1 is 2x4, A2 is 1x4
    # Split B (4x6) into [B1 B2] where B1 is 4x3, B2 is 4x3
    # Result C (3x6) = [A1*B1, A1*B2; A2*B1, A2*B2]
    
    # For the 2x2x2 part (A1[0:2,0:2]*B1[0:2,0:2]), use Strassen (rank 7)
    # For other 2x2x2 blocks, we can also use Strassen if we partition properly
    
    # Alternative: Direct construction with rank reduction
    # The matrix multiplication tensor for n x m x p has rank at least max(n*m, m*p, p*n)
    # For 3x4x6: max(12, 24, 18) = 24, so rank >= 24
    # Schoolbook is 72, so we can potentially do much better
    
    # Let's try to construct a rank-24 decomposition using the structure
    # Actually, for matrix multiplication, the rank is known to be less than n*m*p
    # For 2x2x2, rank is 7 (Strassen) instead of 8
    
    # For 3x4x6, let's try a hybrid approach:
    # Use Strassen for 2x2 subblocks and fill in the rest
    
    # First, let's get a baseline schoolbook decomposition
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Now try to reduce rank by combining terms
    # Key insight: if two terms have the same (i,j) in U and (j,k) in V, 
    # we might be able to combine them
    
    # Let's try a different approach: use the fact that for fixed j,
    # the terms (i,j,k) for varying i and k form a rank-1 update
    # We can use SVD-like decomposition on these slices
    
    # For now, return schoolbook as safe fallback
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # Try to find a lower rank decomposition
    # Approach: For each column index j in [0,m), the sub-tensor T[:,j,:] has shape (n,p)
    # We can write T[:,j,:] = sum_c U[:,j,c] * V[j,:,c] * W[:,:,c]
    # This is a rank decomposition of the "slice" tensor
    
    # Let's try to use the structure more cleverly
    # For 3x4x6, we know that rank(3x4x6) <= rank(2x2x2) * 3 * 2 + corrections
    # Actually, let's just try to optimize from schoolbook
    
    # Use a simple greedy approach to merge terms
    # Two terms c1 and c2 can be merged if there exists a linear combination
    # that reduces the support
    
    # For now, let's try a fixed structured decomposition
    # Based on known results, rank(3,4,6) should be significantly less than 72
    
    # Let's try rank = 48 first (half of schoolbook)
    # Actually, let's just return schoolbook for now and improve iteratively
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
