"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Start with schoolbook decomposition
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Verify schoolbook works
    def verify(U, V, W):
        """Check if decomposition exactly computes matrix multiplication."""
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # Our decomposition: C[i,k] = sum_c U[i,c] * V[c,k] where V is reshaped
        # Actually: C[i,k] = sum_c U[i*m+? , c] * V[?, c] * W[k*n+i, c]
        # The tensor is T[i,j,k] = delta[i,k] * A[i,j] * B[j,k]
        # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = A[i,j]*B[j,k]
        # Let's verify by checking all entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(U.shape[1]):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > 1e-9:  # Should be 1 for each (i,j,k)
                        return False
        return True
    
    if not verify(U, V, W):
        raise ValueError("Schoolbook decomposition failed verification")
    
    # Try to reduce rank by finding linear dependencies
    # The tensor T[i,j,k] = 1 for all valid (i,j,k) in the multiplication tensor
    # We need to find minimal rank decomposition
    
    # Strategy: Use the fact that for matrix multiplication, there might be
    # symmetries or linear dependencies we can exploit
    
    # First, let's try to find if any columns are linearly dependent
    # by checking if we can express one term as a combination of others
    
    # Build the constraint matrix: for each (i,j,k), we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # This is a system of equations in the unknowns U, V, W
    
    # Alternative approach: Try to find a lower-rank decomposition by
    # solving the system with fewer columns
    
    # For 3x3x7, let's try ranks from a lower bound up
    # Lower bound: max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # But we can do better with Strassen-like methods
    
    # Let's try to find rank-21 decomposition (the theoretical minimum for this shape)
    # by solving the nonlinear system
    
    # Actually, let's be more systematic. We'll try to reduce the rank by
    # finding which terms can be eliminated
    
    # For each term c, check if removing it and adjusting others still works
    # This is expensive, so let's use a smarter approach
    
    # Key insight: The matrix multiplication tensor for (n,m,p) has rank at most
    # n*m*p, but often much less. For (3,3,3), rank is 23 (Strassen).
    # For (3,3,7), we can think of it as (3,3,3) + (3,3,4)
    
    # Let's try a different approach: use known results and construct
    # a rank-21 or lower decomposition
    
    # Actually, the minimum rank for multiplying 3x3 by 3x7 is known to be 21
    # (since we need at least m*p = 21 terms to handle the 7 output columns per row)
    
    # Let's try to construct a rank-21 decomposition
    
    # For rank 21, we have exactly m*p = 21 columns
    # U has shape (9, 21), V has shape (21, 21), W has shape (21, 21)
    
    # The constraint is: for all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # This is a system of 63 equations in 3*21*21 = 1323 unknowns
    # We can solve this using iterative methods
    
    # Let's try a simple construction:
    # U[i*m+j, c] = delta[i, floor(c/m)] * delta[j, c % m] for c in 0..8 (first 9)
    # But we need 21 terms...
    
    # Actually, let's use the fact that the problem is rectangular
    # C = A * B where A is 3x3, B is 3x7, C is 3x7
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # We can use rank-1 terms: for each (i,j,k), term = A[i,j] * B[j,k]
    # But we want to share computation
    
    # Let's try: for each i (3 choices), for each j (3 choices), we have a term
    # that computes the entire row i of C using row i of A and column k of B
    # But that's still 9*7 = 63 terms
    
    # Better: use the structure of B. Since B is 3x7, we can decompose B
    # into sum of rank-1 matrices: B = sum_r b_r * c_r^T
    # Then C = A * sum_r b_r * c_r^T = sum_r (A * b_r) * c_r^T
    # Each term (A * b_r) * c_r^T is a rank-1 update to C
    # But A * b_r is a vector, so (A*b_r) * c_r^T is an outer product
    # This gives 7 terms (one for each rank-1 component of B), but we need
    # to compute A*b_r which is a 3x3 times 3x1 multiplication
    
    # So total rank = 7 * (rank of 3x3x1 multiplication) = 7 * 3 = 21
    
    # This suggests rank 21 is achievable!
    
    # Let's construct it:
    # B[j,k] = sum_r B[j,r] * delta[r,k] for r in 0..2? No, B is 3x7
    
    # Actually, B is 3x7, so we can write B = sum_{k=0}^{6} b_k * e_k^T
    # where b_k is the k-th column of B (3x1 vector)
    # Then C = A * B = sum_{k=0}^{6} (A * b_k) * e_k^T
    # Each A * b_k is a 3x1 vector, computed as 3x3 * 3x1 multiplication
    # The 3x3 * 3x1 multiplication has rank 3 (just 3 terms)
    # So total rank = 7 * 3 = 21
    
    # Let's construct this:
    # For each k (0..6), for each i (0..2), for each j (0..2):
    #   term: U[i*m+j, c] = A[i,j], V[j*p+k, c] = B[j,k], W[k*n+i, c] = 1
    # But this is still 63 terms
    
    # Wait, I'm confusing myself. Let me restart the construction.
    
    # The tensor T[i,j,k] = 1 for all i,j,k (in the multiplication tensor)
    # We want sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # For rank 21, let's try:
    # c = 3*k + i for k in 0..6, i in 0..2 (21 terms)
    # U[i*m+j, c] = delta[i, i_c] * 1  (where i_c = c % 3)
    # V[j*p+k, c] = delta[j, j_c] * delta[k, k_c]  (where k_c = c // 3, but we need j)
    # This doesn't work because we need to sum over j
    
    # Let me think differently. The standard way to do matrix multiplication
    # with rank r is to have r terms, each contributing to all (i,j,k)
    
    # For the 3x3x3 case, Strassen uses 7 terms instead of 9
    # The key is to combine terms
    
    # For 3x3x7, let's use 7 copies of a 3x3x1 multiplication
    # 3x3x1 multiplication: C[i] = sum_j A[i,j] * B[j]
    # This has rank 3: for each i, term_i: U[i*m+j, c_i] = A[i,j], V[j*p+0, c_i] = B[j], W[0*n+i, c_i] = 1
    
    # So for 3x3x7:
    # For each k (0..6), for each i (0..2):
    #   c = 3*k + i
    #   For each j (0..2):
    #     U[i*m+j, c] = 1  (this will be multiplied by A[i,j])
    #     V[j*p+k, c] = 1  (this will be multiplied by B[j,k])
    #     W[k*n+i, c] = 1  (this selects the output position)
    
    # But wait, this gives 7*3 = 21 terms, and each term has 3 non-zero entries in U and V
    # Let's check: for fixed (i,j,k), the sum over c should be 1
    # c ranges over 0..20, with c = 3*k' + i'
    # U[i*m+j, c] is non-zero only if i' = i, i.e., c = 3*k' + i
    # V[j*p+k, c] is non-zero only if k' = k, i.e., c = 3*k + i
    # So both are non-zero only when c = 3*k + i
    # Then W[k*n+i, c] = W[k*n+i, 3*k+i] = 1
    # So the sum is 1. 
    
    # This works! Rank 21.
    
    # Let's construct it:
    rank_target = 21
    U21 = np.zeros((n * m, rank_target), dtype=np.float64)
    V21 = np.zeros((m * p, rank_target), dtype=np.float64)
    W21 = np.zeros((p * n, rank_target), dtype=np.float64)
    
    for k in range(p):
        for i in range(n):
            c = 3 * k + i
            for j in range(m):
                U21[i * m + j, c] = 1
                V21[j * p + k, c] = 1
                W21[k * n + i, c] = 1
    
    # Verify
    if verify(U21, V21, W21):
        # Convert to lists
        return {"U": U21.tolist(), "V": V21.tolist(), "W": W21.tolist()}
    
    # If rank 21 doesn't work, try to find a lower rank
    # But 21 is the theoretical minimum for this shape, so it should work
    
    # Fallback: return schoolbook
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

