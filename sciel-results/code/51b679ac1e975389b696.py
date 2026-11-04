"""Tensor decomposition for 3x3x7 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = 3, 3, 7
    
    # The matrix multiplication tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Try to find minimum rank by solving the system
    # For a given rank r, we have:
    # Unknowns: U (n*m x r), V (m*p x r), W (p*n x r)
    # Equations: n*m*p equations (one for each i,j,k)
    # Each equation: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    # This is a non-linear system. Let's try a different approach.
    
    # Key insight: We can fix W and solve for U and V, or use an iterative method.
    
    # Alternative: Use the fact that for fixed k, the equation becomes:
    # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j
    
    # Let's try a greedy/iterative approach or use known decompositions.
    
    # For now, let's try to find a good decomposition by optimization.
    # But given time constraints, let's use a structured approach.
    
    # Approach: Start with trivial rank 63, then try to reduce.
    # Or: Use the fact that we can group by j.
    
    # For each j, we have terms A[i,j]*B[j,k] for all i,k.
    # This is an outer product: for fixed j, the contribution is A[:,j] * B[j,:]
    # which is a rank-1 matrix in the (i,k) space.
    
    # So C = sum_j A[:,j] * B[j,:] = sum_j outer(A[:,j], B[j,:])
    
    # In tensor form: for each j, we have a "block" that contributes to all (i,k) pairs.
    # The tensor entry for (i,j,k) is 1, and this can be written as:
    # T[i,j,k] = delta_{j,j} * 1 * 1 (trivially)
    
    # But we can share: if we define for each j a term where:
    # U[i*m+j, r_j] = 1 for all i, V[j*p+k, r_j] = 1 for all k, W[k*n+i, r_j] = 1 for all i,k
    # Then for this term r_j: U[i*m+j', r_j]*V[j'*p+k, r_j]*W[k*n+i, r_j] = 1 only if j'=j
    # So sum over j gives 1 for each (i,j,k). This is rank m = 3!
    
    # Wait, that's not right either. Let me re-examine.
    
    # If I set for r = 0,1,2 (corresponding to j=0,1,2):
    # U[i*m+j, r] = 1 if j == r, else 0
    # V[j'*p+k, r] = 1 if j' == r, else 0  
    # W[k*n+i, r] = 1 for all k,i
    
    # Then for a given (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # = sum_r [1 if j==r else 0] * [1 if j==r else 0] * 1
    # = 1 (only when r=j)
    
    # So rank 3 works! Let me verify.
    
    rank = m  # = 3
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for j in range(m):
        for i in range(n):
            U[i * m + j, j] = 1
        for k in range(p):
            V[j * p + k, j] = 1
        for k in range(p):
            for i in range(n):
                W[k * n + i, j] = 1
    
    # Verify
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C = A @ B
    
    C_recon = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            for j in range(m):
                C_recon[i, k] += U[i*m+j, j] * V[j*p+k, j] * W[k*n+i, j]
    
    if not np.allclose(C, C_recon, atol=1e-10):
        # Fallback to trivial
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
    
    elapsed = time.time() - start_time
    if elapsed > 100:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
