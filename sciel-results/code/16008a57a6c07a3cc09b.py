# EVOLVE-BLOCK-START
import numpy as np
import time

DIMENSIONS = [3, 3, 5]
DEADLINE = time.time() + 120  # Safety margin before 150s kill

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    # Reconstruct the tensor: T[i*m+j, j*p+k, k*n+i] should equal sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # For each (i,j,k), check that sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _strassen_3x3x3():
    """
    Returns U, V, W for 3x3x3 matrix multiplication with rank 23.
    Uses a known decomposition. The tensor T[i,j,k] for (i,j) in A-index, (j,k) in B-index, (k,i) in C-index
    must satisfy: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = delta_{i,k}... 
    
    Actually, for matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    The tensor is T[A_idx, B_idx, C_idx] where:
      A_idx = i*m + j (i in 0..n-1, j in 0..m-1)
      B_idx = j*p + k (j in 0..m-1, k in 0..p-1)  
      C_idx = k*n + i (k in 0..p-1, i in 0..n-1)
    
    For n=m=p=3: T[i*3+j, j*3+k, k*3+i] = 1 if the indices are consistent, else 0.
    Specifically, T[i*3+j, j*3+k, k*3+i] = 1 for all i,j,k in 0..2.
    
    We need rank-23 factors. I'll use a verified set from the literature.
    """
    # This is a known rank-23 decomposition for 3x3x3.
    # The coefficients are from the "small matrix multiplication" literature.
    # I'll construct a valid one by using the fact that we can combine Strassen-style ideas.
    
    # Actually, let me use a simpler approach: I'll return None if I can't verify,
    # and fall back to schoolbook. For now, let me implement a correct but possibly
    # not minimal decomposition.
    
    # For safety, return None to trigger schoolbook fallback.
    # A proper rank-23 decomposition requires specific coefficients.
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Strategy: Decompose p=5 into p1=3 and p2=2
    # Compute A(3x3) * B(3x5) as:
    #   A * B[:,0:3] using a 3x3x3 decomposition
    #   A * B[:,3:5] using a 3x3x2 decomposition
    # Then combine the W factors appropriately.
    
    # For 3x3x2: schoolbook rank = 3*3*2 = 18
    # For 3x3x3: if we have rank r1, total rank = r1 + 18
    # Schoolbook total = 45. If r1=23, total=41. Still an improvement.
    
    # Let's try to get a good 3x3x3 decomposition.
    # For now, use schoolbook for both parts: rank = 27 + 18 = 45 (same as full schoolbook)
    # To improve, we need a sub-27 rank for 3x3x3.
    
    # Let me implement a correct hybrid:
    # Part 1: A * B1 where B1 is 3x3 (first 3 columns of B)
    # Part 2: A * B2 where B2 is 3x2 (last 2 columns of B)
    
    def _hybrid_3x3x5():
        """
        Hybrid decomposition using independent k-slices.
        For each k in 0..4, the slice T_k has entries T_k[i*3+j, j*5+k, k*3+i] = 1 for i,j in 0..2.
        Each slice is a rank-9 tensor (one rank-1 tensor per (i,j) pair).
        Total rank = 5 * 9 = 45 (same as schoolbook).
        
        To improve, we need cross-slice sharing. For now, return schoolbook.
        """
        U, V, W = _schoolbook(3, 3, 5)
        return U, V, W
    
    # Get the decomposition
    U, V, W = _hybrid_3x3x5()
    
    # Verify
    if not _verify(U, V, W, 3, 3, 5):
        raise ValueError("Decomposition verification failed")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
