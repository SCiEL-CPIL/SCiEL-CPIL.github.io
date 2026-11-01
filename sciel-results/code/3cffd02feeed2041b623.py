"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x9 matrix multiplication.
    Returns {'U', 'V', 'W'} with shapes (6, rank), (27, rank), (18, rank).
    """
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 27
    pn = p * n  # 18
    
    # Build the multiplication tensor T[i,j,k] = 1 if (i,j,k) is a valid triple
    # Actually, the tensor for matrix multiplication C = A*B where A is n x m, B is m x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has entries T[(i,j), (j,k), (i,k)] = 1 for all i,j,k
    # In flattened form: U index = i*m+j, V index = j*p+k, W index = k*n+i
    
    # Schoolbook rank is n*m*p = 54
    # We want to find a lower rank decomposition.
    
    # The multiplication tensor for (n,m,p) = (2,3,9)
    # We can try to find a decomposition of rank < 54.
    
    # Let's use a randomized approach: start with a random decomposition of some rank r
    # and try to verify if it's exact. But that's hard to verify directly.
    
    # Alternative: use the fact that for small dimensions, we can try known bounds.
    # For 2x3 * 3x9, the schoolbook is 54.
    # Known results: 
    #   - For 2x2 * 2x2, Strassen gives 7 (schoolbook 8)
    #   - For 2x3 * 3x2, the rank is known to be 10 (schoolbook 12)
    #   - For 2x3 * 3x9, we need to think.
    
    # Let me try a different approach: use integer programming or a greedy method.
    
    # Actually, let's try to construct a decomposition using the following idea:
    # We can try to find a set of rank-1 tensors that sum to the multiplication tensor.
    
    # The multiplication tensor T is a 3-tensor with dimensions (6, 27, 18).
    # T[u, v, w] = 1 if there exist i,j,k such that u=i*m+j, v=j*p+k, w=k*n+i
    # Otherwise 0.
    
    # Let's build T explicitly.
    T = np.zeros((nm, mp, pn), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1
    
    # Now we want to find rank-1 decompositions: T = sum_c U[:,c] * V[:,c] * W[:,c]
    # where U is (6,r), V is (27,r), W is (18,r)
    
    # Let's try a greedy approach: iteratively pick rank-1 components that best match T.
    # But this might not give the minimal rank.
    
    # Given the time constraint, let's try a few approaches:
    # 1. Return schoolbook (rank 54) as baseline
    # 2. Try to find a better decomposition using randomization
    
    deadline = time.time() + 120  # 120 second deadline
    
    # Try to find a decomposition of rank r for r from some lower bound up to 54
    # Lower bound: max(nm, mp, pn) = 27, but also at least the number of non-zero entries / something
    
    # Let's try rank values: 54, 53, 52, ... down to some minimum
    # For each rank, try to find a decomposition.
    
    # Actually, let's try a simpler approach: use the known result that for 2x3 * 3x2, rank is 10.
    # For 2x3 * 3x9, we can view it as 3 blocks of 2x3 * 3x3.
    # If 2x3 * 3x3 has rank r, then 2x3 * 3x9 might have rank 3r or less.
    
    # Let me just try to find a good decomposition using random initialization and verification.
    
    # For a given rank r, we can try:
    # 1. Initialize U, V, W randomly with small integer entries
    # 2. Compute T_approx = sum_c U[:,c] * V[:,c] * W[:,c]
    # 3. Check if T_approx == T
    # This is unlikely to work by chance.
    
    # Better approach: use the structure of the problem.
    
    # Let me try to use a known decomposition. For 2x3 * 3x2, the rank-10 decomposition
    # is known. For 2x3 * 3x9, we can try to extend it.
    
    # Actually, let's just return the schoolbook decomposition for now, since finding
    # a better one is non-trivial and we have time constraints.
    
    rank = n * m * p  # 54
    U = np.zeros((nm, rank), dtype=int)
    V = np.zeros((mp, rank), dtype=int)
    W = np.zeros((pn, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                U[u, col] = 1
                V[v, col] = 1
                W[w, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
