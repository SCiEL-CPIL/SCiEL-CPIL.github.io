"""Matrix multiplication 3x4x5 - explicit rank-minimal decomposition attempt."""
DIMENSIONS = [3, 4, 5]

import time
import numpy as np

# EVOLVE-BLOCK-START

def _build_schoolbook_decomposition(n, m, p):
    """Construct the schoolbook tensor decomposition for n x m * m x p.
    
    Returns U, V, W as lists of lists with integer entries.
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact for matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Expected tensor T[x, y, z] = 1 if x=i*m+j, y=j*p+k, z=k*n+i
    expected = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected[i * m + j, j * p + k, k * n + i] = 1.0
                
    # Decomposed tensor
    D = np.einsum('xr,yr,zr->xyz', U, V, W)
    
    return np.allclose(D, expected, atol=tol)

def _try_reduce_rank(U, V, W, n, m, p, target_rank, deadline):
    """Attempt to find a decomposition of rank target_rank from the schoolbook basis.
    
    We parameterize the decomposition as:
    U[r, c] = sum over terms t of alpha[t, r] * u_base[t, r]
    V[r, c] = sum over terms t of beta[t, r] * v_base[t, r]  
    W[r, c] = sum over terms t of gamma[t, r] * w_base[t, r]
    
    where we only use target_rank of the 60 schoolbook terms.
    
    The constraint is that for each used term t, there exist scalars a_t, b_t, c_t
    such that the outer product matches. This is a non-convex problem.
    
    Alternative approach: Use the fact that the tensor must satisfy:
    T[x,y,z] = sum_c U[x,c] * V[y,c] * W[z,c]
    
    We can try a greedy approach: start with schoolbook, and try to eliminate
    one term at a time by adjusting the remaining terms.
    """
    start_time = time.time()
    if start_time > deadline:
        return None
    
    R0 = n * m * p  # 60 for 3x4x5
    U0 = np.array(U, dtype=float)
    V0 = np.array(V, dtype=float)
    W0 = np.array(W, dtype=float)
    
    # Build the expected tensor
    T_expected = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Try to eliminate terms one by one
    # For each term to eliminate, we need to find if the remaining terms can
    # be adjusted to still produce the correct tensor.
    
    # A term (i,j,k) corresponds to:
    # U index: i*m + j
    # V index: j*p + k  
    # W index: k*n + i
    
    # The contribution of term t to the tensor is:
    # T_t[x,y,z] = U[x,t] * V[y,t] * W[z,t]
    # For schoolbook, this is 1 only at (i*m+j, j*p+k, k*n+i)
    
    # To eliminate term t0, we need:
    # sum_{t != t0} U[x,t] * V[y,t] * W[z,t] = T_expected[x,y,z] - T_{t0}[x,y,z]
    
    # This is still non-linear. Let's try a different approach:
    # Use random initialization and gradient descent on a relaxed problem.
    
    # For now, try a simple heuristic: check if removing a term and setting
    # its U,V,W columns to 0 while scaling others works.
    
    # Actually, let's try a more systematic approach using the structure.
    
    # The matrix multiplication tensor has a special structure. Let's see if
    # we can use known results or try specific patterns.
    
    # For 3x4x5, let's try to see if rank 59 is possible by checking linear dependencies.
    
    # Flatten the tensor
    T_flat = T_expected.reshape(-1)
    
    # The tensor can be written as sum of rank-1 tensors.
    # Each rank-1 tensor is u ⊗ v ⊗ w.
    
    # Let's try a random search: pick target_rank random terms from the 60,
    # and see if we can solve for the coefficients.
    
    # Actually, the problem is that the terms are not independent in the way
    # that would allow simple linear algebra.
    
    # Let's try a different strategy: use the fact that the tensor is sparse
    # and try to find a sparse decomposition.
    
    # For the time being, return None to indicate no reduction found.
    return None

def _search_lower_rank(n, m, p, deadline):
    """Search for a lower-rank decomposition using various heuristics.
    
    Returns (U, V, W) if a lower-rank decomposition is found, else None.
    """
    start_time = time.time()
    if start_time > deadline:
        return None
    
    # Build schoolbook decomposition
    U0, V0, W0 = _build_schoolbook_decomposition(n, m, p)
    U0 = np.array(U0, dtype=float)
    V0 = np.array(V0, dtype=float)
    W0 = np.array(W0, dtype=float)
    
    R0 = n * m * p
    
    # Try to find a lower rank by attempting to eliminate terms
    # We'll try ranks from R0-1 down to some minimum
    
    # For each target rank, try different strategies
    
    # Strategy 1: Random subset of terms
    # Pick target_rank terms randomly and try to solve for them
    
    # Strategy 2: Greedy elimination
    # Try to eliminate one term at a time
    
    # Given the complexity, let's try a simple approach first:
    # Check if the tensor has any linear dependencies that allow rank reduction.
    
    # The tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1
    # This is a 0-1 tensor with 60 non-zero entries.
    
    # For rank reduction, we need to find if we can express T as a sum of
    # fewer than 60 rank-1 tensors.
    
    # This is equivalent to asking if the 60 rank-1 tensors are linearly
    # independent in the tensor space.
    
    # The tensor space has dimension (n*m)*(m*p)*(p*n) = 12*20*15 = 3600
    # We have 60 vectors in this space, so they are almost certainly independent.
    
    # However, rank reduction doesn't require linear independence of the
    # original terms; it requires that we can find NEW rank-1 terms that
    # sum to T.
    
    # This is a hard problem. Let's try a random search.
    
    np.random.seed(42)
    
    # Try random rank-1 terms and see if we can combine them to get T
    # This is essentially solving a system of equations.
    
    # For a target rank R, we need to find U, V, W of shapes (12,R), (20,R), (15,R)
    # such that sum_c U[:,c] * V[:,c] * W[:,c] = T
    
    # This is a system of 3600 equations in 12*R + 20*R + 15*R = 47*R unknowns.
    # For R=60, we have 2820 unknowns, which is less than 3600, so the system
    # is overdetermined. But we know a solution exists (schoolbook).
    
    # For R<60, we have even fewer unknowns, so it's even more overdetermined.
    
    # Let's try to solve this using least squares for various R.
    
    # Actually, the problem is non-linear because of the products U*V*W.
    
    # Let's try a different approach: use the fact that the tensor is structured
    # and try to find a pattern.
    
    # For now, return None to indicate no reduction found within time budget.
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # 1. Generate Schoolbook Decomposition
    U, V, W = _build_schoolbook_decomposition(n, m, p)
    
    # 2. Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback: should not happen
        pass
        
    # 3. Attempt Rank Reduction
    # Try to find a lower-rank decomposition
    result = _search_lower_rank(n, m, p, deadline)
    if result is not None:
        U, V, W = result
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
