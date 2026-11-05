"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 20
    pn = p * n  # 10
    
    # Build target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Structured rank-1 peeling for rank 20
    # Key insight: For each j, the 10 entries (i,k) can be compressed using 
    # the fact that n=2. We use 5 rank-1 terms per j, each covering 2 k-values.
    
    target_rank = 20
    U = np.zeros((nm, target_rank))
    V = np.zeros((mp, target_rank))
    W = np.zeros((pn, target_rank))
    
    col = 0
    for j in range(m):
        # For this j, we have entries at (i*m+j, j*p+k, k*n+i) for i in {0,1}, k in {0,...,4}
        # Use 5 rank-1 terms, one for each k
        # Each term covers both i=0 and i=1 using a clever construction
        
        for k in range(p):
            # This rank-1 term will handle the diagonal entries for this (j,k)
            # u has support at a=0*m+j=j and a=1*m+j=m+j
            # v has support at b=j*p+k
            # w has support at c=k*n and c=k*n+1
            # 
            # But this creates off-diagonal terms. To fix, we use the structure:
            # We need T[j, j*p+k, k*n] = 1, T[m+j, j*p+k, k*n+1] = 1
            # and T[j, j*p+k, k*n+1] = 0, T[m+j, j*p+k, k*n] = 0
            #
            # Use two rank-1 terms per (j,k) with cancellation, or find a better structure.
            #
            # Alternative: use 1 term per (j,k) with u=[1,0] pattern and separate handling.
            # 
            # Let's use a different approach: for each (j,k), use one term that only
            # activates on the correct (i,c) pair by making u and w have specific patterns.
            
            # Actually, let me use the following construction:
            # For term (j,k), set:
            #   U[j, col] = 1, U[m+j, col] = 1
            #   V[j*p+k, col] = 1
            #   W[k*n, col] = 1, W[k*n+1, col] = 1
            # This gives 4 non-zero entries, but we only want 2.
            # The extra 2 entries will need to be cancelled by other terms.
            
            # This approach is getting complex. Let me try a direct numerical solve instead.
            pass
    
    # Fallback: use iterative refinement to find rank-20 decomposition
    # Initialize with schoolbook-like structure and optimize
    
    np.random.seed(123)
    U = np.zeros((nm, target_rank))
    V = np.zeros((mp, target_rank))
    W = np.zeros((pn, target_rank))
    
    # Initialize: for each (j,k) pair, assign one rank-1 term
    # This gives 4*5=20 terms
    col = 0
    for j in range(m):
        for k in range(p):
            # Term for (j,k): try to cover both i=0 and i=1
            # Use u with both i values, v for this (j,k), w for both c values
            U[j, col] = 1.0
            U[m + j, col] = 1.0
            V[j * p + k, col] = 1.0
            W[k * n, col] = 1.0
            W[k * n + 1, col] = 1.0
            col += 1
    
    # Now check if this is close to T, and if not, try to fix
    # Compute current tensor
    def compute_tensor(U, V, W):
        result = np.zeros((nm, mp, pn))
        for t in range(target_rank):
            u = U[:, t]
            v = V[:, t]
            w = W[:, t]
            result += np.einsum('a,b,c->abc', u, v, w)
        return result
    
    current = compute_tensor(U, V, W)
    residual = T - current
    
    # The residual has off-diagonal terms that need to be cancelled
    # Try to fix by adjusting coefficients
    # For each (j,k), the off-diagonal terms are at (j, j*p+k, k*n+1) and (m+j, j*p+k, k*n)
    # We need to subtract these. Use additional structure or modify existing terms.
    
    # Since we're constrained to rank 20, we need to modify the existing terms.
    # Let's use a local optimization: for each term, adjust to minimize residual
    
    # Simple gradient-free approach: try small perturbations
    best_residual = np.linalg.norm(residual)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to fix the off-diagonal entries by adjusting W coefficients
    # For term (j,k), if we set W[k*n, col] = 1 and W[k*n+1, col] = 0,
    # we only get the i=0 term. Similarly for i=1.
    # But then we need separate terms for i=0 and i=1, giving 40 terms.
    
    # Alternative: use half-integer coefficients to create interference
    # For term (j,k): U[j]=1, U[m+j]=1, V[j*p+k]=1, W[k*n]=1, W[k*n+1]=1
    # This gives: T[j,j*p+k,k*n]+=1, T[j,j*p+k,k*n+1]+=1, 
    #            T[m+j,j*p+k,k*n]+=1, T[m+j,j*p+k,k*n+1]+=1
    # We want: T[j,j*p+k,k*n]=1, T[m+j,j*p+k,k*n+1]=1, others=0
    # So we need to cancel T[j,j*p+k,k*n+1] and T[m+j,j*p+k,k*n]
    
    # These off-diagonal entries are unique to each (j,k) term, so no cancellation between terms.
    # This means this construction doesn't work for rank 20.
    
    # Let me try a completely different construction for rank 20.
    # 
    # New idea: Use the fact that the tensor can be seen as a block matrix.
    # Reshape T as a (nm x mp) matrix for each c, or (mp x pn) for each a, etc.
    # 
    # For fixed a=i*m+j, the slice T[a,:,:] has non-zeros only at (j*p+k, k*n+i) for k=0..4
    # That's 5 non-zeros in a 20x10 matrix.
    # 
    # The slice rank (max rank of any 2D slice) might give a lower bound.
    
    # Let me just return schoolbook if I can't find rank 20 quickly.
    # But first, let me try one more construction.
    
    # Construction: use 20 terms where each term handles a unique (a,b) pair but shares c
    # For each a=i*m+j and each k, we have b=j*p+k and c=k*n+i
    # That's still 40 unique (a,b,c) triples.
    
    # I think rank 20 may not be achievable with a simple construction.
    # Let me try rank 30 as a compromise, or just return schoolbook.
    
    # Return schoolbook (rank 40) as guaranteed solution
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
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
