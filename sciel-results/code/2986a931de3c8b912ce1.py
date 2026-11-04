"""Rewritten program: search for rank-minimal exact tensor decomposition of 3x4 * 4x5 matmul."""
import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

def _verify(U, V, W, n, m, p, tol=1e-12):
    """Check if U, V, W (lists of lists) exactly decompose the matmul tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Build the matmul tensor T[i,j,k] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need to check for all i,j,k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
    # (since each output entry C[i,k] = sum_j A[i,j]*B[j,k], the tensor is the identity in j)
    
    # Actually the matmul tensor T is: T[i, j, k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # The standard tensor: T[i,j,k] = 1 for all i,j,k (it's the "multiplication" tensor)
    # Wait, let me reconsider.
    
    # For matrix multiplication C = A*B where A is n x m, B is m x p, C is n x p:
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    # The tensor T has entries T[i,j,k] = 1 (all ones) - this is the "matmul" tensor.
    # A rank-r decomposition means: T[i,j,k] = sum_{r=0}^{r-1} U[i,j,r] * V[j,k,r] * W[k,i,r]
    # where U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
    
    # So we need: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_{r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    # Compute efficiently using einsum
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # Use einsum: 'ijr,jkr,kir->ijk'
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    # Check if T is all ones
    if not np.allclose(T, np.ones((n, m, p)), atol=tol):
        return False
    return True

def _try_construct_fixed_rank(n, m, p, rank, seed=42):
    """Try to find a rank-`rank` decomposition. Returns (U, V, W) as lists or None."""
    rng = np.random.default_rng(seed)
    
    # For small problems, we can try random initializations and gradient-free approaches
    # But first, let's try known bounds:
    # For 3x4 * 4x5, the schoolbook rank is 60.
    # Strassen-like improvements exist but finding exact low-rank decompositions is hard.
    # 
    # Let's try a structured approach:
    # The matmul tensor has a natural structure. We can try to find decompositions
    # where U, V, W have sparse or low-complexity entries.
    
    # Strategy: Start with schoolbook (rank=60) and try to reduce.
    # For now, return schoolbook if rank >= 60.
    if rank >= n * m * p:
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
    
    # For ranks below schoolbook, we need a search.
    # This is computationally hard. Let's try a few random approaches.
    
    # Approach: Random low-rank factorization with integer/half-integer constraints
    # We'll try random sparse matrices and check.
    
    max_attempts = 1000
    for attempt in range(max_attempts):
        # Generate random sparse U, V, W with entries in {-1, 0, 1, 0.5, -0.5}
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Make them sparse
        for r in range(rank):
            # Each column of U, V, W should be sparse
            nnz_u = rng.integers(1, max(2, n * m // 2))
            nnz_v = rng.integers(1, max(2, m * p // 2))
            nnz_w = rng.integers(1, max(2, p * n // 2))
            
            idx_u = rng.choice(n * m, size=nnz_u, replace=False)
            idx_v = rng.choice(m * p, size=nnz_v, replace=False)
            idx_w = rng.choice(p * n, size=nnz_w, replace=False)
            
            for idx in idx_u:
                U[idx, r] = rng.choice([-1, 1, 0.5, -0.5])
            for idx in idx_v:
                V[idx, r] = rng.choice([-1, 1, 0.5, -0.5])
            for idx in idx_w:
                W[idx, r] = rng.choice([-1, 1, 0.5, -0.5])
        
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 60
    
    # Start with schoolbook as fallback
    U_fb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_fb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_fb = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
    
    best_rank = schoolbook_rank
    best_result = (U_fb, V_fb, W_fb)
    
    # Try to find lower rank decompositions
    # Known lower bounds: For n x m * m x p, rank >= max(n*m, m*p, n*p) = max(12, 20, 15) = 20
    # Also, rank >= 2*max(n,m,p) - 1 is a weak bound.
    # For 3x4*4x5, let's try ranks from 20 up to 59.
    
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin
    
    # Try known small ranks first (more likely to succeed if possible)
    for rank in range(20, schoolbook_rank):
        if time.time() > deadline:
            break
        
        result = _try_construct_fixed_rank(n, m, p, rank, seed=rank)
        if result is not None:
            U, V, W = result
            best_rank = rank
            best_result = (U, V, W)
            # Found a valid decomposition, try to go lower
            break
    
    # If we found a lower rank, we could try to go even lower, but for now
    # let's just return the best we found.
    
    if best_rank < schoolbook_rank:
        U, V, W = best_result
        return {"U": U, "V": V, "W": W}
    else:
        return {"U": U_fb, "V": V_fb, "W": W_fb}
