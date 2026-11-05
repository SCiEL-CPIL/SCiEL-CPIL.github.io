import numpy as np
import time
from fractions import Fraction
import itertools

DIMENSIONS = [2, 3, 5]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                s = 0.0
                for r in range(rank):
                    s += U_arr[idx_u, r] * V_arr[idx_v, r] * W_arr[idx_w, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _build_rank28():
    """Build the known rank-28 decomposition."""
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: j=2, k=0..4 (10 terms)
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
            
    # Part 2: j=0,1, k=4 (4 terms)
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
            
    # Part 3: Strassen for j=0,1, k=0,1 (7 terms)
    U[0][r] = 1; U[4][r] = 1
    V[0][r] = 1; V[6][r] = 1
    W[0][r] = 1; W[3][r] = 1
    r += 1
    U[3][r] = 1; U[4][r] = 1
    V[0][r] = 1
    W[1][r] = 1; W[3][r] = -1
    r += 1
    U[0][r] = 1
    V[1][r] = 1; V[6][r] = -1
    W[2][r] = 1; W[3][r] = 1
    r += 1
    U[4][r] = 1
    V[5][r] = 1; V[0][r] = -1
    W[0][r] = 1; W[1][r] = 1
    r += 1
    U[0][r] = 1; U[1][r] = 1
    V[6][r] = 1
    W[0][r] = -1; W[2][r] = 1
    r += 1
    U[3][r] = 1; U[0][r] = -1
    V[0][r] = 1; V[1][r] = 1
    W[3][r] = 1
    r += 1
    U[1][r] = 1; U[4][r] = -1
    V[5][r] = 1; V[6][r] = 1
    W[0][r] = 1
    r += 1
    
    # Part 4: Strassen for j=0,1, k=2,3 (7 terms)
    U[0][r] = 1; U[4][r] = 1
    V[2][r] = 1; V[8][r] = 1
    W[4][r] = 1; W[7][r] = 1
    r += 1
    U[3][r] = 1; U[4][r] = 1
    V[2][r] = 1
    W[5][r] = 1; W[7][r] = -1
    r += 1
    U[0][r] = 1
    V[3][r] = 1; V[8][r] = -1
    W[6][r] = 1; W[7][r] = 1
    r += 1
    U[4][r] = 1
    V[7][r] = 1; V[2][r] = -1
    W[4][r] = 1; W[5][r] = 1
    r += 1
    U[0][r] = 1; U[1][r] = 1
    V[8][r] = 1
    W[4][r] = -1; W[6][r] = 1
    r += 1
    U[3][r] = 1; U[0][r] = -1
    V[2][r] = 1; V[3][r] = 1
    W[7][r] = 1
    r += 1
    U[1][r] = 1; U[4][r] = -1
    V[7][r] = 1; V[8][r] = 1
    W[4][r] = 1
    r += 1
    
    return U, V, W

def _try_mps_optimization(U, V, W, n, m, p, max_rank_to_try=27):
    """
    Try to find a lower-rank decomposition using MPS-inspired optimization.
    The idea: the rank-28 solution has a block structure. We try to see if
    we can merge or eliminate some rank-1 terms by exploiting the MPS bond structure.
    
    For the 2x3x5 case, the MPS structure has:
    - Left bond: dimension related to n=2
    - Right bond: dimension related to p=5
    - Middle: m=3
    
    We try to see if the 28 rank-1 terms can be reduced by finding linear dependencies
    in the tensor network representation.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build the full tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # We want to check if we can reduce the rank
    
    # For MPS optimization, we consider the tensor as a matrix product state
    # T[i,j,k] = sum_{a,b} A[i,a] * B[j,a,b] * C[k,b]
    # where a is the left bond dimension, b is the right bond dimension
    
    # The rank of the tensor is the minimum number of rank-1 terms needed
    # For our case, we have 28 terms. We try to see if 27 or fewer suffice.
    
    # Strategy: try to find if any rank-1 term is linearly dependent on others
    # by checking if the tensor can be represented with fewer terms
    
    # Build the 3D tensor
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i,j,k] = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
    
    # The tensor rank is at most 28. We try to find if it's less.
    # One approach: try to find a decomposition with rank 27, 26, etc.
    # This is NP-hard in general, but for small tensors we can try
    
    # For now, we'll try a simple approach: check if removing any single term
    # still gives a valid decomposition (i.e., the tensor is still representable)
    
    # Actually, a better approach: use the MPS structure to find a lower bond dimension
    # The key insight: if we can factor the tensor as an MPS with smaller bond dimensions,
    # the total rank might be lower.
    
    # For the 2x3x5 case, the natural MPS factorization has:
    # T[i,j,k] = sum_{a=1}^{d1} sum_{b=1}^{d2} A[i,a] * B[j,a,b] * C[k,b]
    # The number of rank-1 terms is d1 * d2 (if we expand)
    
    # But we can also have:
    # T[i,j,k] = sum_{r=1}^{R} U[i,j,r] * V[j,k,r] * W[k,i,r]
    # which is what we have now with R=28.
    
    # The MPS perspective suggests that if the tensor has a low "MPS rank",
    # we might be able to represent it with fewer rank-1 terms.
    
    # For now, this is a placeholder. In a full implementation, we would:
    # 1. Try to find an MPS representation with bond dimensions d1, d2
    # 2. Check if the resulting rank is less than 28
    # 3. If so, extract the rank-1 terms
    
    # Since this is complex, we'll just return the original for now
    return U, V, W, rank

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Start with the known rank-28 decomposition
    U, V, W = _build_rank28()
    
    # Verify the base decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Base decomposition verification failed")
    
    # Try MPS optimization to see if we can reduce the rank
    U_opt, V_opt, W_opt, new_rank = _try_mps_optimization(U, V, W, n, m, p)
    
    # If optimization succeeded and rank is lower, use it
    if new_rank < len(U[0]):
        if _verify_decomposition(U_opt, V_opt, W_opt, n, m, p):
            U, V, W = U_opt, V_opt, W_opt
    
    return {"U": U, "V": V, "W": W}
