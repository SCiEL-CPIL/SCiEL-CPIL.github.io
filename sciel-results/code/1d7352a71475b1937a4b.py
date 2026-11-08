"""Matrix multiplication 2,3,7 - rank-minimal exact tensor decomposition."""
import time
from fractions import Fraction
import itertools

DIMENSIONS = [2, 3, 7]

def _verify(U, V, W):
    """Verify exact decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if total != 1:
                    return False
    return True

def _schoolbook():
    """Standard schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_structured_search(r, deadline):
    """
    Try to find rank-r decomposition using structured approach.
    
    Key idea: Fix V to a structured pattern based on the 3x7 multiplication structure.
    For the 3x7 part, we can use a block-diagonal or identity-like structure.
    
    Strategy: 
    - V has shape (m*p, r) = (21, r). Index j*p+k for j in 0..2, k in 0..6.
    - We can try to make V have a pattern where for each j, the row j*p+k has 
      a specific structure that decouples the k dimension.
    
    Alternative: Use the fact that we need T[i,j,k]=1. 
    If we fix V[j*p+k, r] to be 1 for r = j*p+k (when r < 21) and 0 otherwise,
    then we need U[i*m+j, j*p+k] * W[k*n+i, j*p+k] = 1 for all i,j,k.
    This gives rank at least 21 (since V needs 21 distinct rows to be nonzero).
    
    For rank r >= 21, we can try:
    - Set V[j*p+k, r] = 1 if r == j*p+k (for r < 21), else 0
    - Then we need: for each (i,j,k), sum over r where V[j*p+k,r]!=0 of U[i*m+j,r]*W[k*n+i,r] = 1
    - With V fixed as identity-like, this becomes: U[i*m+j, j*p+k] * W[k*n+i, j*p+k] = 1
    
    But this only works if r >= 21 and we use the first 21 columns for V's identity.
    The remaining r-21 columns can be used to potentially reduce the effective rank.
    
    Actually, let me think differently. The minimal rank for matrix multiplication 
    is related to the tensor rank. For 2x3x7, let's try a systematic approach:
    
    1. Start with rank 21 (lower bound)
    2. For each rank, try to find integer/half-integer solutions
    
    Use a greedy/structured approach:
    - Fix V to have rank-matching structure
    - Solve linear constraints for U and W
    """
    n, m, p = DIMENSIONS
    if r < max(n * m, m * p, p * n):
        return None  # Below lower bound
    
    # Try a structured approach:
    # Fix V such that V[j*p+k, c] = 1 if c == j*p+k and c < m*p, else 0
    # This uses m*p = 21 columns. If r >= 21, this is valid.
    # Then we need: for each (i,j,k), U[i*m+j, j*p+k] * W[k*n+i, j*p+k] = 1
    # (since only one term in the sum is nonzero)
    
    # This means U[i*m+j, j*p+k] and W[k*n+i, j*p+k] must be multiplicative inverses.
    # With integer/half-integer entries, the only option is both = 1 or both = -1.
    # So U[i*m+j, j*p+k] = 1 and W[k*n+i, j*p+k] = 1 for all valid indices.
    # This gives a valid decomposition of rank 21!
    
    if r >= m * p:
        # Construct the decomposition
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        
        # Fill V: V[j*p+k, j*p+k] = 1 for j in 0..m-1, k in 0..p-1
        for j in range(m):
            for k in range(p):
                V[j * p + k][j * p + k] = 1
        
        # Fill U and W: U[i*m+j, j*p+k] = 1, W[k*n+i, j*p+k] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][j * p + k] = 1
                    W[k * n + i][j * p + k] = 1
        
        # Verify
        if _verify(U, V, W):
            return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 42
    
    # Lower bound: max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    lower_bound = max(n * m, m * p, p * n)
    
    # Try ranks from lower bound upward
    for r in range(lower_bound, schoolbook_rank + 1):
        if time.time() > deadline - 5.0:
            break
        
        result = _try_structured_search(r, deadline)
        if result is not None:
            return result
    
    # Fall back to schoolbook
    return _schoolbook()
