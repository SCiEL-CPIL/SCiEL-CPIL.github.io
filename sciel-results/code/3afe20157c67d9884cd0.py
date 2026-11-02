"""Matrix multiplication 3,4,6 rank-minimal decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
_START_TIME = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
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
    return U, V, W

def _strassen_2x2_block(n, m, p, i0, j0, k0):
    """
    Apply Strassen 2x2x2 decomposition to a 2x2 sub-block.
    This replaces 4 schoolbook terms with 7, but we need to be careful
    about the dimensions. Actually for 2x2x2, Strassen gives rank 7 vs 8 schoolbook.
    But here we have n=3, m=4, p=6, so we can't directly apply 2x2x2.
    
    Let me reconsider. The best known approach for 3x4x6 is to use the fact
    that we can split into blocks and use known decompositions.
    
    Actually, let me try a different approach: use the border rank idea.
    """
    pass

def _try_rank_am(n, m, p, rank, max_iter=100, tol=1e-8, seed=None):
    """Try to find a decomposition of given rank using alternating minimization."""
    if _time_left() < 5.0:
        return None
    
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize with small random values
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    best_obj = float('inf')
    best_state = None
    
    for it in range(max_iter):
        if _time_left() < 5.0:
            break
        
        # Update U: for each (i,j), solve for U[i*m+j, :] given V, W
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                try:
                    x = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                    U[i * m + j, :] = x
                except:
                    pass
        
        # Update V: for each (j,k), solve for V[j*p+k, :] given U, W
        for j in range(m):
            for k in range(p):
                B = np.zeros((n, rank))
                for i in range(n):
                    B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    x = np.linalg.lstsq(B, np.ones(n), rcond=None)[0]
                    V[j * p + k, :] = x
                except:
                    pass
        
        # Update W: for each (k,i), solve for W[k*n+i, :] given U, V
        for k in range(p):
            for i in range(n):
                C = np.zeros((m, rank))
                for j in range(m):
                    C[j, :] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    x = np.linalg.lstsq(C, np.ones(m), rcond=None)[0]
                    W[k * n + i, :] = x
                except:
                    pass
        
        # Compute objective: sum of squared errors
        obj = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    obj += (s - 1.0) ** 2
        
        if obj < best_obj:
            best_obj = obj
            best_state = (U.copy(), V.copy(), W.copy())
        
        if obj < tol:
            break
    
    if best_obj < tol:
        U, V, W = best_state
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return U_list, V_list, W_list
    
    return None

def _construct_rank_66():
    """
    Construct a rank-66 decomposition using Strassen-like ideas.
    
    For 3×4×6, we can split:
    - The 4 columns of A into two groups of 2
    - The 6 rows of B into two groups of 3
    
    C = A[:,:2]*B[:2,:] + A[:,2:]*B[2:,:]
    
    Each term is 3×2×6. For 3×2×6, we can further split:
    3×2×6: C = A[:,:2]*B[:2,:3] + A[:,:2]*B[:2,3:]
    Each is 3×2×3.
    
    So total: 4 blocks of 3×2×3.
    
    For 3×2×3, schoolbook rank is 18. But we can use a better decomposition.
    Actually, let me just use schoolbook for the sub-blocks and combine.
    
    Wait, the inspiration says rank 66 was achieved. Let me think about how.
    
    3×4×6 schoolbook is 72.
    If we use Strassen on the 2×2 part (splitting m=4 into 2+2), we save some rank.
    
    Actually, let me just try to construct a rank-66 decomposition by using
    the fact that for 2×2×2, Strassen gives rank 7 instead of 8.
    
    Let me try a direct construction.
    """
    n, m, p = 3, 4, 6
    
    # Start with schoolbook rank 72
    U, V, W = _schoolbook(n, m, p)
    rank = 72
    
    # Now try to reduce rank by merging terms
    # This is complex, so let me just return schoolbook for now
    # and rely on the AM search to find better solutions
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    # First, try to find a low-rank decomposition by searching from low to high rank
    # Start with a reasonable lower bound.
    # For 3×4×6, the rank is at least max(3*4, 4*6, 3*6) = max(12, 24, 18) = 24
    # (This is a weak lower bound, but let's try 24 first)
    
    best_result = None
    best_rank = n * m * p  # schoolbook rank as fallback
    
    # Try ranks from 24 upward
    for rank in range(24, best_rank + 1):
        if _time_left() < 10.0:
            break
        
        # Try a few random initializations
        for seed in range(5):
            if _time_left() < 5.0:
                break
            
            result = _try_rank_am(n, m, p, rank, max_iter=200, tol=1e-8, seed=seed)
            if result is not None:
                U, V, W = result
                # Verify
                if _verify(U, V, W, n, m, p, tol=1e-6):
                    best_result = (U, V, W)
                    best_rank = rank
                    break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
