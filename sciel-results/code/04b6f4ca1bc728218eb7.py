"""Matrix multiplication 3,3,7 tensor decomposition."""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 3, 7]


def _schoolbook():
    n, m, p = DIMENSIONS
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


def _try_find_low_rank(deadline):
    """Attempt to find a rank < 63 decomposition using structured search."""
    n, m, p = DIMENSIONS
    
    # Known result: rank(M_{3,3,7}) could be as low as 21
    # Let's try to find a rank 21 decomposition with a structured approach
    
    # The tensor T has T[i,j,k] = 1 for all i,j,k
    # We need: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k
    
    # Strategy: try rank 21 with a structured pattern
    # If we can find it, great. If not, return None.
    
    # Try a systematic approach: use the fact that we can group
    # Consider that for each (i,k) pair, we need sum_j V[j,k,c]*W[k,i,c]*U[i,j,c] = 1
    # with the constraint that the tensor factorizes.
    
    # Let's try a specific ansatz:
    # For rank 21, we might use 3 copies of a 3x3x3-like structure
    # Actually, let's try to solve it via random search with integer/half-integer entries
    
    # Time budget: use at most 30 seconds for this
    start = time.time()
    max_time = min(30.0, deadline - start)
    
    # Try rank 21 first, then 24, 27, etc.
    for target_rank in [21, 24, 27, 30, 36, 42, 48, 54, 60]:
        if time.time() - start > max_time:
            return None
        
        # Try to find a decomposition of the given rank
        # Use a greedy / structured approach
        # 
        # Key insight: T[i,j,k] = 1 for all i,j,k
        # We can try: U[i,j,c], V[j,k,c], W[k,i,c]
        #
        # Let's try a pattern where for each column c, we assign
        # non-zero entries in a structured way.
        
        # Approach: try all combinations of small integer values
        # This is only feasible for small ranks with structure
        
        # Try a specific structured decomposition for rank 21:
        # 7 columns for each of 3 groups (corresponding to i=0,1,2 or k=0,1,2)
        
        if target_rank == 21:
            # Try: for each k (7 values) and each i (3 values), one column
            # U[i,j,c]: non-zero for fixed i, varies with j
            # V[j,k,c]: non-zero for fixed k, varies with j  
            # W[k,i,c]: non-zero for fixed k,i
            #
            # For column c corresponding to (i0, k0):
            # U[i,j,c] = delta(i,i0) * a_j
            # V[j,k,c] = delta(k,k0) * b_j
            # W[k,i,c] = delta(k,k0)*delta(i,i0) * w
            #
            # Contribution to T[i,j,k]: sum over c where i0=i, k0=k: a_j * b_j * w
            # = (a_j * b_j * w) for the unique c matching (i,k)
            # We need a_j * b_j * w = 1 for all j
            # So we need a_j * b_j = constant for all j
            # Let a_j = 1, b_j = 1, w = 1 for all j
            # Then each column contributes 1 to T[i,j,k] when i=i0 and k=k0
            # But there's exactly one column for each (i0,k0), so T[i,j,k] = 1. 
            # 
            # Wait, but we need to sum over ALL columns. For a fixed (i,j,k),
            # only the column with i0=i and k0=k contributes (since delta functions).
            # So T[i,j,k] = 1 * 1 * 1 = 1. 
            #
            # This works! Rank 21!
            
            U = [[0] * 21 for _ in range(n * m)]
            V = [[0] * 21 for _ in range(m * p)]
            W = [[0] * 21 for _ in range(p * n)]
            
            col = 0
            for i in range(n):
                for k in range(p):
                    # Column col corresponds to (i, k)
                    for j in range(m):
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
            
            # Verify
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for c in range(21):
                            s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                        if s != 1:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            
            if ok:
                return {"U": U, "V": V, "W": W}
    
    return None


def construct():
    start = time.time()
    deadline = start + 120.0  # Safety margin
    
    # Try to find a low-rank decomposition
    result = _try_find_low_rank(deadline)
    
    if result is not None:
        return result
    
    # Fallback to schoolbook
    return _schoolbook()
