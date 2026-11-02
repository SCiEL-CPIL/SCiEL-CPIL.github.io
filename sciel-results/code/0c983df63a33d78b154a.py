"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]
import time
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the tensor decomposition is exact."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check that the (i,j,k) entry of the multiplication tensor is reproduced
                # The tensor T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                # For matrix multiplication, T[i,j,k] = 1 if this is a valid entry, else 0
                # Actually, the matrix multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k)
                # Wait, that's not right. Let me reconsider.
                
                # The matrix multiplication tensor T has entries T[i,j,k] where:
                # - i indexes the row of A (0 to n-1)
                # - j indexes the column of A / row of B (0 to m-1) 
                # - k indexes the column of B (0 to p-1)
                # T[i,j,k] = 1 for all valid indices, because (AB)[i,k] = sum_j A[i,j]*B[j,k]
                
                # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                val = 0.0
                for c in range(rank):
                    val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(val - 1.0) > 1e-10:
                    return False
    return True

def _try_rank(r, n, m, p, deadline):
    """Try to find a rank-r decomposition. Returns (U, V, W) or None."""
    # For now, use a simple approach: try to find integer/half-integer solutions
    # This is a search problem, so we'll use a basic approach
    
    # Start with the schoolbook and try to reduce rank
    # For small ranks, we can try direct search
    if r >= n * m * p:
        # Schoolbook
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W
    
    # For lower ranks, try a systematic search
    # This is complex, so for now just return None if we can't find one quickly
    # In a real implementation, we would use more sophisticated methods
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Try ranks from low to high
    min_possible = max(n*m, m*p, p*n)  # Lower bound
    max_rank = n * m * p  # Schoolbook rank
    
    # For 2x5x6, try to find a better decomposition than 60
    # Known results suggest rank might be around 40-50 for this size
    
    # Start with schoolbook as fallback
    best_result = None
    U_school = [[0] * max_rank for _ in range(n * m)]
    V_school = [[0] * max_rank for _ in range(m * p)]
    W_school = [[0] * max_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try to find lower rank decompositions
    # For this specific case, try a few candidate ranks
    for r in range(50, max_rank + 1):  # Try ranks from 50 up to 60
        if time.time() > deadline:
            break
        
        result = _try_rank(r, n, m, p, deadline)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p):
                best_result = {"U": U, "V": V, "W": W}
                break  # Found the lowest rank that works
    
    return best_result
# EVOLVE-BLOCK-END
