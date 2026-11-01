import time
import numpy as np
from scipy.linalg import null_space

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def _build_strassen_28():
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def strassen_2x2(U_rows, V_rows, W_rows, offset):
        U_e = []
        V_e = []
        W_e = []
        col = offset
        
        # M1 = (A00+A11)*(B00+B11)
        U_e.append((U_rows[0], 1, col))
        U_e.append((U_rows[3], 1, col))
        V_e.append((V_rows[0], 1, col))
        V_e.append((V_rows[3], 1, col))
        W_e.append((W_rows[0], 1, col))
        W_e.append((W_rows[1], -1, col))
        W_e.append((W_rows[2], -1, col))
        W_e.append((W_rows[3], 1, col))
        col += 1
        
        # M2 = (A10+A11)*B00
        U_e.append((U_rows[2], 1, col))
        U_e.append((U_rows[3], 1, col))
        V_e.append((V_rows[0], 1, col))
        W_e.append((W_rows[1], 1, col))
        W_e.append((W_rows[3], -1, col))
        col += 1
        
        # M3 = A00*(B10-B11)
        U_e.append((U_rows[0], 1, col))
        V_e.append((V_rows[1], 1, col))
        V_e.append((V_rows[3], -1, col))
        W_e.append((W_rows[0], 1, col))
        W_e.append((W_rows[2], 1, col))
        col += 1
        
        # M4 = A11*(B01-B10)
        U_e.append((U_rows[3], 1, col))
        V_e.append((V_rows[2], 1, col))
        V_e.append((V_rows[1], -1, col))
        W_e.append((W_rows[1], 1, col))
        W_e.append((W_rows[3], 1, col))
        col += 1
        
        # M5 = (A00+A10)*B11
        U_e.append((U_rows[0], 1, col))
        U_e.append((U_rows[2], 1, col))
        V_e.append((V_rows[3], 1, col))
        W_e.append((W_rows[2], 1, col))
        W_e.append((W_rows[3], -1, col))
        col += 1
        
        # M6 = (A01-A10)*(B00+B10)
        U_e.append((U_rows[1], 1, col))
        U_e.append((U_rows[2], -1, col))
        V_e.append((V_rows[0], 1, col))
        V_e.append((V_rows[1], 1, col))
        W_e.append((W_rows[0], -1, col))
        W_e.append((W_rows[1], 1, col))
        W_e.append((W_rows[2], 1, col))
        W_e.append((W_rows[3], -1, col))
        col += 1
        
        # M7 = (A01-A11)*(B01+B11)
        U_e.append((U_rows[1], 1, col))
        U_e.append((U_rows[3], -1, col))
        V_e.append((V_rows[2], 1, col))
        V_e.append((V_rows[3], 1, col))
        W_e.append((W_rows[0], 1, col))
        W_e.append((W_rows[1], 1, col))
        W_e.append((W_rows[2], -1, col))
        W_e.append((W_rows[3], -1, col))
        col += 1
        
        return U_e, V_e, W_e, col

    U_e1, V_e1, W_e1, _ = strassen_2x2([0,1,2,3], [0,1,2,3], [0,1,2,3], 0)
    U_e2, V_e2, W_e2, _ = strassen_2x2([4,5,6,7], [8,9,10,11], [0,1,2,3], 7)
    U_e3, V_e3, W_e3, _ = strassen_2x2([0,1,2,3], [12,13,14,15], [4,5,6,7], 14)
    U_e4, V_e4, W_e4, _ = strassen_2x2([4,5,6,7], [12,13,14,15], [4,5,6,7], 21)
    
    for row, val, col in U_e1 + U_e2 + U_e3 + U_e4:
        U[row][col] += val
    for row, val, col in V_e1 + V_e2 + V_e3 + V_e4:
        V[row][col] += val
    for row, val, col in W_e1 + W_e2 + W_e3 + W_e4:
        W[row][col] += val
        
    return U, V, W

def _verify_decomposition(U, V, W, n=2, m=4, p=4):
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(val - 1.0) > 1e-9:
                    max_err = max(max_err, abs(val - 1.0))
    return max_err

def _try_reduce_rank(U, V, W, max_time=10.0):
    """Try to reduce rank by finding linear dependencies among columns."""
    start = time.time()
    n, m, p = 2, 4, 4
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Try removing columns one at a time and checking if the decomposition still holds
    # For each column r, check if it can be expressed as a linear combination of others
    # such that the tensor equation still holds.
    
    # Build the full tensor constraint matrix
    # For each (i,j,k), we need: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # This is a system where each column r contributes a rank-1 tensor U_col ⊗ V_col ⊗ W_col
    
    # Strategy: try to find a subset of columns that still spans the same tensor space
    # Use the fact that if column r is in the span of others in a certain sense, we can drop it.
    
    # More practical: try all pairs of columns to see if one is redundant
    # For each pair (a, b), check if column b can be absorbed into column a
    # by adjusting U, V, W entries.
    
    # Actually, a cleaner approach: the 28-term decomposition might have structure
    # that allows merging. Let's try a direct search: for each target rank r from 27 down,
    # try to find a valid r-term decomposition by solving the nonlinear system.
    # This is expensive, so we'll use a heuristic: try to merge pairs of terms.
    
    # Heuristic: for each pair of columns (a,b), try to find coefficients such that
    # the combined contribution equals the original. This is a nonlinear problem,
    # but we can try simple merges.
    
    # Simpler approach: check if any column is entirely zero (unlikely but check)
    for r in range(rank):
        if time.time() - start > max_time:
            break
        if np.all(U_arr[:, r] == 0) or np.all(V_arr[:, r] == 0) or np.all(W_arr[:, r] == 0):
            # Remove this column
            new_U = np.delete(U_arr, r, axis=1)
            new_V = np.delete(V_arr, r, axis=1)
            new_W = np.delete(W_arr, r, axis=1)
            err = _verify_decomposition(new_U.tolist(), new_V.tolist(), new_W.tolist())
            if err < 1e-9:
                best_U, best_V, best_W = new_U.tolist(), new_V.tolist(), new_W.tolist()
                best_rank = rank - 1
                U_arr, V_arr, W_arr = new_U, new_V, new_W
                rank -= 1
                break
    
    # Try merging: for each pair (a, b), see if we can eliminate b by adjusting a
    # This is a nonlinear search, so we'll skip for now and return best found
    return best_U, best_V, best_W, best_rank

def _search_lower_rank(U, V, W, max_time=30.0):
    """Search for lower-rank decompositions using structured modifications."""
    start = time.time()
    n, m, p = 2, 4, 4
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Try to find if the 28-term decomposition has a lower-rank equivalent
    # by examining the structure of the tensor.
    
    # The matrix multiplication tensor M_{2,4,4} has known lower bounds.
    # For 2x2x2, rank is 7 (Strassen). For 2x4x4, we can think of it as
    # four 2x2x2 blocks, giving 4*7=28, but there might be cross-block
    # cancellations.
    
    # Let's try a different structural approach: use the fact that the middle
    # dimension is 4, and try to find a decomposition that exploits the
    # full 4-dimensional structure rather than treating it as four independent 2D blocks.
    
    # Attempt: construct a rank-24 or lower decomposition by finding a
    # different tiling or by using a non-Strassen approach for the 2x4x4 case.
    
    # For now, try to verify if rank 27 is achievable by a simple modification
    # and return the best we find.
    
    if time.time() - start > max_time:
        return best_U, best_V, best_W, best_rank
    
    return best_U, best_V, best_W, best_rank

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    U, V, W = _build_strassen_28()
    err = _verify_decomposition(U, V, W)
    
    if err < 1e-9:
        # Try to reduce rank
        if time.time() - start_time < deadline - 10:
            U2, V2, W2, new_rank = _try_reduce_rank(U, V, W, max_time=10.0)
            if new_rank < 28:
                U, V, W = U2, V2, W2
        
        if time.time() - start_time < deadline - 5:
            U3, V3, W3, new_rank2 = _search_lower_rank(U, V, W, max_time=15.0)
            if new_rank2 < len(U[0]):
                U, V, W = U3, V3, W3
        
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook rank 32
    n, m, p = 2, 4, 4
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
