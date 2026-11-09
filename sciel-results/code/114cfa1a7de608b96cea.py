"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from fractions import Fraction

def _schoolbook_decomposition(n, m, p):
    """Return the standard schoolbook decomposition with rank = n*m*p."""
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

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the factor matrices exactly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The tensor T[i,j,k] for (i in [0,n), j in [0,m), k in [0,p)) should equal
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    rank = len(U[0]) if U else 0
    # Build tensor from factors
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1 if (i, j, k) else 0
                # Actually, the matrix multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k)
                # because C[i,k] = sum_j A[i,j]*B[j,k]
                # The tensor decomposition means: for each (i,k), C[i,k] = sum_j A[i,j] B[j,k]
                # So the 3-tensor T[i,j,k] = 1 for all i,j,k
                # And we need: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                val = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                if val != 1:
                    return False
    return True

def _verify_decomposition_fast(U, V, W, n, m, p):
    """Faster verification using numpy."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For each (i,j,k), check sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
    # We can vectorize this
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U_arr[i*m+j]
                v_row = V_arr[j*p+k]
                w_row = W_arr[k*n+i]
                val = np.dot(u_row, v_row * w_row)
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _try_rank_reduction(n, m, p, target_rank, time_budget):
    """
    Try to find a decomposition of rank target_rank by modifying the schoolbook one.
    Strategy: Start with schoolbook, try to merge columns using linear dependencies.
    """
    start_time = time.time()
    rank = n * m * p
    
    # Initialize with schoolbook
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    # Try to reduce rank by finding linear combinations
    # For each pair of columns, check if we can eliminate one
    # This is a heuristic approach
    
    current_rank = rank
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    attempts = 0
    max_attempts = 1000
    
    while current_rank > target_rank and attempts < max_attempts:
        attempts += 1
        if time.time() - start_time > time_budget * 0.8:
            break
        
        # Try to find two columns that are "similar" and can be merged
        # Randomly pick two column indices
        if current_rank < 2:
            break
        
        c1 = random.randint(0, current_rank - 1)
        c2 = random.randint(0, current_rank - 1)
        if c1 == c2:
            continue
        
        # Check if column c2 can be expressed as a combination involving c1
        # For exact decomposition, we need:
        # For all (i,j,k): U[i*m+j,c2]*V[j*p+k,c2]*W[k*n+i,c2] 
        #   = alpha * U[i*m+j,c1]*V[j*p+k,c1]*W[k*n+i,c1]
        # This is a very restrictive condition.
        
        # Alternative: try to find a linear dependency among the "rank-1 tensors"
        # Each column r defines a tensor T_r[i,j,k] = U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # We need sum_r T_r[i,j,k] = 1 for all (i,j,k)
        # If T_c2 = T_c1, we can remove one (set its contribution to 0 by scaling)
        
        # Check if the two columns define the same tensor
        same = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    t1 = U_arr[i*m+j, c1] * V_arr[j*p+k, c1] * W_arr[k*n+i, c1]
                    t2 = U_arr[i*m+j, c2] * V_arr[j*p+k, c2] * W_arr[k*n+i, c2]
                    if abs(t1 - t2) > 1e-9:
                        same = False
                        break
                if not same:
                    break
            if not same:
                break
        
        if same:
            # Remove column c2 by setting it to zero
            U_arr[:, c2] = 0
            V_arr[:, c2] = 0
            W_arr[:, c2] = 0
            current_rank -= 1
    
    if current_rank > target_rank:
        return None
    
    # Convert back to lists
    U_list = U_arr.tolist()
    V_list = V_arr.tolist()
    W_list = W_arr.tolist()
    
    # Trim to current_rank
    U_list = [row[:current_rank] for row in U_list]
    V_list = [row[:current_rank] for row in V_list]
    W_list = [row[:current_rank] for row in W_list]
    
    # Verify
    if _verify_decomposition_fast(U_list, V_list, W_list, n, m, p):
        return U_list, V_list, W_list
    
    return None

def _structured_search(n, m, p, time_budget):
    """
    Structured search for lower-rank decompositions.
    For 3x4x4 matrix multiplication, the tensor has 3*4*4 = 48 entries.
    The schoolbook rank is 48. We try to find lower ranks.
    
    Known lower bounds: rank >= max(n*m, m*p, p*n) = max(12, 16, 12) = 16
    Upper bound: 48 (schoolbook)
    
    Strategy: Use the structure of matrix multiplication.
    C[i,k] = sum_j A[i,j] B[j,k]
    
    We can try to find a decomposition where some products are shared.
    """
    start_time = time.time()
    best_rank = n * m * p
    best_result = None
    
    U0, V0, W0 = _schoolbook_decomposition(n, m, p)
    best_result = (U0, V0, W0)
    best_rank = n * m * p
    
    # Try target ranks from just below schoolbook down
    for target in range(best_rank - 1, 15, -1):
        if time.time() - start_time > time_budget * 0.7:
            break
        
        result = _try_rank_reduction(n, m, p, target, time_budget * 0.2)
        if result is not None:
            U, V, W = result
            best_result = (U, V, W)
            best_rank = target
            break  # Found a lower rank, stop
    
    return best_result, best_rank

def construct():
    n, m, p = DIMENSIONS
    deadline = 130.0  # Leave 20s safety margin
    start = time.time()
    
    # First, get the schoolbook decomposition as fallback
    U, V, W = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Try structured search for better rank
    if time.time() - start < deadline - 5:
        time_budget = min(deadline - time.time() - 5, 30.0)
        result, rank = _structured_search(n, m, p, time_budget)
        if result is not None and rank < best_rank:
            U, V, W = result
            best_rank = rank
    
    # Final verification
    if _verify_decomposition_fast(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
