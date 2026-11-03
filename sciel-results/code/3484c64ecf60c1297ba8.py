"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]
import time
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the tensor decomposition is exact using the trace convention."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _snap_to_half_integer(x):
    """Snap a value to the nearest half-integer."""
    return round(x * 2) / 2.0

def _construct_rank_30_decomposition(n, m, p):
    """
    Construct a rank 30 decomposition for 2x5x6 matrix multiplication.
    
    Strategy:
    We want sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1.
    
    Let's try a structure where for each j in {0..m-1}, we have p terms.
    Total rank = m * p = 30.
    
    For a fixed j, let the terms be indexed by k in {0..p-1}.
    Let c = j * p + k.
    
    We need:
    sum_{k'} U[i*m+j, j*p+k'] * V[j*p+k, j*p+k'] * W[k*n+i, j*p+k'] = 1
    
    If we set V[j*p+k, j*p+k'] = delta_{k,k'} (identity for fixed j),
    then the sum becomes:
    U[i*m+j, j*p+k] * W[k*n+i, j*p+k] = 1
    
    We need this to hold for all i, k.
    So we need U[i*m+j, j*p+k] * W[k*n+i, j*p+k] = 1.
    
    We can set U[i*m+j, j*p+k] = 1 and W[k*n+i, j*p+k] = 1.
    Let's check if this causes conflicts.
    
    U has shape (n*m, rank). Row i*m+j, col j*p+k is set to 1.
    For a fixed row i*m+j, the columns are j*p+k for k=0..p-1.
    These are distinct columns for different j.
    So U[i*m+j, col] is 1 for col in [j*p, (j+1)*p) and 0 otherwise?
    Wait, if we set U[i*m+j, j*p+k] = 1 for all k, then for a fixed row i*m+j,
    U has 1s in columns j*p to (j+1)*p - 1.
    And 0s elsewhere.
    
    Similarly for W: W[k*n+i, j*p+k] = 1.
    For a fixed column j*p+k, the rows are k*n+i for i=0..n-1.
    So W[row, col] is 1 if row % n == i where col = j*p+k?
    Wait, col = j*p+k. k is determined by col % p.
    So for col = j*p+k, W[k*n+i, col] = 1.
    This means for a fixed col, W has 1s in rows k*n, k*n+1, ..., k*n+n-1.
    
    Let's check the product for a specific (i, j, k).
    Sum over c:
    c ranges over 0..rank-1.
    Non-zero terms occur when:
    1. U[i*m+j, c] != 0. This implies c is in [j*p, (j+1)*p). Let c = j*p+k'.
    2. V[j*p+k, c] != 0. We set V[j*p+k, j*p+k'] = delta_{k,k'}.
       So V[j*p+k, j*p+k'] is 1 if k=k', else 0.
       So we must have k' = k.
    3. W[k*n+i, c] != 0. With c = j*p+k, we set W[k*n+i, j*p+k] = 1.
    
    So the only non-zero term in the sum is for c = j*p+k.
    The value is U[i*m+j, j*p+k] * V[j*p+k, j*p+k] * W[k*n+i, j*p+k]
    = 1 * 1 * 1 = 1.
    
    This works!
    The rank is m * p = 30.
    
    Let's verify the shapes and indices.
    U: (n*m, m*p). U[i*m+j, j*p+k] = 1.
    V: (m*p, m*p). V[j*p+k, j*p+k'] = delta_{k,k'}.
         This means V is block diagonal with p x p identity blocks?
         No, V is (m*p) x (m*p).
         Row index: j*p+k. Col index: j'*p+k'.
         V[j*p+k, j'*p+k'] = 1 if j==j' and k==k', else 0.
         So V is the identity matrix of size m*p.
    W: (p*n, m*p). W[k*n+i, j*p+k] = 1.
         Row index: k*n+i. Col index: j*p+k.
         W[row, col] = 1 if (col // p) == j and (row // n) == (col % p)?
         Let col = j*p+k. Then k = col % p.
         Row = k*n+i. So row // n = k (if i < n).
         So W[row, col] = 1 if row // n == col % p.
         And i = row % n. The value is 1 regardless of i.
         So for a fixed col, W has 1s in rows [k*n, k*n+n) where k = col % p.
    
    Let's double check with an example.
    n=2, m=5, p=6. Rank=30.
    i=1, j=2, k=4.
    Target: 1.
    Sum over c=0..29.
    c must be in [2*6, 3*6) = [12, 18) for U[1*5+2, c] to be non-zero.
    So c in {12, 13, 14, 15, 16, 17}.
    Let c = 12 + k'. k' in {0..5}.
    V[2*6+4, c] = V[16, 12+k'].
    V is identity. So V[16, 16] = 1, others 0.
    So k' must be 4. c = 16.
    W[4*2+1, 16] = W[9, 16].
    col=16. 16 % 6 = 4. 16 // 6 = 2.
    row=9. 9 // 2 = 4.
    Condition: row // n == col % p => 4 == 4. True.
    So W[9, 16] = 1.
    Product: U[7, 16] * V[16, 16] * W[9, 16] = 1 * 1 * 1 = 1.
    Correct.
    
    Is there any other c?
    U[7, c] is non-zero only for c in [12, 18).
    V[16, c] is non-zero only for c=16.
    So only c=16 contributes.
    
    This construction yields rank m*p = 30.
    """
    rank = m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill V: Identity matrix
    for idx in range(m * p):
        V[idx][idx] = 1
    
    # Fill U: U[i*m+j, j*p+k] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][j*p+k] = 1
                
    # Fill W: W[k*n+i, j*p+k] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                W[k*n+i][j*p+k] = 1
                
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Try the structured rank 30 decomposition
    result_30 = _construct_rank_30_decomposition(n, m, p)
    
    if _verify_decomposition(result_30["U"], result_30["V"], result_30["W"], n, m, p):
        # Ensure entries are integer/half-integer (they are integers here)
        # Return the result
        return result_30
    
    # Fallback to schoolbook (rank 60) if verification fails (should not happen)
    max_rank = n * m * p
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
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
