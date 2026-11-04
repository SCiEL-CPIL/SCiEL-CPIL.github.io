"""
Rank-minimal exact tensor decomposition for 2x3 * 3x7 matrix multiplication.

Strategy:
- Use the schoolbook decomposition as a safe fallback (rank 42).
- Attempt to find a lower-rank decomposition via structured search.
- Use exact rational arithmetic for verification.
- Check the FULL tensor condition: diagonal terms = 1, cross terms = 0.
"""

import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 3, 7]

def _verify_decomposition_full(U, V, W, n, m, p):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    
    U: (n*m, rank) - rows indexed by (i,j) with i in [0,n), j in [0,m)
    V: (m*p, rank) - rows indexed by (j,k) with j in [0,m), k in [0,p)
    W: (p*n, rank) - rows indexed by (k,i) with k in [0,p), i in [0,n)
    
    The tensor T[(i,j), (j',k'), (i'',k'')] should equal:
        delta_{i,i''} * delta_{j,j'} * delta_{k,k'}
    
    So:
    - For i=i'', j=j', k=k'': sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    - For all other combinations: sum = 0
    """
    rank = len(U[0])
    
    # Check diagonal terms (should equal 1)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    u_val = Fraction(str(U[i*m+j][c]))
                    v_val = Fraction(str(V[j*p+k][c]))
                    w_val = Fraction(str(W[k*n+i][c]))
                    total += u_val * v_val * w_val
                if total != 1:
                    return False
    
    # Check cross terms (should equal 0)
    # We need to check all combinations where (i,j,k) != (i'',j',k') in the sense that
    # i != i'' or j != j' or k != k''
    # But actually, the tensor is defined on triples ((i,j), (j',k'), (i'',k''))
    # and equals 1 iff i=i'', j=j', k=k'', else 0.
    
    # To be thorough, check a sample of cross terms or all of them.
    # For efficiency, check all combinations where at least one index mismatches.
    
    # Total number of combinations: n*m * m*p * p*n = (n*m)*(m*p)*(p*n)
    # For n=2,m=3,p=7: 6*21*14 = 1764 combinations
    # This is manageable.
    
    for i in range(n):
        for j in range(m):
            for jp in range(m):  # j'
                for k in range(p):
                    for ip in range(n):  # i''
                        for kp in range(p):  # k'
                            # Check if this is a diagonal term
                            if i == ip and j == jp and k == kp:
                                continue  # Already checked
                            # This should be 0
                            total = Fraction(0)
                            for c in range(rank):
                                u_val = Fraction(str(U[i*m+j][c]))
                                v_val = Fraction(str(V[jp*p+k][c]))
                                w_val = Fraction(str(W[kp*n+ip][c]))
                                total += u_val * v_val * w_val
                            if total != 0:
                                return False
    
    return True

def _verify_decomposition_fast(U, V, W, n, m, p):
    """
    Fast verification using numpy for the linear algebra part.
    Converts to float for speed, then checks with tolerance.
    Used for quick rejection during search.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Reshape
    U_mat = U_arr.reshape(n, m, rank)  # U[i, j, c]
    V_mat = V_arr.reshape(m, p, rank)  # V[j, k, c]
    W_mat = W_arr.reshape(p, n, rank)  # W[k, i, c]
    
    # Check diagonal: for each (i,j,k), sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                prod = U_mat[i, j, :] * V_mat[j, k, :] * W_mat[k, i, :]
                if abs(np.sum(prod) - 1.0) > 1e-10:
                    return False
    
    # Check a sample of cross terms
    # For efficiency, check random cross terms
    for _ in range(100):
        i = random.randint(0, n-1)
        j = random.randint(0, m-1)
        jp = random.randint(0, m-1)
        k = random.randint(0, p-1)
        ip = random.randint(0, n-1)
        kp = random.randint(0, p-1)
        if i == ip and j == jp and k == kp:
            continue
        prod = U_mat[i, j, :] * V_mat[jp, k, :] * W_mat[kp, ip, :]
        if abs(np.sum(prod)) > 1e-10:
            return False
    
    return True

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

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    best_result = _schoolbook()
    best_rank = n * m * p
    
    # Try to find lower-rank decompositions
    # For 2x3*3x7, try ranks from 20 to 41
    for rank in range(20, best_rank):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random initialization with small integer values
            U = [[random.randint(-1, 1) for _ in range(rank)] for _ in range(nm)]
            V = [[random.randint(-1, 1) for _ in range(rank)] for _ in range(mp)]
            W = [[random.randint(-1, 1) for _ in range(rank)] for _ in range(pn)]
            
            # Fast check first
            if _verify_decomposition_fast(U, V, W, n, m, p):
                # Full exact verification
                if _verify_decomposition_full(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
    
    # If no lower rank found, return schoolbook
    return best_result
