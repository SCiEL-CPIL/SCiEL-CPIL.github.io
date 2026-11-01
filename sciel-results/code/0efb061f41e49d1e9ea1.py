"""Tensor decomposition for 2x4 * 4x7 matrix multiplication."""
import time
import random

DIMENSIONS = [2, 4, 7]

def construct():
    """Return a rank-minimal exact tensor decomposition for 2x4 * 4x7."""
    start_time = time.time()
    DEADLINE = 135  # seconds, safety margin before 150s kill
    
    n, m, p = 2, 4, 7
    # Tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We want: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Strategy: Use block decomposition.
    # Split the middle dimension m=4 into two blocks of size 2.
    # T = T_1 + T_2 where T_1 uses j in {0,1} and T_2 uses j in {2,3}.
    # Each sub-tensor is 2x2x7.
    # For a 2x2x7 tensor, we can use Strassen-like or schoolbook.
    # Schoolbook for 2x2x7: rank = 2*2*7 = 28. Total = 56.
    # But we can do better by exploiting structure.
    
    # Actually, let's think about this differently.
    # The tensor is T[i,j,k] = 1 for all i,j,k.
    # This is the all-ones tensor of shape 2x4x7.
    # 
    # Key insight: The all-ones tensor of shape n x m x p has rank min(n*m, m*p, p*n) 
    # if we can factor it. Actually, rank of all-ones tensor = min(n, m, p) ... no.
    # 
    # Let's think: T[i,j,k] = 1. We need sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1.
    # 
    # Simple decomposition: For each (i,j,k) triple, one term. Rank = n*m*p = 56.
    # 
    # Better: Note that T[i,j,k] = (1 for i)*(1 for j)*(1 for k).
    # This is a rank-1 tensor in the "outer product" sense, but our decomposition
    # is different. In our decomposition, the indices are coupled:
    #   U depends on (i,j), V depends on (j,k), W depends on (k,i).
    # 
    # Let's try: U[i*m+j, c] = a_i * b_j, V[j*p+k, c] = c_j * d_k, W[k*n+i, c] = e_k * f_i
    # Then sum_c a_i b_j c_j d_k e_k f_i = (sum_c a_i f_i) ... no, this doesn't separate cleanly.
    # 
    # Let's try a different approach. Since all entries are 1:
    # Try rank = m = 4. 
    # U[i*m+j, c] for c in 0..3: let U[i*m+j, c] = 1 if j == c, else 0.
    # V[j*p+k, c]: let V[j*p+k, c] = 1 if j == c, else 0.
    # W[k*n+i, c]: let W[k*n+i, c] = 1 for all k,i,c.
    # Then sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = sum_c [j==c][j==c][1] = 1 if j in [0,4), which is always 1.
    # Wait: sum_c [j==c] * [j==c] * 1 = sum_c [j==c]^2 = 1 (since j is in 0..3, exactly one c matches).
    # So this works! Rank = 4.
    # 
    # Let me verify:
    # U has shape (2*4, 4) = (8, 4). U[i*m+j, c] = 1 if j==c else 0.
    # V has shape (4*7, 4) = (28, 4). V[j*p+k, c] = 1 if j==c else 0.
    # W has shape (7*2, 4) = (14, 4). W[k*n+i, c] = 1 for all.
    # 
    # Check: For any (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # = sum_c (1 if j==c else 0) * (1 if j==c else 0) * 1
    # = sum_c (1 if j==c else 0)  [since (1 if j==c)^2 = 1 if j==c]
    # = 1 (exactly one c equals j)
    # 
    # This is correct! Rank = 4.
    
    rank = m  # = 4
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            U[i * m + j][j] = 1
    
    for j in range(m):
        for k in range(p):
            V[j * p + k][j] = 1
    
    for k in range(p):
        for i in range(n):
            for c in range(rank):
                W[k * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}
