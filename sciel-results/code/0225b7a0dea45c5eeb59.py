"""Matrix multiplication 2,3,4 - rank-minimal exact tensor decomposition."""
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 4]

def construct():
    """
    Find a low-rank exact tensor decomposition for 2x3 x 3x4 matrix multiplication.
    
    The tensor condition is:
    For all (i, j, k) with i in [0,n), j in [0,m), k in [0,p):
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    Additionally, for the bilinear algorithm to be valid, we need:
    For all (i, k, a, b, c, d):
        sum_r W[k*n+i, r] * U[a*m+b, r] * V[c*p+d, r] = 1 if (a==i and b==c and d==k) else 0
    
    We'll use a systematic approach: start with the schoolbook rank-24 solution
    and try to find a lower-rank solution by solving a linear system.
    
    Actually, let me use a constructive approach based on the structure of the problem.
    
    For 2x3 x 3x4:
    - U has 6 rows (i,j): (0,0),(0,1),(0,2),(1,0),(1,1),(1,2)
    - V has 12 rows (j,k): (0,0),(0,1),(0,2),(0,3),(1,0),(1,1),(1,2),(1,3),(2,0),(2,1),(2,2),(2,3)
    - W has 8 rows (k,i): (0,0),(0,1),(1,0),(1,1),(2,0),(2,1),(3,0),(3,1)
    
    The full tensor condition (bilinear validity) is:
    For each output (i,k) and each input pair (a,b) from A and (c,d) from B:
        sum_r W[k*n+i, r] * U[a*m+b, r] * V[c*p+d, r] = delta(a,i)*delta(b,c)*delta(d,k)
    
    This is equivalent to saying that for each r, the outer product U_r ⊗ V_r ⊗ W_r
    must contribute correctly.
    
    Let me try a rank-20 construction. I'll use the following idea:
    Group the 24 (i,j,k) triples and find 20 bilinear forms that cover them.
    
    Actually, let me just use a computational approach: set up the linear system
    and solve for a rank-20 solution.
    
    The conditions are:
    For all (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    This is a system of 24 equations. But we also need the cross-term conditions.
    
    Let me use a different approach: I'll construct the solution using the fact that
    the matrix multiplication tensor for (n,m,p) has a known structure.
    
    For 2x3x4, let me try rank 20 with a structured construction.
    
    I'll define 20 columns as follows:
    - 6 columns for the "diagonal" terms (one per (i,j) pair)
    - 14 columns for the remaining terms
    
    Actually, let me just use the schoolbook solution with rank 24. It's guaranteed correct.
    If I can find a better solution, I'll use it, but correctness is paramount.
    
    Wait, let me try one more thing: a rank-20 solution using a specific pattern.
    
    For each j in {0,1,2}, I'll create 7 columns (total 21, then merge one pair to get 20).
    For a fixed j, the terms are A[i,j]*B[j,k] for i in {0,1}, k in {0,1,2,3}.
    That's 8 terms per j, so 24 total. With 7 columns per j, I can handle 7 of the 8 terms
    directly and get the 8th from a linear combination.
    
    Let me try: for j=0, use columns that compute:
    t1 = A[0,0]*B[0,0]
    t2 = A[0,0]*B[0,1]
    t3 = A[0,0]*B[0,2]
    t4 = A[0,0]*B[0,3]
    t5 = A[1,0]*B[0,0]
    t6 = A[1,0]*B[0,1]
    t7 = (A[0,0]+A[1,0])*(B[0,2]+B[0,3])
    
    Then A[1,0]*B[0,2] + A[1,0]*B[0,3] = t7 - A[0,0]*B[0,2] - A[0,0]*B[0,3] - A[1,0]*B[0,0]... 
    No, t7 = A[0,0]*B[0,2] + A[0,0]*B[0,3] + A[1,0]*B[0,2] + A[1,0]*B[0,3]
    So A[1,0]*B[0,2] + A[1,0]*B[0,3] = t7 - t3 - t4
    
    But I need A[1,0]*B[0,2] and A[1,0]*B[0,3] separately, not their sum.
    
    This approach doesn't work directly. Let me just use the schoolbook solution.
    """
    n, m, p = DIMENSIONS  # 2, 3, 4
    
    # Schoolbook decomposition: rank = n*m*p = 24
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
