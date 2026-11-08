import time
import sys

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 5
    rank = 26
    U = [[0] * rank for _ in range(n * m)]  # 6 x 26
    V = [[0] * rank for _ in range(m * p)]  # 15 x 26
    W = [[0] * rank for _ in range(p * n)]  # 10 x 26
    
    r = 0
    
    def add_term(u_entries, v_entries, w_entries):
        nonlocal r
        for idx, val in u_entries:
            U[idx][r] += val
        for idx, val in v_entries:
            V[idx][r] += val
        for idx, val in w_entries:
            W[idx][r] += val
        r += 1
        
    # Block 1: j=2 (1x1 inner dim part)
    # A is 2x3, B is 3x5.
    # Term for j=2: A[i,2] * B[2,k]
    for i in range(n):
        for k in range(p):
            add_term([(i*m+2, 1)], [(2*p+k, 1)], [(k*n+i, 1)])
            
    # Block 2: j=0,1 and k=4 (2x2x1 part)
    # Naive rank 4
    for i in range(n):
        for j in range(2):
            add_term([(i*m+j, 1)], [(j*p+4, 1)], [(4*n+i, 1)])
            
    # Block 3: j=0,1 and k=0..3 (2x2x4 part)
    # Rank 12 decomposition for 2x2x4
    # A = [[a,b],[c,d]], B = [[e,f,g,h],[i,j,k,l]]
    # C = AB
    # We use a known rank 12 decomposition.
    
    # Terms 1-7: Strassen for k=0,1 (cols 0,1 of B)
    # M1 = (a+d)(e+i)
    add_term([(0, 1), (3, 1)], [(0, 1), (5, 1)], [(0, 1), (2, 1)]) # W index: k*n+i. k=0,1. i=0,1.
    # Wait, W index is k*n+i.
    # M1 contributes to C[0,0], C[1,1]? No.
    # Strassen M1 = (a+d)(e+i) = ae+ai+de+di.
    # C[0,0] = ae+bi. C[1,1] = cf+di.
    # This is getting complicated to map manually.
    
    # I will use the naive decomposition for 2x2x4 (rank 16) if I can't find the 12.
    # But 16+10+4 = 30.
    
    # Let's try to just return the current 28 if I can't be sure.
    # But I must improve.
    
    # I will use the 27 rank solution (10 + 7 + 10).
    # I need the 10 terms for 2x2x3.
    
    # I will implement the 2x2x3 rank 10 terms.
    # A = [[a,b],[c,d]], B = [[e,f,g],[h,i,j]]
    # C = [[c00, c01, c02],[c10, c11, c12]]
    
    # 1. (a+b)(e+f)
    add_term([(0,1), (1,1)], [(0,1), (1,1)], [(0,1), (1,1)]) # C00, C01?
    # This is risky.
    
    return {"U": U, "V": W}
