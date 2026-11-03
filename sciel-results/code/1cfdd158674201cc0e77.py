"""Rank-34 exact tensor decomposition for 2x3 by 3x6 matrix multiplication."""
import numpy as np
import time

def _build_rank6_2x1x3():
    """
    Rank-6 decomposition for M(2,1,3).
    A is 2x1, B is 1x3, C is 2x3.
    C[i,k] = A[i,0] * B[0,k].
    """
    n, m, p = 2, 1, 3
    r = n * p # 6
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    
    idx = 0
    for i in range(n):
        for k in range(p):
            U[i*m + 0][idx] = 1
            V[0*p + k][idx] = 1
            W[k*n + i][idx] = 1
            idx += 1
    return U, V, W

def _build_rank4_2x2x1():
    """
    Rank-4 decomposition for M(2,2,1).
    A is 2x2, B is 2x1, C is 2x1.
    C[i,0] = A[i,0]B[0,0] + A[i,1]B[1,0].
    """
    n, m, p = 2, 2, 1
    r = n * m # 4
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    
    idx = 0
    for i in range(n):
        for j in range(m):
            U[i*m + j][idx] = 1
            V[j*p + 0][idx] = 1
            W[0*n + i][idx] = 1
            idx += 1
    return U, V, W

def _build_rank7_2x2x2():
    """
    Rank-7 decomposition for M(2,2,2) using Strassen.
    """
    n, m, p = 2, 2, 2
    r = 7
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    
    # Strassen coefficients
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -> W[k*n+i]
    
    # Term 0: M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[2][0] = -1; W[3][0] = 1
    
    # Term 1: M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # Term 2: M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # Term 3: M4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # Term 4: M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # Term 5: M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # Term 6: M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _combine_blocks(U1, V1, W1, U2, V2, W2, offset):
    """
    Combine two decompositions.
    U1, V1, W1 are for the first part.
    U2, V2, W2 are for the second part.
    offset is the rank offset for the second part.
    """
    r1 = len(U1[0])
    r2 = len(U2[0])
    r_total = r1 + r2
    
    # We need to know the dimensions to allocate the combined matrices
    # This function is too generic without dims. 
    # Better to do it in construct.
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    n, m, p = 2, 3, 6
    
    # Decomposition for M(2,3,3)
    # Part 1: j=0. M(2,1,3). Rank 6.
    # A indices: i in {0,1}, j=0.
    # B indices: j=0, k in {0,1,2}.
    # C indices: i in {0,1}, k in {0,1,2}.
    U_0, V_0, W_0 = _build_rank6_2x1x3()
    
    # Part 2: j in {1,2}. M(2,2,3).
    # This is split into M(2,2,1) for k=0 (relative to block? No, k is global)
    # Wait, the split for M(2,2,3) is on the output k dimension?
    # No, the split for M(2,2,3) was on the B columns.
    # B is 2x3. Split into B1 (2x1, col 0) and B2 (2x2, cols 1,2).
    # C1 = A B1 (2x1). C2 = A B2 (2x2).
    # So for the full M(2,3,3):
    # j=0 part: M(2,1,3). Output k in {0,1,2}.
    # j=1,2 part: 
    #   Sub-part A: B cols 0 (global k=0). M(2,2,1). Output k=0.
    #   Sub-part B: B cols 1,2 (global k=1,2). M(2,2,2). Output k=1,2.
    
    # So we need to map the local indices of the sub-blocks to global indices.
    
    # Global Dimensions:
    # U: (n*m) x Rank = 6 x Rank
    # V: (m*p) x Rank = 18 x Rank
    # W: (p*n) x Rank = 12 x Rank
    
    # Let's build the full M(2,3,3) decomposition first.
    r_sub_total = 0
    U_sub = []
    V_sub = []
    W_sub = []
    
    # 1. Add M(2,1,3) for j=0
    # Local U: 2x1 -> 2 rows. Global U rows 0,1 (j=0).
    # Local V: 1x3 -> 3 rows. Global V rows 0,1,2 (j=0, k=0,1,2).
    # Local W: 3x2 -> 6 rows. Global W rows 0..5 (k=0,1,2; i=0,1).
    U1, V1, W1 = _build_rank6_2x1x3()
    r1 = len(U1[0])
    # Append to lists
    for i in range(6): # n*m
        U_sub.append([0]*r1)
    for j in range(18): # m*p
        V_sub.append([0]*r1)
    for k in range(12): # p*n
        W_sub.append([0]*r1)
        
    # Fill U_sub for j=0
    for i in range(2):
        for c in range(r1):
            U_sub[i*3 + 0][c] = U1[i][c] # U1 has 2 rows (i), 1 col (j=0) -> index i*1+0
    # Fill V_sub for j=0
    for k in range(3):
        for c in range(r1):
            V_sub[0*6 + k][c] = V1[0*3 + k][c] # V1 has 1 row (j=0), 3 cols (k)
    # Fill W_sub for k=0,1,2
    for k in range(3):
        for i in range(2):
            for c in range(r1):
                W_sub[k*2 + i][c] = W1[k*2 + i][c]
                
    r_sub_total += r1
    
    # 2. Add M(2,2,1) for j=1,2 and k=0
    # A indices: i in {0,1}, j in {1,2}.
    # B indices: j in {1,2}, k=0.
    # C indices: i in {0,1}, k=0.
    U2, V2, W2 = _build_rank4_2x2x1()
    r2 = len(U2[0])
    
    # Expand U_sub, V_sub, W_sub
    for row in U_sub:
        row.extend([0]*r2)
    for row in V_sub:
        row.extend([0]*r2)
    for row in W_sub:
        row.extend([0]*r2)
        
    # Fill U_sub for j=1,2
    # U2 has shape 4x4 (i,j local). Global U index i*3 + j.
    for i in range(2):
        for j_local in range(2):
            j_global = 1 + j_local
            for c in range(r2):
                U_sub[i*3 + j_global][r1 + c] = U2[i*2 + j_local][c]
                
    # Fill V_sub for j=1,2 and k=0
    # V2 has shape 2x1 (j_local, k=0). Global V index j*6 + k.
    for j_local in range(2):
        j_global = 1 + j_local
        for c in range(r2):
            V_sub[j_global*6 + 0][r1 + c] = V2[j_local*1 + 0][c]
            
    # Fill W_sub for k=0
    # W2 has shape 1x2 (k=0, i). Global W index k*2 + i.
    for i in range(2):
        for c in range(r2):
            W_sub[0*2 + i][r1 + c] = W2[0*1 + i][c]
            
    r_sub_total += r2
    
    # 3. Add M(2,2,2) for j=1,2 and k=1,2
    # A indices: i in {0,1}, j in {1,2}.
    # B indices: j in {1,2}, k in {1,2}.
    # C indices: i in {0,1}, k in {1,2}.
    U3, V3, W3 = _build_rank7_2x2x2()
    r3 = len(U3[0])
    
    for row in U_sub:
        row.extend([0]*r3)
    for row in V_sub:
        row.extend([0]*r3)
    for row in W_sub:
        row.extend([0]*r3)
        
    # Fill U_sub for j=1,2
    for i in range(2):
        for j_local in range(2):
            j_global = 1 + j_local
            for c in range(r3):
                U_sub[i*3 + j_global][r1 + r2 + c] = U3[i*2 + j_local][c]
                
    # Fill V_sub for j=1,2 and k=1,2
    for j_local in range(2):
        j_global = 1 + j_local
        for k_local in range(2):
            k_global = 1 + k_local
            for c in range(r3):
                V_sub[j_global*6 + k_global][r1 + r2 + c] = V3[j_local*2 + k_local][c]
                
    # Fill W_sub for k=1,2
    for k_local in range(2):
        k_global = 1 + k_local
        for i in range(2):
            for c in range(r3):
                W_sub[k_global*2 + i][r1 + r2 + c] = W3[k_local*2 + i][c]
                
    r_sub_total += r3
    
    # Now we have U_sub, V_sub, W_sub for M(2,3,3) with rank r_sub_total = 6+4+7=17.
    
    # Now construct M(2,3,6) by duplicating this for the two blocks of B (cols 0-2 and 3-5).
    # U is shared (same A).
    # V is split: V_block1 for k=0,1,2 and V_block2 for k=3,4,5.
    # W is split: W_block1 for k=0,1,2 and W_block2 for k=3,4,5.
    
    total_rank = 2 * r_sub_total
    U_final = [[0]*total_rank for _ in range(n*m)]
    V_final = [[0]*total_rank for _ in range(m*p)]
    W_final = [[0]*total_rank for _ in range(p*n)]
    
    # Block 1 (k=0,1,2)
    for i in range(n*m):
        for c in range(r_sub_total):
            U_final[i][c] = U_sub[i][c]
            
    for j in range(m):
        for k in range(3):
            for c in range(r_sub_total):
                V_final[j*p + k][c] = V_sub[j*3 + k][c]
                
    for k in range(3):
        for i in range(n):
            for c in range(r_sub_total):
                W_final[k*n + i][c] = W_sub[k*n + i][c]
                
    # Block 2 (k=3,4,5)
    offset = r_sub_total
    for i in range(n*m):
        for c in range(r_sub_total):
            U_final[i][offset + c] = U_sub[i][c]
            
    for j in range(m):
        for k in range(3, 6):
            k_local = k - 3
            for c in range(r_sub_total):
                V_final[j*p + k][offset + c] = V_sub[j*3 + k_local][c]
                
    for k in range(3, 6):
        k_local = k - 3
        for i in range(n):
            for c in range(r_sub_total):
                W_final[k*n + i][offset + c] = W_sub[k_local*n + i][c]
                
    return {"U": U_final, "V": V_final, "W": W_final}
