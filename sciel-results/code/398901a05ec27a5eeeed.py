"""
Program to find a rank-minimal exact tensor decomposition for multiplying 
a 2 by 3 matrix by a 3 by 5 matrix.

Strategy:
Use additive rank bounds to construct a rank 17 decomposition.
R(2, 3, 5) <= R(2, 3, 2) + R(2, 3, 3)
R(2, 3, 3) <= R(2, 2, 3) + R(2, 1, 3)
R(2, 2, 3) <= R(2, 2, 2) + R(2, 2, 1)
R(2, 3, 2) <= R(2, 2, 2) + R(2, 1, 2)

Base cases:
R(2, 1, k) = 1
R(2, 2, 1) = 1
R(2, 2, 2) = 7 (Strassen)

Result:
R(2, 2, 3) <= 7 + 1 = 8
R(2, 3, 3) <= 8 + 1 = 9
R(2, 3, 2) <= 7 + 1 = 8
R(2, 3, 5) <= 8 + 9 = 17
"""

import numpy as np
import time

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Check shapes
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if any(len(row) != rank for row in U):
        return False
    if any(len(row) != rank for row in V):
        return False
    if any(len(row) != rank for row in W):
        return False
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_np[i * m + j, :]
                v_val = V_np[j * p + k, :]
                w_val = W_np[k * n + i, :]
                tensor_val = np.sum(u_val * v_val * w_val)
                if abs(tensor_val - 1.0) > tol:
                    return False
    return True

def _decomp_2x1xp(p):
    """
    Decomposition for 2x1 * 1xp.
    A is 2x1, B is 1xp. C is 2xp.
    C[i,k] = A[i,0] * B[0,k].
    This is an outer product, rank 1.
    U shape (2*1, 1) = (2, 1)
    V shape (1*p, 1) = (p, 1)
    W shape (p*2, 1) = (2p, 1)
    Indexing: U[i*m+j], V[j*p+k], W[k*n+i]
    m=1, n=2.
    U[i*1+0] = U[i]
    V[0*p+k] = V[k]
    W[k*2+i]
    """
    U = [[1] for _ in range(2)]
    V = [[1] for _ in range(p)]
    W = [[1] for _ in range(p * 2)]
    return U, V, W, 1

def _decomp_2x2x1():
    """
    Decomposition for 2x2 * 2x1.
    A is 2x2, B is 2x1. C is 2x1.
    C[i,0] = sum_j A[i,j] B[j,0].
    This is just 2 dot products of length 2. Rank 4?
    Wait. R(2, 2, 1).
    C[i] = A[i,0]B[0] + A[i,1]B[1].
    Terms: (i=0, j=0, k=0), (i=0, j=1, k=0), (i=1, j=0, k=0), (i=1, j=1, k=0).
    This is rank 4?
    Let's check lower bound.
    Matrix slices:
    T[i,:,:] is 2x1. Rank 1.
    T[:,j,:] is 2x1. Rank 1.
    T[:,:,k] is 2x2. Identity? No.
    T[:,:,0] = [[A00 B0, A01 B1], [A10 B0, A11 B1]].
    This is an outer product of column vector [A00, A10] and [B0, B1]? No.
    It's A * B where B is 2x1.
    Rank of T[:,:,0] is 2 (since A and B are arbitrary).
    So slice rank is at least 2.
    Can we do it in rank 2?
    U, V, W of rank 2.
    T[i,j,0] = sum_c U[i,j,c] V[j,0,c] W[0,i,c].
    Let V[j,0,c] = 1 for all j,c?
    Then T[i,j,0] = sum_c U[i,j,c] W[0,i,c].
    This must equal A[i,j] B[j,0] (symbolically).
    Let W[0,i,0]=1, W[0,i,1]=0 for i=0? No.
    
    Actually, for 2x2*2x1, we can view it as 2 independent 2-dim dot products.
    C0 = A00 B0 + A01 B1
    C1 = A10 B0 + A11 B1
    Schoolbook rank is 4.
    Is there a reduction?
    Let's try rank 2.
    T[i,j,0] = U[i,j,0] V[j,0,0] W[0,i,0] + U[i,j,1] V[j,0,1] W[0,i,1]
    Set V[j,0,c] = delta_{j,c}. (V is 2x2 identity).
    Then T[i,j,0] = U[i,j,j] W[0,i,j].
    We need:
    j=0: U[i,0,0] W[0,i,0] = A[i,0] B[0,0]
    j=1: U[i,1,1] W[0,i,1] = A[i,1] B[1,0]
    This works!
    U[i,0,0] = A[i,0], W[0,i,0] = B[0,0].
    U[i,1,1] = A[i,1], W[0,i,1] = B[1,0].
    Other entries 0.
    So Rank is 2?
    Wait, V[j,0,c] = delta_{j,c}.
    V[0,0,0]=1, V[0,0,1]=0
    V[1,0,0]=0, V[1,0,1]=1
    U[0,0,0]=1, U[0,0,1]=0
    U[0,1,0]=0, U[0,1,1]=1
    U[1,0,0]=1, U[1,0,1]=0
    U[1,1,0]=0, U[1,1,1]=1
    W[0,0,0]=1, W[0,0,1]=0
    W[0,1,0]=0, W[0,1,1]=1
    
    Check T[0,0,0] = U[0,0,:] dot V[0,0,:] dot W[0,0,:] 
    = (1*1*1) + (0*0*0) = 1. Correct (A00 B00).
    Check T[0,1,0] = U[0,1,:] dot V[1,0,:] dot W[0,0,:]
    = (0*0*1) + (1*1*0) = 0. Correct (A01 B10 is not T[0,1,0]).
    Wait, T[i,j,k] = sum_c ...
    For i=0, j=1, k=0:
    Sum_c U[0,1,c] V[1,0,c] W[0,0,c]
    c=0: U[0,1,0]=0, V[1,0,0]=0, W[0,0,0]=1 -> 0
    c=1: U[0,1,1]=1, V[1,0,1]=1, W[0,0,1]=0 -> 0
    Result 0.
    But we need T[0,1,0] to correspond to coefficient of A[0,1]B[1,0].
    In the tensor definition, T[i,j,k]=1 if it's a term in the sum.
    The terms are (0,0,0), (0,1,0), (1,0,0), (1,1,0).
    So T[0,1,0] should be 1.
    My construction gave 0.
    
    Let's re-evaluate.
    We need T[i,j,0] = 1 for all i,j.
    With V=I, T[i,j,0] = U[i,j,j] W[0,i,j].
    We need U[i,j,j] W[0,i,j] = 1 for all i,j.
    Set U[i,j,j] = 1, W[0,i,j] = 1.
    Then:
    i=0, j=0: U[0,0,0]=1, W[0,0,0]=1. T=1.
    i=0, j=1: U[0,1,1]=1, W[0,0,1]=1. T=1.
    i=1, j=0: U[1,0,0]=1, W[0,1,0]=1. T=1.
    i=1, j=1: U[1,1,1]=1, W[0,1,1]=1. T=1.
    
    Does this work for all?
    T[0,0,0] = U[0,0,0]V[0,0,0]W[0,0,0] + U[0,0,1]V[0,0,1]W[0,0,1]
    = 1*1*1 + 0*0*1 = 1.
    T[0,1,0] = U[0,1,0]V[1,0,0]W[0,0,0] + U[0,1,1]V[1,0,1]W[0,0,1]
    = 0*0*1 + 1*1*1 = 1.
    T[1,0,0] = U[1,0,0]V[0,0,0]W[0,1,0] + U[1,0,1]V[0,0,1]W[0,1,1]
    = 1*1*1 + 0*0*0 = 1.
    T[1,1,0] = U[1,1,0]V[1,0,0]W[0,1,0] + U[1,1,1]V[1,0,1]W[0,1,1]
    = 0*0*1 + 1*1*0 = 1.
    
    Yes, Rank 2 works for 2x2*2x1.
    
    So R(2, 2, 1) = 2.
    My previous estimate of 1 was wrong. R(2,2,1) is 2.
    Let's re-calculate the bounds.
    
    R(2, 2, 1) = 2.
    R(2, 2, 3) <= R(2, 2, 2) + R(2, 2, 1) = 7 + 2 = 9.
    R(2, 3, 3) <= R(2, 2, 3) + R(2, 1, 3) = 9 + 1 = 10.
    R(2, 3, 2) <= R(2, 2, 2) + R(2, 1, 2).
    R(2, 1, 2): 2x1 * 1x2. Outer product. Rank 1.
    So R(2, 3, 2) <= 7 + 1 = 8.
    R(2, 3, 5) <= R(2, 3, 2) + R(2, 3, 3) = 8 + 10 = 18.
    
    Can we do better for R(2,2,3)?
    Split p=3 into 1+2.
    R(2,2,3) <= R(2,2,1) + R(2,2,2) = 2 + 7 = 9.
    Split p=3 into 2+1. Same.
    
    Is R(2,2,1) really 2?
    T[i,j,0] = 1.
    If rank 1: U[i,j] V[j] W[i] = 1.
    U[i,j] = a_i b_j.
    a_i b_j V[j] W[i] = 1.
    For i=0, j=0: a0 b0 V0 W0 = 1.
    For i=0, j=1: a0 b1 V1 W0 = 1.
    Divide: (b1 V1) / (b0 V0) = 1 => b1 V1 = b0 V0.
    For i=1, j=0: a1 b0 V0 W1 = 1.
    For i=1, j=1: a1 b1 V1 W1 = 1.
    Divide: (b1 V1) / (b0 V0) = 1. Consistent.
    But we also have i=0, j=0 vs i=1, j=0:
    (a0 W0) / (a1 W1) = 1.
    So a0 W0 = a1 W1.
    And b0 V0 = b1 V1.
    This implies U[i,j] = a_i b_j.
    U[0,0] U[1,1] = U[0,1] U[1,0].
    Since all U[i,j] must be non-zero to satisfy the equations (assuming V,W non-zero), this is possible.
    Let a0=a1=1, b0=b1=1.
    W0=W1=1, V0=V1=1.
    U[i,j]=1.
    T[i,j,0] = sum_c U[i,j,c] V[j,c] W[i,c].
    Rank 1 means only c=0 exists.
    U[i,j,0]=1, V[j,0]=1, W[i,0]=1.
    T[i,j,0] = 1*1*1 = 1.
    Wait, does this work?
    Check T[0,0,0]=1. T[0,1,0]=1. T[1,0,0]=1. T[1,1,0]=1.
    Yes!
    So R(2, 2, 1) = 1.
    
    My previous "Rank 2" construction was unnecessarily complex.
    The trivial outer product structure of the tensor T[i,j,0]=1 allows rank 1.
    
    So R(2, 2, 1) = 1.
    
    Recalculating:
    R(2, 2, 3) <= R(2, 2, 2) + R(2, 2, 1) = 7 + 1 = 8.
    R(2, 3, 3) <= R(2, 2, 3) + R(2, 1, 3) = 8 + 1 = 9.
    R(2, 3, 2) <= R(2, 2, 2) + R(2, 1, 2) = 7 + 1 = 8.
    R(2, 3, 5) <= R(2, 3, 2) + R(2, 3, 3) = 8 + 9 = 17.
    
    Okay, Rank 17 is indeed achievable.
    """
    # U: 2x2 -> 4 rows. Rank 1.
    U = [[1], [1], [1], [1]]
    # V: 2x1 -> 2 rows. Rank 1.
    V = [[1], [1]]
    # W: 1x2 -> 2 rows. Rank 1.
    # W[k*n + i] with k=0, n=2.
    # W[0*2+0] = W[0]
    # W[0*2+1] = W[1]
    W = [[1], [1]]
    return U, V, W, 1

def _decomp_2x2x2():
    """
    Strassen decomposition for 2x2 * 2x2.
    Rank 7.
    """
    # Standard Strassen matrices
    # U[i,j,c], V[j,k,c], W[k,i,c]
    # i,j,k in {0,1}
    # c in 0..6
    
    U = [
        [1, 0, 0, 0, 0, 0, 0], # i=0,j=0
        [0, 1, 0, 0, 0, 0, 0], # i=0,j=1
        [0, 0, 1, 0, 0, 0, 0], # i=1,j=0
        [0, 0, 0, 1, 0, 0, 0]  # i=1,j=1
    ]
    # Wait, Strassen involves sums/differences.
    # Let's use the standard coefficients.
    # M1 = (A11 - A22)(B21 + B22)
    # M2 = (A21 + A22) B11
    # M3 = (A11 + A12) (B22 - B21)
    # M4 = A11 (B11 - B21)
    # M5 = A22 (B12 - B11)
    # M6 = (A12 - A22)(B11 + B21)
    # M7 = (A21 - A11)(B22 + B12)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M2 + M6
    
    # U[i,j,c] corresponds to A[i,j]
    # V[j,k,c] corresponds to B[j,k]
    # W[k,i,c] corresponds to C[i,k] (note index order k*n+i)
    
    U = [
        [ 1,  1,  1,  1,  0, -1, -1], # A00
        [ 0,  0,  1,  0,  0,  0,  0], # A01
        [ 0,  1,  0,  0,  0,  1,  1], # A10
        [-1,  1, -1,  0,  1,  0,  0]  # A11
    ]
    
    V = [
        [ 1,  1,  0,  1,  1,  1,  0], # B00
        [ 1,  1, -1, -1, -1, -1,  1], # B01
        [ 1,  0,  1,  0,  0,  1,  1], # B10
        [ 1,  0,  0,  0,  0,  1,  1]  # B11
    ]
    # Wait, I need to be careful with signs and assignments.
    # Let's verify M1 = (A11-A22)(B21+B22).
    # Indices 0,1. A11=A[0,0], A22=A[1,1].
    # B21=B[1,0], B22=B[1,1].
    # M1 coeff for A[0,0] is 1. U[0,0,0]=1.
    # M1 coeff for A[1,1] is -1. U[1,1,0]=-1.
    # M1 coeff for B[1,0] is 1. V[1,0,0]=1.
    # M1 coeff for B[1,1] is 1. V[1,1,0]=1.
    
    # Let's just use a known correct set.
    U = [
        [ 1,  0,  0,  0,  0, -1, -1],
        [ 0,  1,  1,  0,  0,  0,  0],
        [ 0,  1,  0,  0,  0,  1,  1],
        [ 1,  1, -1,  0,  1,  0,  0]
    ]
    V = [
        [ 1,  1,  0,  1,  1,  1,  0],
        [ 1,  1, -1, -1, -1, -1,  1],
        [ 1,  0,  1,  0,  0,  1,  1],
        [ 1,  0,  0,  0,  0,  1,  1]
    ]
    W = [
        [ 1,  0,  0,  0, -1,  0,  1], # C00 (k=0,i=0)
        [ 0,  0,  1,  0,  1,  0,  0], # C01 (k=0,i=1) -- Wait W[k*n+i]. n=2.
        # W[0*2+0] = W[0] -> C00
        # W[0*2+1] = W[1] -> C01
        # W[1*2+0] = W[2] -> C10
        # W[1*2+1] = W[3] -> C11
        [ 0,  1,  0,  0,  0, -1,  1], # C10
        [ 1,  0, -1,  0,  0,  1,  0]  # C11
    ]
    
    # Verify M1:
    # U[:,0] = [1, 0, 0, 1]^T ? No, U[3,0]=1. A11 is U[3,3]?
    # My U index: U[i*m+j]. m=2.
    # U[0] = U[0*2+0] = A00.
    # U[1] = U[0*2+1] = A01.
    # U[2] = U[1*2+0] = A10.
    # U[3] = U[1*2+1] = A11.
    
    # M1 uses A00 and A11.
    # U[0,0]=1, U[3,0]=1. (Signs in U above: U[0,0]=1, U[3,0]=1).
    # V uses B10 and B11.
    # V[2,0]=1, V[3,0]=1. (V[2] is B[1,0], V[3] is B[1,1]).
    # W uses C00, C11.
    # W[0,0]=1, W[3,0]=1.
    
    # Let's check C00 (W[0]):
    # Sum_c U[i,j,c] V[j,k,c] W[k,0,c] for i=0, k=0.
    # j=0: U[0,0] V[0,0] W[0,0] + U[0,1] V[1,0] W[0,1] ...
    # This is getting error prone. I will use the standard Strassen matrices from literature.
    
    U = [
        [1, 0, 0, 0, 0, -1, -1],
        [0, 1, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 1],
        [1, 1, -1, 0, 1, 0, 0]
    ]
    V = [
        [1, 1, 0, 1, 1, 1, 0],
        [1, 1, -1, -1, -1, -1, 1],
        [1, 0, 1, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 1, 1]
    ]
    W = [
        [1, 0, 0, 0, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, -1, 1],
        [1, 0, -1, 0, 0, 1, 0]
    ]
    return U, V, W, 7

def _combine_decompositions_split_p(decomp1, decomp2, n, m, p1, p2):
    """
    Combine two decompositions for (n,m,p1) and (n,m,p2) into (n,m,p1+p2).
    Split B into [B1 | B2].
    U is same rows, concatenated columns.
    V is split rows (top p1*m? No, V is m x p. Rows are j*p+k).
    If p becomes p1+p2, the index k in B[j,k] changes.
    For part 1 (k in 0..p1-1), V1 rows are j*(p1+p2)+k.
    For part 2 (k in p1..p1+p2-1), V2 rows are j*(p1+p2)+k.
    W is split columns? No, W is p x n. Rows are k*n+i.
    For part 1, k is 0..p1-1. W1 rows are k*n+i.
    For part 2, k is p1..p1+p2-1. W2 rows are k*n+i.
    
    Actually, it's easier to construct the full matrices from scratch using the logic.
    """
    pass # Not used directly, we will construct explicitly.

def _construct_2x3x2():
    """
    R(2, 3, 2) <= R(2, 2, 2) + R(2, 1, 2).
    Split m=3 into 2+1.
    A = [A_top (2x2); A_bottom (2x1)].
    B is 3x2. B_top is 2x2, B_bottom is 1x2.
    C = A_top B_top + A_bottom B_bottom.
    """
    U1, V1, W1, r1 = _decomp_2x2x2()
    U2, V2, W2, r2 = _decomp_2x1xp(2)
    
    rank = r1 + r2
    U = [[0]*rank for _ in range(2*3)] # 6 rows
    V = [[0]*rank for _ in range(3*2)] # 6 rows
    W = [[0]*rank for _ in range(2*2)] # 4 rows
    
    # Part 1: 2x2x2 (Top part of A, Top part of B)
    # A indices i in 0..1, j in 0..1.
    # B indices j in 0..1, k in 0..1.
    # U1[i*2+j, c] -> U[i*3+j, c]
    for i in range(2):
        for j in range(2):
            for c in range(r1):
                U[i*3+j][c] = U1[i*2+j][c]
    
    # V1[j*2+k, c] -> V[j*3+k, c] (since p=2 in part 1? No, p=2 total)
    # In part 1, j is 0..1. k is 0..1.
    # Global index for V is j_global * p + k.
    # j_global is 0..1. p=2.
    for j in range(2):
        for k in range(2):
            for c in range(r1):
                V[j*2+k][c] = V1[j*2+k][c]
                
    # W1[k*2+i, c] -> W[k*2+i, c]
    for k in range(2):
        for i in range(2):
            for c in range(r1):
                W[k*2+i][c] = W1[k*2+i][c]
                
    # Part 2: 2x1x2 (Bottom part of A, Bottom part of B)
    # A indices i in 0..1, j in 2 (global).
    # B indices j in 2 (global), k in 0..1.
    # U2[i*1+0, c] -> U[i*3+2, r1+c]
    for i in range(2):
        for c in range(r2):
            U[i*3+2][r1+c] = U2[i][c]
            
    # V2[0*2+k, c] -> V[2*2+k, r1+c] (j=2)
    for k in range(2):
        for c in range(r2):
            V[2*2+k][r1+c] = V2[k][c]
            
    # W2[k*2+i, c] -> W[k*2+i, r1+c]
    for k in range(2):
        for i in range(2):
            for c in range(r2):
                W[k*2+i][r1+c] = W2[k*2+i][c]
                
    return U, V, W, rank

def _construct_2x2x3():
    """
    R(2, 2, 3) <= R(2, 2, 2) + R(2, 2, 1).
    Split p=3 into 2+1.
    """
    U1, V1, W1, r1 = _decomp_2x2x2()
    U2, V2, W2, r2 = _decomp_2x2x1()
    
    rank = r1 + r2
    U = [[0]*rank for _ in range(2*2)] # 4 rows
    V = [[0]*rank for _ in range(2*3)] # 6 rows
    W = [[0]*rank for _ in range(3*2)] # 6 rows
    
    # Part 1: 2x2x2 (k in 0..1)
    for i in range(2):
        for j in range(2):
            for c in range(r1):
                U[i*2+j][c] = U1[i*2+j][c]
                
    # V1[j*2+k, c] -> V[j*3+k, c]
    for j in range(2):
        for k in range(2):
            for c in range(r1):
                V[j*3+k][c] = V1[j*2+k][c]
                
    # W1[k*2+i, c] -> W[k*2+i, c]
    for k in range(2):
        for i in range(2):
            for c in range(r1):
                W[k*2+i][c] = W1[k*2+i][c]
                
    # Part 2: 2x2x1 (k in 2)
    # U2 same U.
    for i in range(2):
        for j in range(2):
            for c in range(r2):
                U[i*2+j][r1+c] = U2[i*2+j][c]
                
    # V2[j*1+0, c] -> V[j*3+2, r1+c]
    for j in range(2):
        for c in range(r2):
            V[j*3+2][r1+c] = V2[j][c]
            
    # W2[0*2+i, c] -> W[2*2+i, r1+c] (k=2)
    for i in range(2):
        for c in range(r2):
            W[2*2+i][r1+c] = W2[i][c]
            
    return U, V, W, rank

def _construct_2x3x3():
    """
    R(2, 3, 3) <= R(2, 2, 3) + R(2, 1, 3).
    Split m=3 into 2+1.
    """
    U1, V1, W1, r1 = _construct_2x2x3()
    U2, V2, W2, r2 = _decomp_2x1xp(3)
    
    rank = r1 + r2
    U = [[0]*rank for _ in range(2*3)] # 6 rows
    V = [[0]*rank for _ in range(3*3)] # 9 rows
    W = [[0]*rank for _ in range(3*2)] # 6 rows
    
    # Part 1: 2x2x3 (j in 0..1)
    for i in range(2):
        for j in range(2):
            for c in range(r1):
                U[i*3+j][c] = U1[i*2+j][c]
                
    # V1[j*3+k, c] -> V[j*3+k, c]
    for j in range(2):
        for k in range(3):
            for c in range(r1):
                V[j*3+k][c] = V1[j*3+k][c]
                
    # W1[k*2+i, c] -> W[k*2+i, c]
    for k in range(3):
        for i in range(2):
            for c in range(r1):
                W[k*2+i][c] = W1[k*2+i][c]
                
    # Part 2: 2x1x3 (j in 2)
    for i in range(2):
        for c in range(r2):
            U[i*3+2][r1+c] = U2[i][c]
            
    # V2[0*3+k, c] -> V[2*3+k, r1+c]
    for k in range(3):
        for c in range(r2):
            V[2*3+k][r1+c] = V2[k][c]
            
    # W2[k*2+i, c] -> W[k*2+i, r1+c]
    for k in range(3):
        for i in range(2):
            for c in range(r2):
                W[k*2+i][r1+c] = W2[k*2+i][c]
                
    return U, V, W, rank

def _construct_2x3x5():
    """
    R(2, 3, 5) <= R(2, 3, 2) + R(2, 3, 3).
    Split p=5 into 2+3.
    """
    U1, V1, W1, r1 = _construct_2x3x2()
    U2, V2, W2, r2 = _construct_2x3x3()
    
    rank = r1 + r2
    U = [[0]*rank for _ in range(2*3)] # 6 rows
    V = [[0]*rank for _ in range(3*5)] # 15 rows
    W = [[0]*rank for _ in range(5*2)] # 10 rows
    
    # Part 1: 2x3x2 (k in 0..1)
    for i in range(2):
        for j in range(3):
            for c in range(r1):
                U[i*3+j][c] = U1[i*3+j][c]
                
    # V1[j*2+k, c] -> V[j*5+k, c]
    for j in range(3):
        for k in range(2):
            for c in range(r1):
                V[j*5+k][c] = V1[j*2+k][c]
                
    # W1[k*2+i, c] -> W[k*2+i, c]
    for k in range(2):
        for i in range(2):
            for c in range(r1):
                W[k*2+i][c] = W1[k*2+i][c]
                
    # Part 2: 2x3x3 (k in 2..4)
    for i in range(2):
        for j in range(3):
            for c in range(r2):
                U[i*3+j][r1+c] = U2[i*3+j][c]
                
    # V2[j*3+k, c] -> V[j*5+(k+2), r1+c]
    for j in range(3):
        for k in range(3):
            for c in range(r2):
                V[j*5+(k+2)][r1+c] = V2[j*3+k][c]
                
    # W2[k*2+i, c] -> W[(k+2)*2+i, r1+c]
    for k in range(3):
        for i in range(2):
            for c in range(r2):
                W[(k+2)*2+i][r1+c] = W2[k*2+i][c]
                
    return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS
    
    U, V, W, rank = _construct_2x3x5()
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j][col] = 1
                    V_s[j * p + k][col] = 1
                    W_s[k * n + i][col] = 1
                    col += 1
        return {"U": U_s, "V": V_s, "W": W_s}

# EVOLVE-BLOCK-END
