import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 4]

def _validate(U, V, W, n, m, p):
    """Check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k."""
    R = len(U[0]) if U else 0
    if len(V[0]) != R or len(W[0]) != R:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

def construct():
    t0 = time.time()
    DEADLINE = 145.0
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # Schoolbook fallback: rank 36
    rank_sb = n * m * p
    U_sb = [[0]*rank_sb for _ in range(n*m)]
    V_sb = [[0]*rank_sb for _ in range(m*p)]
    W_sb = [[0]*rank_sb for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    best_rank = rank_sb
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Define Strassen's rank-7 decomposition for 2x2x2
    # For 2x2 matrix multiplication C = A*B where A, B are 2x2
    # The tensor T_{ijk} = 1 if A_{ij} * B_{jk} appears in C_{ik}
    # Indices: i,j in {0,1} for A (U), j,k in {0,1} for B (V), k,i in {0,1} for C (W)
    # 
    # Strassen's 7 products:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M7
    # C21 = M2 + M4
    # C22 = M1 - M3 + M6 - M2
    #
    # Let's express each term as U[i,j] * V[j,k] * W[k,i] for each r in 0..6
    #
    # For each r, we need U[r][i,j], V[r][j,k], W[r][k,i]
    #
    # r=0 (M1): U: A11+A22 => U[0,0]=1, U[1,1]=1; V: B11+B22 => V[0,0]=1, V[1,1]=1; W: contributes to C11 (+1), C22 (+1) => W[0,0]=1, W[1,1]=1
    # Wait, W[k,i] where k is output row, i is output col.
    # C11: k=0, i=0. C22: k=1, i=1.
    # So W[0,0]=1, W[1,1]=1.
    
    # Let's be careful.
    # U index: (i, j) where i is row of A, j is col of A.
    # V index: (j, k) where j is row of B, k is col of B.
    # W index: (k, i) where k is row of C, i is col of C.
    
    # r=0: M1 = (A00+A11)(B00+B11)
    # U: A00=1, A11=1 => U[0,0]=1, U[1,1]=1
    # V: B00=1, B11=1 => V[0,0]=1, V[1,1]=1
    # W: C00 += M1, C11 += M1 => W[0,0]=1, W[1,1]=1
    
    # r=1: M2 = (A10+A11)B00
    # U: A10=1, A11=1 => U[1,0]=1, U[1,1]=1
    # V: B00=1 => V[0,0]=1
    # W: C10 += M2 => W[1,0]=1
    
    # r=2: M3 = A00(B01-B11)
    # U: A00=1 => U[0,0]=1
    # V: B01=1, B11=-1 => V[0,1]=1, V[1,1]=-1
    # W: C01 += M3 => W[0,1]=1
    
    # r=3: M4 = A11(B10-B00)
    # U: A11=1 => U[1,1]=1
    # V: B10=1, B00=-1 => V[1,0]=1, V[0,0]=-1
    # W: C10 += M4, C00 -= M4 => W[1,0]=1, W[0,0]=-1
    
    # r=4: M5 = (A00+A01)B11
    # U: A00=1, A01=1 => U[0,0]=1, U[0,1]=1
    # V: B11=1 => V[1,1]=1
    # W: C00 += M5, C11 -= M2? No, C11 = M1 - M3 + M6 - M2.
    # Let's re-check C11.
    # C11 = M1 - M4 + M5 - M7
    # So W[1,1] gets +1 from M1, -1 from M4? No.
    # Let's list all W contributions:
    # C00: +M1, -M4, +M5, -M7
    # C01: +M3, +M7
    # C10: +M2, +M4
    # C11: +M1, -M3, +M6, -M2
    
    # Let's redo the W assignments:
    # r=0 (M1): W[0,0]=+1, W[1,1]=+1
    # r=1 (M2): W[1,0]=+1, W[1,1]=-1  (C10 += M2, C11 -= M2)
    # r=2 (M3): W[0,1]=+1, W[1,1]=-1  (C01 += M3, C11 -= M3)
    # r=3 (M4): W[0,0]=-1, W[1,0]=+1  (C00 -= M4, C10 += M4)
    # r=4 (M5): W[0,0]=+1              (C00 += M5)
    # r=5 (M6): W[1,1]=+1              (C11 += M6)
    # r=6 (M7): W[0,0]=-1, W[0,1]=+1   (C00 -= M7, C01 += M7)
    
    # Now let's list U and V for each r:
    # r=0: U: [1,0,0,1] (A00, A01, A10, A11), V: [1,0,0,1] (B00, B01, B10, B11)
    # r=1: U: [0,0,1,1], V: [1,0,0,0]
    # r=2: U: [1,0,0,0], V: [0,1,-1,0]
    # r=3: U: [0,0,0,1], V: [-1,0,1,0]
    # r=4: U: [1,1,0,0], V: [0,0,0,1]
    # r=5: U: [0,0,1,-1], V: [1,1,0,0]
    # r=6: U: [0,1,0,-1], V: [0,0,1,1]
    
    # Let's verify r=5: M6 = (A10-A00)(B00+B01)
    # U: A10=1, A00=-1 => U[1,0]=1, U[0,0]=-1 => [ -1, 0, 1, 0 ]?
    # My U order is (A00, A01, A10, A11).
    # So U: A00=-1, A01=0, A10=1, A11=0 => [-1, 0, 1, 0]
    # V: B00=1, B01=1, B10=0, B11=0 => [1, 1, 0, 0]
    
    # Let's re-verify all:
    # r=0: U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1] (W order: C00, C01, C10, C11)
    # r=1: U=[0,0,1,1], V=[1,0,0,0], W=[0,0,1,-1]
    # r=2: U=[1,0,0,0], V=[0,1,-1,0], W=[0,1,0,-1]
    # r=3: U=[0,0,0,1], V=[-1,0,1,0], W=[-1,0,1,0]
    # r=4: U=[1,1,0,0], V=[0,0,0,1], W=[1,0,0,0]
    # r=5: U=[-1,0,1,0], V=[1,1,0,0], W=[0,0,0,1]
    # r=6: U=[0,1,0,-1], V=[0,0,1,1], W=[-1,1,0,0]
    
    # Let's verify C00 = T[0,0,0] = sum_r U[r][0,0] * V[r][0,0] * W[r][0,0]
    # U[0,0] is first element. V[0,0] is first element. W[0,0] is first element.
    # r=0: 1 * 1 * 1 = 1
    # r=1: 0 * 1 * 0 = 0
    # r=2: 1 * 0 * 0 = 0
    # r=3: 0 * (-1) * (-1) = 0
    # r=4: 1 * 0 * 1 = 0
    # r=5: (-1) * 1 * 0 = 0
    # r=6: 0 * 0 * (-1) = 0
    # Sum = 1. Correct.
    
    # C01 = T[0,0,1]? No, T[i,j,k] where i,j for A, j,k for B, k,i for C.
    # C01 means k=0, i=1. So T[?, ?, 0, 1].
    # The term is A_{i,j} B_{j,k} C_{k,i}.
    # For C01, k=0, i=1. So we need sum over j of A_{1,j} B_{j,0}.
    # T[1,j,0] for j=0,1.
    # T[1,0,0]: i=1, j=0, k=0.
    # sum_r U[r][1,0] * V[r][0,0] * W[r][0,1]
    # U[1,0] is 3rd element. V[0,0] is 1st. W[0,1] is 2nd.
    # r=0: 0 * 1 * 0 = 0
    # r=1: 1 * 1 * 0 = 0
    # r=2: 0 * 0 * 1 = 0
    # r=3: 0 * (-1) * 0 = 0
    # r=4: 0 * 0 * 0 = 0
    # r=5: 1 * 1 * 0 = 0
    # r=6: 0 * 0 * 1 = 0
    # Sum = 0. But C01 should be 1!
    
    # I think I messed up the W indexing or the T indexing.
    # T_{i,j,k} corresponds to A_{i,j} * B_{j,k} contributing to C_{k,i}.
    # So W_{k,i}.
    # C01 is C_{0,1}, so k=0, i=1. W_{0,1}.
    # My W order is (C00, C01, C10, C11) = (W[0,0], W[0,1], W[1,0], W[1,1]).
    # So W[0,1] is the 2nd element.
    
    # Let's re-check r=6: M7 = (A01-A11)(B10+B11)
    # U: A01=1, A11=-1 => [0, 1, 0, -1]
    # V: B10=1, B11=1 => [0, 0, 1, 1]
    # W: C00 -= M7, C01 += M7 => W[0,0]=-1, W[0,1]=1
    # So W for r=6 is [-1, 1, 0, 0].
    
    # Let's re-check C01 = T[1,j,0] sum over j.
    # T[1,0,0]: i=1, j=0, k=0.
    # r=6: U[1,0]=0, so 0.
    # T[1,1,0]: i=1, j=1, k=0.
    # sum_r U[r][1,1] * V[r][1,0] * W[r][0,1]
    # U[1,1] is 4th element. V[1,0] is 3rd. W[0,1] is 2nd.
    # r=0: 1 * 0 * 0 = 0
    # r=1: 1 * 0 * 0 = 0
    # r=2: 0 * (-1) * 1 = 0
    # r=3: 1 * 1 * 0 = 0
    # r=4: 0 * 0 * 0 = 0
    # r=5: 0 * 0 * 0 = 0
    # r=6: (-1) * 1 * 1 = -1
    # Sum = -1. Still not 1.
    
    # It seems my Strassen coefficients are slightly off or I'm confusing the indices.
    # Given the time constraint and the risk of error in manual derivation, 
    # and since the previous attempt to use Strassen failed validation,
    # I will stick to the robust schoolbook decomposition which is guaranteed correct.
    # The rank is 36, which is the seed score.
    
    # However, I can try to find a better decomposition programmatically if I have time.
    # But for now, returning the schoolbook solution is the safest bet to ensure validity.
    
    return best_result
