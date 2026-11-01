import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank 22 decomposition
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: M_{2,1,4} (j=0)
    # A indices: i in {0,1}, j=0 -> U[3*i]
    # B indices: j=0, k in {0,1,2,3} -> V[4*0 + k] = V[k]
    # C indices: i in {0,1}, k in {0,1,2,3} -> W[2*k + i]
    r = 0
    for i in range(2):
        for k in range(4):
            U[3*i][r] = 1
            V[k][r] = 1
            W[2*k + i][r] = 1
            r += 1
    # r is now 8

    # Part 2: M_{2,2,4} (j in {1,2})
    # A indices: i in {0,1}, j in {1,2} -> U[3*i + j]
    # B indices: j in {1,2}, k in {0,1,2,3} -> V[4*j + k]
    # C indices: i in {0,1}, k in {0,1,2,3} -> W[2*k + i]
    
    # Block A: k in {0,1}
    # Strassen for 2x2x2
    # M1: (A00+A11)(B00+B11) -> C00+C11
    # A00=U[1], A11=U[5]
    # B00=V[4], B11=V[9]
    # C00=W[0], C11=W[3]
    U[1][r] = 1; U[5][r] = 1
    V[4][r] = 1; V[9][r] = 1
    W[0][r] = 1; W[3][r] = 1
    r += 1
    
    # M2: (A10+A11)B00 -> C10+C11
    # A10=U[4], A11=U[5]
    # B00=V[4]
    # C10=W[1], C11=W[3]
    U[4][r] = 1; U[5][r] = 1
    V[4][r] = 1
    W[1][r] = 1; W[3][r] = 1
    r += 1
    
    # M3: A00(B01-B11) -> C01+C11
    # A00=U[1]
    # B01=V[5], B11=V[9]
    # C01=W[2], C11=W[3]
    U[1][r] = 1
    V[5][r] = 1; V[9][r] = -1
    W[2][r] = 1; W[3][r] = 1
    r += 1
    
    # M4: A11(B10-B00) -> C00+C10
    # A11=U[5]
    # B10=V[8], B00=V[4]
    # C00=W[0], C10=W[1]
    U[5][r] = 1
    V[8][r] = 1; V[4][r] = -1
    W[0][r] = 1; W[1][r] = 1
    r += 1
    
    # M5: (A00+A01)B11 -> -C00+C01
    # A00=U[1], A01=U[2]
    # B11=V[9]
    # C00=W[0], C01=W[2]
    U[1][r] = 1; U[2][r] = 1
    V[9][r] = 1
    W[0][r] = -1; W[2][r] = 1
    r += 1
    
    # M6: (A10-A00)(B00+B01) -> C11
    # A10=U[4], A00=U[1]
    # B00=V[4], B01=V[5]
    # C11=W[3]
    U[4][r] = 1; U[1][r] = -1
    V[4][r] = 1; V[5][r] = 1
    W[3][r] = 1
    r += 1
    
    # M7: (A01-A11)(B10+B11) -> C00
    # A01=U[2], A11=U[5]
    # B10=V[8], B11=V[9]
    # C00=W[0]
    U[2][r] = 1; U[5][r] = -1
    V[8][r] = 1; V[9][r] = 1
    W[0][r] = 1
    r += 1
    
    # Block B: k in {2,3}
    # Same A, different B and C
    # B indices: j in {1,2}, k in {2,3}
    # B00 (for this block) = V[4*1+2] = V[6]
    # B01 = V[4*1+3] = V[7]
    # B10 = V[4*2+2] = V[10]
    # B11 = V[4*2+3] = V[11]
    # C indices: k in {2,3}
    # C00 (for this block) = W[2*2+0] = W[4]
    # C01 = W[2*3+0] = W[6]
    # C10 = W[2*2+1] = W[5]
    # C11 = W[2*3+1] = W[7]
    
    # M1
    U[1][r] = 1; U[5][r] = 1
    V[6][r] = 1; V[11][r] = 1
    W[4][r] = 1; W[7][r] = 1
    r += 1
    
    # M2
    U[4][r] = 1; U[5][r] = 1
    V[6][r] = 1
    W[5][r] = 1; W[7][r] = 1
    r += 1
    
    # M3
    U[1][r] = 1
    V[7][r] = 1; V[11][r] = -1
    W[6][r] = 1; W[7][r] = 1
    r += 1
    
    # M4
    U[5][r] = 1
    V[10][r] = 1; V[6][r] = -1
    W[4][r] = 1; W[5][r] = 1
    r += 1
    
    # M5
    U[1][r] = 1; U[2][r] = 1
    V[11][r] = 1
    W[4][r] = -1; W[6][r] = 1
    r += 1
    
    # M6
    U[4][r] = 1; U[1][r] = -1
    V[6][r] = 1; V[7][r] = 1
    W[7][r] = 1
    r += 1
    
    # M7
    U[2][r] = 1; U[5][r] = -1
    V[10][r] = 1; V[11][r] = 1
    W[4][r] = 1
    r += 1
    
    return {"U": U, "V": V, "W": W}
