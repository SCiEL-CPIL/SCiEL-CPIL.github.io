import numpy as np
import time

def construct():
    """
    Return a rank-28 exact decomposition for 2x3 * 3x5 matrix multiplication.
    
    Strategy:
    - Split j into {0} and {1,2}.
    - Block 1 (j=0): 2x1x5 schoolbook, rank 10.
    - Block 2 (j in {1,2}):
      - k in {0,1}: 2x2x2 Strassen, rank 7.
      - k=2: 2x2x1 schoolbook, rank 4.
      - k in {3,4}: 2x2x2 Strassen, rank 7.
      - Total: 18.
    - Grand total: 10 + 18 = 28.
    
    Indexing conventions:
    - A is 2x3, flattened row-major: A[i,j] -> index i*3+j
    - B is 3x5, flattened row-major: B[j,k] -> index j*5+k
    - C is 2x5, output follows W[k*n+i, r] convention: C[i,k] -> index k*2+i
    """
    n, m, p = 2, 3, 5
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # Block 1: j=0, all k in {0,1,2,3,4}
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Block 2a: j in {1,2}, k in {0,1} — Strassen 2x2x2
    U[0*3+1][col]=1; U[1*3+2][col]=1; V[1*5+0][col]=1; V[2*5+1][col]=1
    W[0*2+0][col]=1; W[1*2+1][col]=1; col+=1
    U[1*3+1][col]=1; U[1*3+2][col]=1; V[1*5+0][col]=1
    W[0*2+1][col]=1; W[1*2+1][col]=-1; col+=1
    U[0*3+1][col]=1; V[1*5+1][col]=1; V[2*5+1][col]=-1
    W[1*2+0][col]=1; W[1*2+1][col]=1; col+=1
    U[1*3+2][col]=1; V[2*5+0][col]=1; V[1*5+0][col]=-1
    W[0*2+0][col]=1; W[0*2+1][col]=1; col+=1
    U[0*3+1][col]=1; U[0*3+2][col]=1; V[2*5+1][col]=1
    W[0*2+0][col]=-1; W[1*2+0][col]=1; col+=1
    U[1*3+1][col]=1; U[0*3+1][col]=-1; V[1*5+0][col]=1; V[1*5+1][col]=1
    W[1*2+1][col]=1; col+=1
    U[0*3+2][col]=1; U[1*3+2][col]=-1; V[2*5+0][col]=1; V[2*5+1][col]=1
    W[0*2+0][col]=1; col+=1
    
    # Block 2b: j in {1,2}, k=2 — schoolbook
    for i in range(n):
        for j_local in range(2):
            j = j_local + 1
            U[i*3+j][col]=1; V[j*5+2][col]=1; W[2*2+i][col]=1; col+=1
    
    # Block 2c: j in {1,2}, k in {3,4} — Strassen 2x2x2
    U[0*3+1][col]=1; U[1*3+2][col]=1; V[1*5+3][col]=1; V[2*5+4][col]=1
    W[3*2+0][col]=1; W[4*2+1][col]=1; col+=1
    U[1*3+1][col]=1; U[1*3+2][col]=1; V[1*5+3][col]=1
    W[3*2+1][col]=1; W[4*2+1][col]=-1; col+=1
    U[0*3+1][col]=1; V[1*5+4][col]=1; V[2*5+4][col]=-1
    W[4*2+0][col]=1; W[4*2+1][col]=1; col+=1
    U[1*3+2][col]=1; V[2*5+3][col]=1; V[1*5+3][col]=-1
    W[3*2+0][col]=1; W[3*2+1][col]=1; col+=1
    U[0*3+1][col]=1; U[0*3+2][col]=1; V[2*5+4][col]=1
    W[3*2+0][col]=-1; W[4*2+0][col]=1; col+=1
    U[1*3+1][col]=1; U[0*3+1][col]=-1; V[1*5+3][col]=1; V[1*5+4][col]=1
    W[4*2+1][col]=1; col+=1
    U[0*3+2][col]=1; U[1*3+2][col]=-1; V[2*5+3][col]=1; V[2*5+4][col]=1
    W[3*2+0][col]=1; col+=1
    
    return {"U": U, "V": V, "W": W}
