import time
import itertools
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def _construct_strassen_222():
    """
    Returns U, V, W for 2x2x2 matrix multiplication with rank 7.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indices: i in 0..1, j in 0..1, k in 0..1.
    U[i*2+j, r], V[j*2+k, r], W[k*2+i, r].
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # Strassen terms
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1 # C00, C11
    # Wait, W[k*2+i, r].
    # C00 is k=0, i=0 -> W[0, r]. C11 is k=1, i=1 -> W[3, r].
    # C00 += M1, C11 += M1.
    # C01 += M1? No.
    
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11) -> C00, C11
    # M2 = (A10+A11)B00 -> C10, C11
    # M3 = A00(B10-B11) -> C01, C11
    # M4 = A11(B01-B00) -> C00, C10
    # M5 = (A00+A01)B11 -> C01, C00
    # M6 = (A10-A00)(B00+B01) -> C10, C11
    # M7 = (A01-A11)(B10+B11) -> C01, C00
    
    # Let's map carefully.
    # r=0: M1
    U[0][0]=1; U[3][0]=1
    V[0][0]=1; V[3][0]=1
    W[0][0]=1; W[3][0]=1 # C00, C11
    
    # r=1: M2
    U[2][1]=1; U[3][1]=1 # A10, A11
    V[0][1]=1 # B00
    W[2][1]=1; W[3][1]=1 # C10, C11
    
    # r=2: M3
    U[0][2]=1 # A00
    V[2][2]=1; V[3][2]=-1 # B10, -B11
    W[1][2]=1; W[3][2]=-1 # C01, -C11
    
    # r=3: M4
    U[3][3]=1 # A11
    V[1][3]=1; V[0][3]=-1 # B01, -B00
    W[0][3]=-1; W[2][3]=1 # -C00, C10
    
    # r=4: M5
    U[0][4]=1; U[1][4]=1 # A00, A01
    V[3][4]=1 # B11
    W[1][4]=1; W[0][4]=1 # C01, C00
    
    # r=5: M6
    U[2][5]=1; U[0][5]=-1 # A10, -A00
    V[0][5]=1; V[1][5]=1 # B00, B01
    W[2][5]=1; W[3][5]=-1 # C10, -C11
    
    # r=6: M7
    U[1][6]=1; U[3][6]=-1 # A01, -A11
    V[2][6]=1; V[3][6]=1 # B10, B11
    W[1][6]=1; W[0][6]=-1 # C01, -C00
    
    return U, V, W

def _construct_trivial_221():
    """
    Returns U, V, W for 2x2x1 matrix multiplication with rank 4.
    i in 0..1, j in 0..1, k=0 (local).
    U shape (4, 4), V shape (2, 4), W shape (2, 4).
    Note: V index is j*1+k = j. W index is k*2+i = i.
    """
    rank = 4
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(2)]
    W = [[0]*rank for _ in range(2)]
    
    # M1 = A00 B0
    U[0][0]=1; V[0][0]=1; W[0][0]=1
    # M2 = A01 B1
    U[1][1]=1; V[1][1]=1; W[0][1]=1
    # M3 = A10 B0
    U[2][2]=1; V[0][2]=1; W[1][2]=1
    # M4 = A11 B1
    U[3][3]=1; V[1][3]=1; W[1][3]=1
    
    return U, V, W

def _construct_trivial_213():
    """
    Returns U, V, W for 2x1x3 matrix multiplication with rank 6.
    i in 0..1, j=0 (local), k in 0..2.
    U shape (2, 6), V shape (3, 6), W shape (6, 6).
    U index: i*1+j = i.
    V index: j*3+k = k.
    W index: k*2+i.
    """
    rank = 6
    U = [[0]*rank for _ in range(2)]
    V = [[0]*rank for _ in range(3)]
    W = [[0]*rank for _ in range(6)]
    
    col = 0
    for i in range(2):
        for k in range(3):
            U[i][col] = 1
            V[k][col] = 1
            W[k*2+i][col] = 1
            col += 1
            
    return U, V, W

def _build_block_233():
    """
    Builds U, V, W for 2x3x3 with rank 17.
    U shape (6, 17), V shape (9, 17), W shape (6, 17).
    """
    U1, V1, W1 = _construct_strassen_222()
    U2, V2, W2 = _construct_trivial_221()
    U3, V3, W3 = _construct_trivial_213()
    
    rank = 17
    U = [[0]*rank for _ in range(6)]
    V = [[0]*rank for _ in range(9)]
    W = [[0]*rank for _ in range(6)]
    
    # Block 1: 2x2x2 (j in 0,1; k in 0,1)
    # U indices: i*3+j. i in 0,1; j in 0,1.
    # V indices: j*3+k. j in 0,1; k in 0,1.
    # W indices: k*2+i. k in 0,1; i in 0,1.
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U[i*3+j][r] = U1[i*2+j][r]
        for j in range(2):
            for k in range(2):
                V[j*3+k][r] = V1[j*2+k][r]
        for k in range(2):
            for i in range(2):
                W[k*2+i][r] = W1[k*2+i][r]
                
    # Block 2: 2x2x1 (j in 0,1; k=2)
    # U indices: i*3+j.
    # V indices: j*3+2.
    # W indices: 2*2+i = 4+i.
    offset = 7
    for r in range(4):
        for i in range(2):
            for j in range(2):
                U[i*3+j][offset+r] = U2[i*2+j][r]
        for j in range(2):
            V[j*3+2][offset+r] = V2[j][r]
        for i in range(2):
            W[4+i][offset+r] = W2[i][r]
            
    # Block 3: 2x1x3 (j=2; k in 0,1,2)
    # U indices: i*3+2.
    # V indices: 2*3+k = 6+k.
    # W indices: k*2+i.
    offset = 11
    for r in range(6):
        for i in range(2):
            U[i*3+2][offset+r] = U3[i][r]
        for k in range(3):
            V[6+k][offset+r] = V3[k][r]
        for k in range(3):
            for i in range(2):
                W[k*2+i][offset+r] = W3[k*2+i][r]
                
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    U_block, V_block, W_block = _build_block_233()
    rank_block = 17
    total_rank = rank_block * 2
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    # Block 1: k in 0..2
    for r in range(rank_block):
        for i in range(n):
            for j in range(m):
                U[i*m+j][r] = U_block[i*m+j][r]
        for j in range(m):
            for k in range(3):
                V[j*p+k][r] = V_block[j*3+k][r]
        for k in range(3):
            for i in range(n):
                W[k*n+i][r] = W_block[k*n+i][r]
                
    # Block 2: k in 3..5
    for r in range(rank_block):
        for i in range(n):
            for j in range(m):
                U[i*m+j][rank_block+r] = U_block[i*m+j][r]
        for j in range(m):
            for k in range(3):
                V[j*p+3+k][rank_block+r] = V_block[j*3+k][r]
        for k in range(3):
            for i in range(n):
                W[(3+k)*n+i][rank_block+r] = W_block[k*n+i][r]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
