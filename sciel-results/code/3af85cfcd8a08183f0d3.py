import numpy as np
import time

DIMENSIONS = [2, 2, 11]

def construct():
    n, m, p = DIMENSIONS
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    for s in range(5):
        k0 = 2 * s
        
        # T1
        U[0][col] = 1; U[3][col] = 1
        V[0*11 + k0][col] = 1; V[1*11 + k0 + 1][col] = 1
        W[(k0+0)*2 + 0][col] = 1; W[(k0+1)*2 + 1][col] = 1
        col += 1
        
        # T2
        U[2][col] = 1; U[3][col] = 1
        V[0*11 + k0][col] = 1
        W[(k0+0)*2 + 1][col] = 1; W[(k0+1)*2 + 1][col] = -1
        col += 1
        
        # T3
        U[0][col] = 1
        V[0*11 + k0 + 1][col] = 1; V[1*11 + k0][col] = -1
        W[(k0+1)*2 + 0][col] = 1; W[(k0+1)*2 + 1][col] = 1
        col += 1
        
        # T4
        U[3][col] = 1
        V[0*11 + k0][col] = -1; V[1*11 + k0][col] = 1
        W[(k0+0)*2 + 0][col] = 1; W[(k0+0)*2 + 1][col] = 1
        col += 1
        
        # T5
        U[0][col] = 1; U[1][col] = 1
        V[1*11 + k0 + 1][col] = 1
        W[(k0+0)*2 + 0][col] = -1; W[(k0+1)*2 + 0][col] = 1
        col += 1
        
        # T6
        U[2][col] = 1; U[0][col] = -1
        V[0*11 + k0][col] = 1; V[0*11 + k0 + 1][col] = 1
        W[(k0+1)*2 + 1][col] = 1
        col += 1
        
        # T7
        U[1][col] = 1; U[3][col] = -1
        V[1*11 + k0][col] = 1; V[1*11 + k0 + 1][col] = 1
        W[(k0+0)*2 + 0][col] = 1
        col += 1

    k0 = 10
    # T8
    U[0][col] = 1
    V[0*11 + k0][col] = 1
    W[k0*2 + 0][col] = 1
    col += 1
    
    # T9
    U[1][col] = 1
    V[1*11 + k0][col] = 1
    W[k0*2 + 0][col] = 1
    col += 1
    
    # T10
    U[2][col] = 1
    V[0*11 + k0][col] = 1
    W[k0*2 + 1][col] = 1
    col += 1
    
    # T11
    U[3][col] = 1
    V[1*11 + k0][col] = 1
    W[k0*2 + 1][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
