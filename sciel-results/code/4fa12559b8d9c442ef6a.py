import time
import numpy as np

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # Define terms: each term is (U_nonzero, V_nonzero, W_nonzero)
    # U_nonzero: list of (u_idx, coeff)
    # V_nonzero: list of (v_idx, coeff)
    # W_nonzero: list of (w_idx, coeff)
    
    terms = []
    
    # Helper to add a simple term: U[i,j]=a, V[j,k]=b, W[k,i]=c
    # This contributes a*b*c to C[i,k]
    def simple_term(i, j, k, a=1, b=1, c=1):
        terms.append([[(i*m+j, a)], [(j*p+k, b)], [(k*n+i, c)]])
    
    # For 3x3 x 3x4 subproblem (columns 0-3):
    # Part 1: A[0:2,0:2] * B[0:2,0:4] using Strassen (14 terms)
    
    # Strassen for A[0:2,0:2] * B[0:2,0:2] (columns 0-1):
    # M0 = A[0,0]*B[0,0]
    simple_term(0, 0, 0, 1, 1, 1)  # C[0,0] += A[0,0]*B[0,0]
    # But Strassen combines these... 
    
    # Let me just use schoolbook for simplicity first, then optimize.
    
    # Schoolbook for 3x3 x 3x8: rank 72
    # For each i,j,k: C[i,k] += A[i,j]*B[j,k]
    
    # Let me just do schoolbook to get a valid answer, then try to improve.
    
    rank = n * m * p  # 72
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = i*m*p + j*p + k
                U[i*m+j][s] = 1
                V[j*p+k][s] = 1
                W[k*n+i][s] = 1
    
    return {"U": U, "V": V, "W": W}
