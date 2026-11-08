import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # A is n x m = 2x3, B is m x p = 3x9, C is n x p = 2x9
    # U shape: (n*m, rank) = (6, rank)
    # V shape: (m*p, rank) = (27, rank)
    # W shape: (p*n, rank) = (18, rank)
    # Constraint: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Strategy:
    # - 4 blocks of 2x2x2 (Strassen, rank 7 each) for k=0..7 -> rank 28
    # - 1 block of 2x2x1 (rank 4) for k=8 -> rank 4
    # - 1 block of 2x1x9 (rank 9) for j=2, all k -> rank 9
    # Total rank = 28 + 4 + 9 = 41
    
    rank = 41
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # 4 blocks of 2x2x2 using Strassen (rank 7 each)
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        v_e = 0 * p + k0
        v_f = 0 * p + k1
        v_g = 1 * p + k0
        v_h = 1 * p + k1
        
        u_a = 0 * m + 0
        u_b = 0 * m + 1
        u_c = 1 * m + 0
        u_d = 1 * m + 1
        
        # Strassen 7 terms
        # Term 0: (A00+A11)(B00+B11)
        U[u_a][col] += 1
        U[u_d][col] += 1
        V[v_e][col] += 1
        V[v_h][col] += 1
        W[k0 * n + 0][col] += 1
        W[k1 * n + 0][col] += 1
        W[k0 * n + 1][col] += 1
        W[k1 * n + 1][col] -= 1
        
        # Term 1: A10*(B00+B01)
        U[u_c][col+1] += 1
        V[v_e][col+1] += 1
        V[v_f][col+1] += 1
        W[k0 * n + 1][col+1] += 1
        W[k1 * n + 1][col+1] += 1
        
        # Term 2: A00*(B10-B11)
        U[u_a][col+2] += 1
        V[v_g][col+2] += 1
        V[v_h][col+2] -= 1
        W[k0 * n + 0][col+2] += 1
        W[k1 * n + 0][col+2] -= 1
        
        # Term 3: A11*(B01-B00)
        U[u_d][col+3] += 1
        V[v_f][col+3] += 1
        V[v_e][col+3] -= 1
        W[k0 * n + 1][col+3] -= 1
        W[k1 * n + 1][col+3] += 1
        
        # Term 4: (A00+A01)*B11
        U[u_a][col+4] += 1
        U[u_b][col+4] += 1
        V[v_h][col+4] += 1
        W[k0 * n + 0][col+4] += 1
        W[k1 * n + 0][col+4] += 1
        
        # Term 5: (A10-A00)*(B00+B10)
        U[u_c][col+5] += 1
        U[u_a][col+5] -= 1
        V[v_e][col+5] += 1
        V[v_g][col+5] += 1
        W[k0 * n + 1][col+5] += 1
        W[k1 * n + 1][col+5] -= 1
        
        # Term 6: (A01-A11)*(B11+B01)
        U[u_b][col+6] += 1
        U[u_d][col+6] -= 1
        V[v_f][col+6] += 1
        V[v_h][col+6] += 1
        W[k0 * n + 0][col+6] -= 1
        W[k1 * n + 0][col+6] += 1
        
        col += 7
    
    # 1 block of 2x2x1 for k=8 (rank 4)
    k8 = 8
    v_e8 = 0 * p + k8
    v_f8 = 1 * p + k8
    
    u_a8 = 0 * m + 0
    u_b8 = 0 * m + 1
    u_c8 = 1 * m + 0
    u_d8 = 1 * m + 1
    
    # C[0,8] = A[0,0]*B[0,8] + A[0,1]*B[1,8]
    # C[1,8] = A[1,0]*B[0,8] + A[1,1]*B[1,8]
    # 4 scalar multiplications
    U[u_a8][col] += 1
    V[v_e8][col] += 1
    W[k8 * n + 0][col] += 1
    
    U[u_b8][col+1] += 1
    V[v_f8][col+1] += 1
    W[k8 * n + 0][col+1] += 1
    
    U[u_c8][col+2] += 1
    V[v_e8][col+2] += 1
    W[k8 * n + 1][col+2] += 1
    
    U[u_d8][col+3] += 1
    V[v_f8][col+3] += 1
    W[k8 * n + 1][col+3] += 1
    
    col += 4
    
    # 1 block of 2x1x9 for j=2 (rank 9)
    # C[i,k] = A[i,2]*B[2,k] for i=0,1 and k=0..8
    # Use 9 terms, one for each k
    u_j2_0 = 0 * m + 2  # A[0,2]
    u_j2_1 = 1 * m + 2  # A[1,2]
    
    for k in range(9):
        v_j2 = 2 * p + k  # B[2,k]
        
        U[u_j2_0][col] += 1
        U[u_j2_1][col] += 1
        V[v_j2][col] += 1
        W[k * n + 0][col] += 1
        W[k * n + 1][col] += 1
        
        col += 1
    
    return {"U": U, "V": V, "W": W}
