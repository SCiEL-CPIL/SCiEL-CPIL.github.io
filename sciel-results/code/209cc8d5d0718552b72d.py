import time
import numpy as np

def construct():
    n, m, p = 3, 4, 4
    dim_u = n * m  # 12
    dim_v = m * p  # 16
    dim_w = p * n  # 12
    rank = 44
    
    U = [[0] * rank for _ in range(dim_u)]
    V = [[0] * rank for _ in range(dim_v)]
    W = [[0] * rank for _ in range(dim_w)]
    
    col = 0
    
    def add_strassen(X_u, Y_v, Z_w, col):
        U[X_u[0]][col] += 1; U[X_u[3]][col] += 1
        V[Y_v[0]][col] += 1; V[Y_v[3]][col] += 1
        W[Z_w[0]][col] += 1; W[Z_w[3]][col] += 1
        col += 1
        U[X_u[2]][col] += 1; U[X_u[3]][col] += 1
        V[Y_v[0]][col] += 1
        W[Z_w[2]][col] += 1; W[Z_w[3]][col] -= 1
        col += 1
        U[X_u[0]][col] += 1
        V[Y_v[1]][col] += 1; V[Y_v[3]][col] -= 1
        W[Z_w[1]][col] += 1; W[Z_w[3]][col] += 1
        col += 1
        U[X_u[3]][col] += 1
        V[Y_v[2]][col] += 1; V[Y_v[0]][col] -= 1
        W[Z_w[0]][col] += 1; W[Z_w[2]][col] += 1
        col += 1
        U[X_u[0]][col] += 1; U[X_u[1]][col] += 1
        V[Y_v[3]][col] += 1
        W[Z_w[0]][col] -= 1; W[Z_w[1]][col] += 1
        col += 1
        U[X_u[2]][col] += 1; U[X_u[0]][col] -= 1
        V[Y_v[0]][col] += 1; V[Y_v[1]][col] += 1
        W[Z_w[3]][col] += 1
        col += 1
        U[X_u[1]][col] += 1; U[X_u[3]][col] -= 1
        V[Y_v[2]][col] += 1; V[Y_v[3]][col] += 1
        W[Z_w[0]][col] += 1
        col += 1
        return col
    
    col = add_strassen([0, 1, 4, 5], [0, 1, 4, 5], [0, 3, 1, 4], col)
    col = add_strassen([2, 3, 6, 7], [8, 9, 12, 13], [0, 3, 1, 4], col)
    col = add_strassen([0, 1, 4, 5], [2, 3, 6, 7], [6, 9, 7, 10], col)
    col = add_strassen([2, 3, 6, 7], [10, 11, 14, 15], [6, 9, 7, 10], col)
    
    for j in range(4):
        u_idx = 8 + j
        U[u_idx][col] += 1; V[j * 4 + 0][col] += 1; W[2][col] += 1; col += 1
        U[u_idx][col] += 1; V[j * 4 + 1][col] += 1; W[5][col] += 1; col += 1
    for j in range(4):
        u_idx = 8 + j
        U[u_idx][col] += 1; V[j * 4 + 2][col] += 1; W[8][col] += 1; col += 1
        U[u_idx][col] += 1; V[j * 4 + 3][col] += 1; W[11][col] += 1; col += 1
    
    return {"U": U, "V": V, "W": W}
