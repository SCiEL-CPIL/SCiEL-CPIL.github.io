"""Matrix multiplication 2x2x9 rank-minimal tensor decomposition."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 120.0
    n, m, p = DIMENSIONS

    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(total - 1) > 1e-9:
                        return False
        return True

    def strassen_2x2x2():
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
        U[2][1]=1; U[3][1]=1; V[0][1]=1; W[1][1]=1
        U[0][2]=1; V[2][2]=1; V[3][2]=-1; W[1][2]=1
        U[3][3]=1; V[1][3]=1; V[0][3]=-1; W[0][3]=-1; W[2][3]=1
        U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=1
        U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[2][5]=1; W[3][5]=1
        U[1][6]=1; U[3][6]=-1; V[1][6]=1; V[3][6]=1; W[2][6]=1
        return U, V, W

    def schoolbook_2x2x1():
        U = [[0]*4 for _ in range(4)]
        V = [[0]*4 for _ in range(2)]
        W = [[0]*4 for _ in range(2)]
        U[0][0]=1; V[0][0]=1; W[0][0]=1
        U[1][1]=1; V[1][1]=1; W[0][1]=1
        U[2][2]=1; V[0][2]=1; W[1][2]=1
        U[3][3]=1; V[1][3]=1; W[1][3]=1
        return U, V, W

    def build_rank32():
        U_s, V_s, W_s = strassen_2x2x2()
        U_1, V_1, W_1 = schoolbook_2x2x1()
        rank = 32
        U_full = [[0]*rank for _ in range(4)]
        V_full = [[0]*rank for _ in range(18)]
        W_full = [[0]*rank for _ in range(18)]
        col_off = 0
        for block in range(4):
            for c in range(7):
                for i in range(4):
                    U_full[i][col_off+c] = U_s[i][c]
                for j in range(2):
                    for k_local in range(2):
                        k_global = 2*block + k_local
                        V_full[j*9 + k_global][col_off+c] = V_s[j*2 + k_local][c]
                for k_local in range(2):
                    for i in range(2):
                        k_global = 2*block + k_local
                        W_full[k_global*2 + i][col_off+c] = W_s[k_local*2 + i][c]
            col_off += 7
        for c in range(4):
            for i in range(4):
                U_full[i][col_off+c] = U_1[i][c]
            for j in range(2):
                V_full[j*9 + 8][col_off+c] = V_1[j][c]
            for i in range(2):
                W_full[8*2 + i][col_off+c] = W_1[i][c]
        return {"U": U_full, "V": V_full, "W": W_full}

    def schoolbook_full():
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Try rank-32
    if time.time() < deadline - 10:
        r32 = build_rank32()
        if verify(r32["U"], r32["V"], r32["W"]):
            return r32

    return schoolbook_full()

# EVOLVE-BLOCK-END
