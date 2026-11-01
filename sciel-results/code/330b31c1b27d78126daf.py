"""Strassen-based rank-21 decomposition for 2x2 * 2x6 matrix multiplication."""
import time

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START

def _strassen_222():
    """U,V,W (each 4x7) for 2x2*2x2 via Strassen.

    U index = i*2+j (A[i,j]); V index = j*2+k (B[j,k]); W index = k*2+i (C[i,k]).
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    # A: a00=0,a01=1,a10=2,a11=3 ; B: b00=0,b01=1,b10=2,b11=3
    # C local: C00(i0k0),C01(i0k1),C10(i1k0),C11(i1k1)
    # W local index = k*2+i : C00->0, C01->2, C10->1, C11->3
    def add(U_,i,c,v): U_[i][c]+=v
    # M1=(a00+a11)(b00+b11) -> C00+=1,C11+=1
    U[0][0]+=1; U[3][0]+=1
    V[0][0]+=1; V[3][0]+=1
    W[0][0]+=1; W[3][0]+=1
    # M2=(a10+a11)b00 -> C10+=1
    U[2][1]+=1; U[3][1]+=1
    V[0][1]+=1
    W[1][1]+=1
    # M3=a00(b01-b11) -> C01+=1
    U[0][2]+=1
    V[1][2]+=1; V[3][2]+=-1
    W[2][2]+=1
    # M4=a11(b10-b00) -> C00+=1,C10+=1
    U[3][3]+=1
    V[2][3]+=1; V[0][3]+=-1
    W[0][3]+=1; W[1][3]+=1
    # M5=(a00+a01)b11 -> C00-=1,C01+=1
    U[0][4]+=1; U[1][4]+=1
    V[3][4]+=1
    W[0][4]+=-1; W[2][4]+=1
    # M6=(a10-a00)(b00+b01) -> C11+=1
    U[2][5]+=1; U[0][5]+=-1
    V[0][5]+=1; V[1][5]+=1
    W[3][5]+=1
    # M7=(a01-a11)(b10+b11) -> C00+=1
    U[1][6]+=1; U[3][6]+=-1
    V[2][6]+=1; V[3][6]+=1
    W[0][6]+=1
    return U, V, W


def construct():
    n, m, p = DIMENSIONS  # 2,2,6
    rank = 21
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(12)]
    W = [[0]*rank for _ in range(12)]
    sU, sV, sW = _strassen_222()
    for t in range(3):
        base = 7*t
        for i in range(4):
            for c in range(7):
                if sU[i][c]:
                    U[i][base+c] += sU[i][c]
        # B local->global (j*6+k): b00=2t,b01=2t+1,b10=6+2t,b11=6+2t+1
        gB = [2*t, 2*t+1, 6+2*t, 6+2*t+1]
        for loc in range(4):
            for c in range(7):
                if sV[loc][c]:
                    V[gB[loc]][base+c] += sV[loc][c]
        # C local->global (k*2+i): C00=4t,C10=4t+1,C01=4t+2,C11=4t+3
        gW = [4*t, 4*t+1, 4*t+2, 4*t+3]
        for loc in range(4):
            for c in range(7):
                if sW[loc][c]:
                    W[gW[loc]][base+c] += sW[loc][c]
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
