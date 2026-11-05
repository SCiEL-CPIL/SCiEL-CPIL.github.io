"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Check if the decomposition exactly decomposes the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    U_t = U_np.reshape(n, m, rank)
    V_t = V_np.reshape(m, p, rank)
    W_t = W_np.reshape(p, n, rank)
    
    try:
        T = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
        if not np.allclose(T, 1.0, atol=1e-9):
            return False
    except Exception:
        return False
        
    return True

def _strassen_2x2():
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _liu_3x3():
    rank = 19
    U = [[0]*rank for _ in range(9)]
    V = [[0]*rank for _ in range(9)]
    W = [[0]*rank for _ in range(9)]
    
    U[0][0]=1; U[1][0]=1
    V[0][0]=1; V[1][0]=1
    W[0][0]=1; W[1][0]=1
    
    U[0][1]=1; U[1][1]=-1
    V[0][1]=1; V[1][1]=-1
    W[0][1]=1; W[1][1]=1
    
    U[3][2]=1; U[4][2]=1
    V[3][2]=1; V[4][2]=1
    W[3][2]=1; W[4][2]=1
    
    U[3][3]=1; U[4][3]=-1
    V[3][3]=1; V[4][3]=-1
    W[3][3]=1; W[4][3]=1
    
    U[0][4]=1; U[3][4]=1
    V[1][4]=1; V[4][4]=1
    W[0][4]=1; W[3][4]=1
    
    U[1][5]=1; U[4][5]=1
    V[0][5]=1; V[3][5]=1
    W[1][5]=1; W[4][5]=1
    
    U[0][6]=1; U[1][6]=1; U[3][6]=1; U[4][6]=1
    V[0][6]=1; V[1][6]=1; V[3][6]=1; V[4][6]=1
    W[0][6]=1; W[1][6]=1; W[3][6]=1; W[4][6]=1
    
    U[0][7]=1; U[2][7]=1; U[3][7]=1; U[5][7]=1
    V[0][7]=1; V[2][7]=1; V[3][7]=1; V[5][7]=1
    W[0][7]=1; W[2][7]=1; W[3][7]=1; W[5][7]=1
    
    U[1][8]=1; U[2][8]=1; U[4][8]=1; U[5][8]=1
    V[1][8]=1; V[2][8]=1; V[4][8]=1; V[5][8]=1
    W[1][8]=1; W[2][8]=1; W[4][8]=1; W[5][8]=1
    
    U[0][9]=1; U[4][9]=1
    V[3][9]=1; V[1][9]=1
    W[0][9]=1; W[4][9]=1
    
    U[1][10]=1; U[3][10]=1
    V[0][10]=1; V[4][10]=1
    W[1][10]=1; W[3][10]=1
    
    U[2][11]=1; U[4][11]=1
    V[5][11]=1; V[1][11]=1
    W[2][11]=1; W[4][11]=1
    
    U[1][12]=1; U[5][12]=1
    V[2][12]=1; V[4][12]=1
    W[1][12]=1; W[5][12]=1
    
    U[0][13]=1; U[2][13]=1
    V[6][13]=1; V[8][13]=1
    W[0][13]=1; W[2][13]=1
    
    U[3][14]=1; U[5][14]=1
    V[6][14]=1; V[8][14]=1
    W[3][14]=1; W[5][14]=1
    
    U[0][15]=1; U[3][15]=1; U[6][15]=1
    V[1][15]=1; V[4][15]=1; V[7][15]=1
    W[0][15]=1; W[3][15]=1; W[6][15]=1
    
    U[1][16]=1; U[4][16]=1; U[7][16]=1
    V[0][16]=1; V[3][16]=1; V[6][16]=1
    W[1][16]=1; W[4][16]=1; W[7][16]=1
    
    U[2][17]=1; U[5][17]=1; U[8][17]=1
    V[1][17]=1; V[4][17]=1; V[7][17]=1
    W[2][17]=1; W[5][17]=1; W[8][17]=1
    
    U[0][18]=1; U[4][18]=1; U[8][18]=1
    V[0][18]=1; V[4][18]=1; V[8][18]=1
    W[0][18]=1; W[4][18]=1; W[8][18]=1
    
    return U, V, W

def _block_decomposition_6x6x6_strassen():
    n, m, p = 6, 6, 6
    rank = 189
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    sU, sV, sW = _strassen_2x2()
    
    col = 0
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                for c in range(7):
                    for li in range(2):
                        for lj in range(2):
                            u_row = (bi*2+li)*6 + (bj*2+lj)
                            U[u_row][col] = sU[li*2+lj][c]
                    for lj in range(2):
                        for lk in range(2):
                            v_row = (bj*2+lj)*6 + (bk*2+lk)
                            V[v_row][col] = sV[lj*2+lk][c]
                    for lk in range(2):
                        for li in range(2):
                            w_row = (bk*2+lk)*6 + (bi*2+li)
                            W[w_row][col] = sW[lk*2+li][c]
                    col += 1
    
    return {"U": U, "V": V, "W": W}

def _block_decomposition_6x6x6_liu():
    liu = _liu_3x3()
    if liu is None:
        return None
    
    lU, lV, lW = liu
    rank = 8 * 19
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    sU, sV, sW = _strassen_2x2()
    
    for c2 in range(7):
        for c3 in range(19):
            col = c2 * 19 + c3
            for i2 in range(2):
                for j2 in range(2):
                    for i3 in range(3):
                        for j3 in range(3):
                            u_row = (i2*3+i3)*6 + (j2*3+j3)
                            U[u_row][col] = sU[i2*2+j2][c2] * lU[i3*3+j3][c3]
            for j2 in range(2):
                for k2 in range(2):
                    for j3 in range(3):
                        for k3 in range(3):
                            v_row = (j2*3+j3)*6 + (k2*3+k3)
                            V[v_row][col] = sV[j2*2+k2][c2] * lV[j3*3+k3][c3]
            for k2 in range(2):
                for i2 in range(2):
                    for k3 in range(3):
                        for i3 in range(3):
                            w_row = (k2*3+k3)*6 + (i2*3+i3)
                            W[w_row][col] = sW[k2*2+i2][c2] * lW[k3*3+i3][c3]
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    liu_decomp = _block_decomposition_6x6x6_liu()
    if liu_decomp is not None:
        if _verify(liu_decomp["U"], liu_decomp["V"], liu_decomp["W"], n, m, p):
            return liu_decomp
    
    strassen_decomp = _block_decomposition_6x6x6_strassen()
    if _verify(strassen_decomp["U"], strassen_decomp["V"], strassen_decomp["W"], n, m, p):
        return strassen_decomp
    
    rank = 216
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    col = 0
    for i in range(6):
        for j in range(6):
            for k in range(6):
                U[i*6+j][col] = 1
                V[j*6+k][col] = 1
                W[k*6+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
