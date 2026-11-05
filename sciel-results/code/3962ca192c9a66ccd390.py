import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def _strassen_2x2x2():
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    # m1: (a11+a22)(b11+b22) -> c11 += m1, c22 += m1
    U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
    # m2: (a21+a22)b11 -> c21 += m2, c22 -= m2
    U[2][1]=1; U[3][1]=1; V[0][1]=1; W[2][1]=1; W[3][1]=-1
    # m3: a11(b12-b22) -> c12 += m3, c22 += m3
    U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[1][2]=1; W[3][2]=1
    # m4: a22(b21-b11) -> c11 += m4, c21 += m4
    U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[0][3]=1; W[2][3]=1
    # m5: (a11+a12)b22 -> c11 -= m5, c12 += m5
    U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=-1; W[1][4]=1
    # m6: (a21-a11)(b11+b12) -> c22 += m6
    U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[3][5]=1
    # m7: (a12-a22)(b21+b22) -> c11 += m7
    U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1
    return U, V, W

def _schoolbook_block(rows, cols, out_cols):
    rank = rows * cols * out_cols
    U = [[0]*rank for _ in range(rows*cols)]
    V = [[0]*rank for _ in range(cols*out_cols)]
    W = [[0]*rank for _ in range(out_cols*rows)]
    c = 0
    for i in range(rows):
        for j in range(cols):
            for k in range(out_cols):
                U[i*cols+j][c]=1; V[j*out_cols+k][c]=1; W[k*rows+i][c]=1; c+=1
    return U, V, W

def _m223():
    U1, V1, W1 = _strassen_2x2x2()
    U2, V2, W2 = _schoolbook_block(2, 2, 1)
    # U2: (4,4), V2: (2,4), W2: (2,4)
    # Need to map V2 and W2 to the 3-column version
    # V2[j*1+k] where j=0,1 and k=0 -> but we need k=2 in the 3-col B
    # So V_full[j*3+2] = V2[j*1+0]
    # W2[k*2+i] where k=0 -> W_full[2*2+i] = W2[0*2+i]
    
    U = [[0]*11 for _ in range(4)]
    V = [[0]*11 for _ in range(6)]
    W = [[0]*11 for _ in range(6)]
    for i in range(4):
        for r in range(7):
            U[i][r] = U1[i][r]
        for r in range(4):
            U[i][7+r] = U2[i][r]
    for j in range(2):
        for r in range(7):
            V[j*3+0][r] = V1[j*2+0][r]
            V[j*3+1][r] = V1[j*2+1][r]
        for r in range(4):
            V[j*3+2][7+r] = V2[j*1+0][r]
    for i in range(2):
        for r in range(7):
            W[0*2+i][r] = W1[0*2+i][r]
            W[1*2+i][r] = W1[1*2+i][r]
        for r in range(4):
            W[2*2+i][7+r] = W2[0*2+i][r]
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    i_blocks = [(0, 2), (2, 3)]
    j_blocks = [(0, 2), (2, 4), (4, 5)]
    k_blocks = [(0, 3), (3, 6)]
    
    sub_decomps = []
    for i_s, i_e in i_blocks:
        for j_s, j_e in j_blocks:
            for k_s, k_e in k_blocks:
                i_sz, j_sz, k_sz = i_e-i_s, j_e-j_s, k_e-k_s
                if i_sz==2 and j_sz==2 and k_sz==3:
                    Us, Vs, Ws = _m223(); rs = 11
                elif i_sz==2 and j_sz==1 and k_sz==3:
                    Us, Vs, Ws = _schoolbook_block(2,1,3); rs = 6
                elif i_sz==1 and j_sz==2 and k_sz==3:
                    Us, Vs, Ws = _schoolbook_block(1,2,3); rs = 6
                elif i_sz==1 and j_sz==1 and k_sz==3:
                    Us, Vs, Ws = _schoolbook_block(1,1,3); rs = 3
                else:
                    Us, Vs, Ws = _schoolbook_block(i_sz,j_sz,k_sz); rs = i_sz*j_sz*k_sz
                sub_decomps.append((i_s,i_sz,j_s,j_sz,k_s,k_sz,Us,Vs,Ws,rs))
    
    total_rank = sum(x[8] for x in sub_decomps)
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    co = 0
    for i_s,i_sz,j_s,j_sz,k_s,k_sz,Us,Vs,Ws,rs in sub_decomps:
        for r in range(rs):
            for i in range(i_sz):
                for j in range(j_sz):
                    U[(i_s+i)*m + (j_s+j)][co+r] = Us[i*j_sz+j][r]
            for j in range(j_sz):
                for k in range(k_sz):
                    V[(j_s+j)*p + (k_s+k)][co+r] = Vs[j*k_sz+k][r]
            for k in range(k_sz):
                for i in range(i_sz):
                    W[(k_s+k)*n + (i_s+i)][co+r] = Ws[k*i_sz+i][r]
        co += rs
    
    return {"U": U, "V": V, "W": W}
