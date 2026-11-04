import numpy as np
import time

def construct():
    n, m, p = 4, 4, 8
    deadline = time.time() + 140
    
    # Strassen 2x2x2 decomposition (rank 7)
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # r=0: (A00+A11)(B00+B11)
    U2[0,0]=1; U2[3,0]=1; V2[0,0]=1; V2[3,0]=1; W2[0,0]=1; W2[3,0]=1
    # r=1: (A10+A11)B00
    U2[2,1]=1; U2[3,1]=1; V2[0,1]=1; W2[1,1]=1; W2[3,1]=-1
    # r=2: A00(B01-B11)
    U2[0,2]=1; V2[1,2]=1; V2[3,2]=-1; W2[2,2]=1; W2[3,2]=1
    # r=3: A11(B10-B00)
    U2[3,3]=1; V2[0,3]=-1; V2[2,3]=1; W2[0,3]=1; W2[1,3]=1
    # r=4: (A00+A01)B11
    U2[0,4]=1; U2[1,4]=1; V2[3,4]=1; W2[0,4]=-1; W2[2,4]=1
    # r=5: (A10-A00)(B00+B01)
    U2[0,5]=-1; U2[2,5]=1; V2[0,5]=1; V2[1,5]=1; W2[3,5]=1
    # r=6: (A01-A11)(B10+B11)
    U2[1,6]=1; U2[3,6]=-1; V2[2,6]=1; V2[3,6]=1; W2[0,6]=1
    
    # Build 4x4x4 decomposition (rank 49) via tensor product of Strassen
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for ro in range(7):
        for ri in range(7):
            r = ro * 7 + ri
            for uo in range(4):
                if U2[uo, ro] != 0:
                    Io = uo // 2; Jo = uo % 2
                    for ui in range(4):
                        if U2[ui, ri] != 0:
                            Ii = ui // 2; Ji = ui % 2
                            i = 2*Io + Ii
                            j = 2*Jo + Ji
                            U4[i*4+j, r] = U2[uo,ro] * U2[ui,ri]
            for vo in range(4):
                if V2[vo, ro] != 0:
                    Jo = vo // 2; Ko = vo % 2
                    for vi in range(4):
                        if V2[vi, ri] != 0:
                            Ji = vi // 2; Ki = vi % 2
                            j = 2*Jo + Ji
                            k = 2*Ko + Ki
                            V4[j*4+k, r] = V2[vo,ro] * V2[vi,ri]
            for wo in range(4):
                if W2[wo, ro] != 0:
                    Ko = wo // 2; Io = wo % 2
                    for wi in range(4):
                        if W2[wi, ri] != 0:
                            Ki = wi // 2; Ii = wi % 2
                            k = 2*Ko + Ki
                            i = 2*Io + Ii
                            W4[k*4+i, r] = W2[wo,ro] * W2[wi,ri]
    
    # Build 4x4x8 decomposition (rank 98)
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j*8+k, 0:49] = V4[j*4+k, :]
            W[k*4+j, 0:49] = W4[k*4+j, :]
    
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j*8+(4+k), 49:98] = V4[j*4+k, :]
            W[(4+k)*4+j, 49:98] = W4[k*4+j, :]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
