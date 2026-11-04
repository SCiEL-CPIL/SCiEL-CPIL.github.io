import numpy as np
import time

def construct():
    n, m, p = 4, 4, 8
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Strassen 2x2x2 decomposition (rank 7)
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # r=0: M1 = (A00+A11)(B00+B11)
    U2[0,0]=1; U2[3,0]=1; V2[0,0]=1; V2[3,0]=1; W2[0,0]=1; W2[3,0]=1
    # r=1: M2 = (A10+A11)B00
    U2[2,1]=1; U2[3,1]=1; V2[0,1]=1; W2[1,1]=1; W2[3,1]=-1
    # r=2: M3 = A00(B01-B11)
    U2[0,2]=1; V2[1,2]=1; V2[3,2]=-1; W2[2,2]=1; W2[3,2]=1
    # r=3: M4 = A11(B10-B00)
    U2[3,3]=1; V2[0,3]=-1; V2[2,3]=1; W2[0,3]=1; W2[1,3]=1
    # r=4: M5 = (A00+A01)B11
    U2[0,4]=1; U2[1,4]=1; V2[3,4]=1; W2[0,4]=-1; W2[2,4]=1
    # r=5: M6 = (A10-A00)(B00+B01)
    U2[0,5]=-1; U2[2,5]=1; V2[0,5]=1; V2[1,5]=1; W2[3,5]=1
    # r=6: M7 = (A01-A11)(B10+B11)
    U2[1,6]=1; U2[3,6]=-1; V2[2,6]=1; V2[3,6]=1; W2[0,6]=1
    
    # Build 4x4x4 decomposition (rank 49) via tensor product of Strassen
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # Pre-compute non-zero entries for efficiency
    U2_nz = [(i, j, U2[i, j]) for i in range(4) for j in range(7) if U2[i, j] != 0]
    V2_nz = [(i, j, V2[i, j]) for i in range(4) for j in range(7) if V2[i, j] != 0]
    W2_nz = [(i, j, W2[i, j]) for i in range(4) for j in range(7) if W2[i, j] != 0]
    
    for ro in range(7):
        for ri in range(7):
            r = ro * 7 + ri
            
            # U: index (i,j) -> i*4+j
            # U2 indices: (I,J) where I in {0,1}, J in {0,1}
            # uo = I*2 + J, ui = ip*2 + jp
            # i = 2*Io + Ii, j = 2*Jo + Ji
            for uo, uo_val in [(uo, U2[uo, ro]) for uo in range(4) if U2[uo, ro] != 0]:
                Io = uo // 2; Jo = uo % 2
                for ui, ui_val in [(ui, U2[ui, ri]) for ui in range(4) if U2[ui, ri] != 0]:
                    Ii = ui // 2; Ji = ui % 2
                    i = 2*Io + Ii
                    j = 2*Jo + Ji
                    U4[i*4+j, r] = uo_val * ui_val
                    
            # V: index (j,k) -> j*4+k
            # vo = Jo*2 + Ko, vi = Ji*2 + Ki
            # j = 2*Jo + Ji, k = 2*Ko + Ki
            for vo, vo_val in [(vo, V2[vo, ro]) for vo in range(4) if V2[vo, ro] != 0]:
                Jo = vo // 2; Ko = vo % 2
                for vi, vi_val in [(vi, V2[vi, ri]) for vi in range(4) if V2[vi, ri] != 0]:
                    Ji = vi // 2; Ki = vi % 2
                    j = 2*Jo + Ji
                    k = 2*Ko + Ki
                    V4[j*4+k, r] = vo_val * vi_val
                    
            # W: index (k,i) -> k*4+i
            # wo = Ko*2 + Io, wi = Ki*2 + Ii
            # k = 2*Ko + Ki, i = 2*Io + Ii
            for wo, wo_val in [(wo, W2[wo, ro]) for wo in range(4) if W2[wo, ro] != 0]:
                Ko = wo // 2; Io = wo % 2
                for wi, wi_val in [(wi, W2[wi, ri]) for wi in range(4) if W2[wi, ri] != 0]:
                    Ki = wi // 2; Ii = wi % 2
                    k = 2*Ko + Ki
                    i = 2*Io + Ii
                    W4[k*4+i, r] = wo_val * wi_val
            
            if time.time() > deadline:
                # Fallback: return partial or current best (though construction is fast)
                pass
    
    # Build 4x4x8 decomposition (rank 98)
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    # Left 4x4 block (output columns 0-3)
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j*8+k, 0:49] = V4[j*4+k, :]
            W[k*4+j, 0:49] = W4[k*4+j, :]
            
    # Right 4x4 block (output columns 4-7)
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j*8+(4+k), 49:98] = V4[j*4+k, :]
            W[(4+k)*4+j, 49:98] = W4[k*4+j, :]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
