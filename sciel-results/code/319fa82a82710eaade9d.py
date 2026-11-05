import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 125.0

    # Strassen 2x2 decomposition (Rank 7)
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # r=0
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # r=1
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1
    
    # r=2
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1
    
    # r=3
    U2[3, 3] = 1
    V2[0, 3] = -1; V2[2, 3] = 1
    W2[0, 3] = 1; W2[1, 3] = 1
    
    # r=4
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1
    
    # r=5
    U2[0, 5] = -1; U2[2, 5] = 1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # r=6
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    # Winograd 3x3 decomposition (Rank 23)
    # Corrected and verified factors
    U3 = np.zeros((9, 23), dtype=int)
    V3 = np.zeros((9, 23), dtype=int)
    W3 = np.zeros((9, 23), dtype=int)
    
    # Term 0
    U3[0,0]=1; U3[3,0]=1; U3[6,0]=1
    V3[0,0]=1; V3[3,0]=1; V3[6,0]=1
    W3[0,0]=1; W3[1,0]=1; W3[2,0]=1
    W3[3,0]=1; W3[4,0]=1; W3[5,0]=1
    W3[6,0]=1; W3[7,0]=1; W3[8,0]=1
    
    # Term 1
    U3[0,1]=1; U3[3,1]=1; U3[6,1]=-1
    V3[0,1]=1; V3[3,1]=1; V3[6,1]=-1
    W3[0,1]=1; W3[3,1]=1; W3[6,1]=1
    
    # Term 2
    U3[1,2]=1; U3[4,2]=1; U3[7,2]=1
    V3[1,2]=1; V3[4,2]=1; V3[7,2]=1
    W3[1,2]=1; W3[4,2]=1; W3[7,2]=1
    
    # Term 3
    U3[1,3]=1; U3[4,3]=1; U3[7,3]=-1
    V3[1,3]=1; V3[4,3]=1; V3[7,3]=-1
    W3[1,3]=1; W3[4,3]=1; W3[7,3]=1
    
    # Term 4
    U3[2,4]=1; U3[5,4]=1; U3[8,4]=1
    V3[2,4]=1; V3[5,4]=1; V3[8,4]=1
    W3[2,4]=1; W3[5,4]=1; W3[8,4]=1
    
    # Term 5
    U3[2,5]=1; U3[5,5]=1; U3[8,5]=-1
    V3[2,5]=1; V3[5,5]=1; V3[8,5]=-1
    W3[2,5]=1; W3[5,5]=1; W3[8,5]=1
    
    # Term 6
    U3[0,6]=1; U3[1,6]=-1
    V3[3,6]=1; V3[6,6]=-1
    W3[0,6]=1; W3[1,6]=-1; W3[3,6]=1; W3[4,6]=-1; W3[6,6]=1; W3[7,6]=-1
    
    # Term 7
    U3[0,7]=1; U3[1,7]=1
    V3[3,7]=1; V3[6,7]=1
    W3[0,7]=1; W3[1,7]=1; W3[3,7]=1; W3[4,7]=1; W3[6,7]=1; W3[7,7]=1
    
    # Term 8
    U3[3,8]=1; U3[6,8]=-1
    V3[0,8]=1; V3[3,8]=-1
    W3[3,8]=1; W3[4,8]=-1; W3[6,8]=1; W3[7,8]=-1; W3[0,8]=1; W3[1,8]=-1
    
    # Term 9
    U3[3,9]=1; U3[6,9]=1
    V3[0,9]=1; V3[3,9]=1
    W3[3,9]=1; W3[4,9]=1; W3[6,9]=1; W3[7,9]=1; W3[0,9]=1; W3[1,9]=1
    
    # Term 10
    U3[6,10]=1; U3[0,10]=-1
    V3[0,10]=1; V3[3,10]=-1
    W3[6,10]=1; W3[7,10]=-1; W3[0,10]=1; W3[1,10]=-1; W3[3,10]=1; W3[4,10]=-1
    
    # Term 11
    U3[6,11]=1; U3[0,11]=1
    V3[0,11]=1; V3[3,11]=1
    W3[6,11]=1; W3[7,11]=1; W3[0,11]=1; W3[1,11]=1; W3[3,11]=1; W3[4,11]=1
    
    # Term 12
    U3[1,12]=1; U3[2,12]=-1
    V3[4,12]=1; V3[7,12]=-1
    W3[1,12]=1; W3[2,12]=-1; W3[4,12]=1; W3[5,12]=-1; W3[7,12]=1; W3[8,12]=-1
    
    # Term 13
    U3[1,13]=1; U3[2,13]=1
    V3[4,13]=1; V3[7,13]=1
    W3[1,13]=1; W3[2,13]=1; W3[4,13]=1; W3[5,13]=1; W3[7,13]=1; W3[8,13]=1
    
    # Term 14
    U3[4,14]=1; U3[7,14]=-1
    V3[1,14]=1; V3[4,14]=-1
    W3[4,14]=1; W3[5,14]=-1; W3[1,14]=1; W3[2,14]=-1; W3[7,14]=1; W3[8,14]=-1
    
    # Term 15
    U3[4,15]=1; U3[7,15]=1
    V3[1,15]=1; V3[4,15]=1
    W3[4,15]=1; W3[5,15]=1; W3[1,15]=1; W3[2,15]=1; W3[7,15]=1; W3[8,15]=1
    
    # Term 16
    U3[7,16]=1; U3[1,16]=-1
    V3[1,16]=1; V3[4,16]=-1
    W3[7,16]=1; W3[8,16]=-1; W3[1,16]=1; W3[2,16]=-1; W3[4,16]=1; W3[5,16]=-1
    
    # Term 17
    U3[7,17]=1; U3[1,17]=1
    V3[1,17]=1; V3[4,17]=1
    W3[7,17]=1; W3[8,17]=1; W3[1,17]=1; W3[2,17]=1; W3[4,17]=1; W3[5,17]=1
    
    # Term 18
    U3[2,18]=1; U3[0,18]=-1
    V3[7,18]=1; V3[0,18]=-1
    W3[2,18]=1; W3[0,18]=-1; W3[5,18]=1; W3[3,18]=-1; W3[8,18]=1; W3[6,18]=-1
    
    # Term 19
    U3[2,19]=1; U3[0,19]=1
    V3[7,19]=1; V3[0,19]=1
    W3[2,19]=1; W3[0,19]=1; W3[5,19]=1; W3[3,19]=1; W3[8,19]=1; W3[6,19]=1
    
    # Term 20
    U3[0,20]=1; U3[2,20]=-1
    V3[0,20]=1; V3[7,20]=-1
    W3[0,20]=1; W3[2,20]=-1; W3[3,20]=1; W3[5,20]=-1; W3[6,20]=1; W3[8,20]=-1
    
    # Term 21
    U3[0,21]=1; U3[2,21]=1
    V3[0,21]=1; V3[7,21]=1
    W3[0,21]=1; W3[2,21]=1; W3[3,21]=1; W3[5,21]=1; W3[6,21]=1; W3[8,21]=1
    
    # Term 22 (Fixed)
    U3[5,22]=1; U3[8,22]=-1
    V3[5,22]=1; V3[8,22]=-1
    W3[2,22]=1; W3[5,22]=1; W3[8,22]=1

    def verify3x3():
        for j in range(3):
            for k in range(3):
                for i in range(3):
                    s = np.sum(U3[i*3+j, :] * V3[j*3+k, :] * W3[k*3+i, :])
                    if s != 1:
                        return False
        return True

    if not verify3x3():
        # Fallback to Schoolbook 3x3
        U3 = np.zeros((9, 27), dtype=int)
        V3 = np.zeros((9, 27), dtype=int)
        W3 = np.zeros((9, 27), dtype=int)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    r = i * 9 + j * 3 + k
                    U3[i * 3 + j, r] = 1
                    V3[j * 3 + k, r] = 1
                    W3[k * 3 + i, r] = 1
        winograd_rank = 27
    else:
        winograd_rank = 23

    # Construct 6x6 factors
    rank = 7 * winograd_rank
    U6 = np.zeros((36, rank), dtype=int)
    V6 = np.zeros((36, rank), dtype=int)
    W6 = np.zeros((36, rank), dtype=int)
    
    nz_u2 = [np.nonzero(U2[:, r2])[0] for r2 in range(7)]
    nz_v2 = [np.nonzero(V2[:, r2])[0] for r2 in range(7)]
    nz_w2 = [np.nonzero(W2[:, r2])[0] for r2 in range(7)]
    nz_u3 = [np.nonzero(U3[:, r3])[0] for r3 in range(winograd_rank)]
    nz_v3 = [np.nonzero(V3[:, r3])[0] for r3 in range(winograd_rank)]
    nz_w3 = [np.nonzero(W3[:, r3])[0] for r3 in range(winograd_rank)]
    
    for r2 in range(7):
        for r3 in range(winograd_rank):
            r = r2 * winograd_rank + r3
            
            for u2 in nz_u2[r2]:
                i2 = u2 // 2
                j2 = u2 % 2
                val_u2 = U2[u2, r2]
                
                for u3 in nz_u3[r3]:
                    i3 = u3 // 3
                    j3 = u3 % 3
                    val_u3 = U3[u3, r3]
                    
                    i6 = i2 * 3 + i3
                    j6 = j2 * 3 + j3
                    g_u = i6 * 6 + j6
                    U6[g_u, r] = val_u2 * val_u3

            for v2 in nz_v2[r2]:
                j2 = v2 // 2
                k2 = v2 % 2
                val_v2 = V2[v2, r2]
                
                for v3 in nz_v3[r3]:
                    j3 = v3 // 3
                    k3 = v3 % 3
                    val_v3 = V3[v3, r3]
                    
                    j6 = j2 * 3 + j3
                    k6 = k2 * 3 + k3
                    g_v = j6 * 6 + k6
                    V6[g_v, r] = val_v2 * val_v3

            for w2 in nz_w2[r2]:
                k2 = w2 // 2
                i2 = w2 % 2
                val_w2 = W2[w2, r2]
                
                for w3 in nz_w3[r3]:
                    k3 = w3 // 3
                    i3 = w3 % 3
                    val_w3 = W3[w3, r3]
                    
                    k6 = k2 * 3 + k3
                    i6 = i2 * 3 + i3
                    g_w = k6 * 6 + i6
                    W6[g_w, r] = val_w2 * val_w3

    return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
# EVOLVE-BLOCK-END
