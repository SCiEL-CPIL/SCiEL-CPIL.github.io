import numpy as np

def construct():
    n, m, p = 4, 4, 9
    
    # Strassen 2x2x2 decomposition
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    U2[0,0]=1; U2[3,0]=1; V2[0,0]=1; V2[3,0]=1; W2[0,0]=1; W2[3,0]=1
    U2[2,1]=1; U2[3,1]=1; V2[0,1]=1; W2[2,1]=1; W2[3,1]=1
    U2[0,2]=1; V2[1,2]=1; V2[3,2]=-1; W2[1,2]=1; W2[3,2]=1
    U2[3,3]=1; V2[2,3]=1; V2[0,3]=-1; W2[0,3]=1; W2[3,3]=1
    U2[0,4]=1; U2[1,4]=1; V2[3,4]=1; W2[0,4]=1; W2[1,4]=-1
    U2[2,5]=1; U2[0,5]=-1; V2[0,5]=1; V2[1,5]=1; W2[2,5]=1
    U2[1,6]=1; U2[3,6]=-1; V2[2,6]=1; V2[3,6]=1; W2[0,6]=1; W2[1,6]=1
    
    # Strassen 4x4x4 decomposition via 2x2 blocks
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    def blk(Ut, Vt, Wt, ab, bb, cb, off):
        for s in range(7):
            c = off + s
            for (r1,c1,sg) in ab:
                for ii in range(2):
                    for jj in range(2):
                        Ut[(2*r1+ii)*4+(2*c1+jj), c] += sg*U2[ii*2+jj, s]
            for (r1,c1,sg) in bb:
                for ii in range(2):
                    for jj in range(2):
                        Vt[(2*r1+ii)*4+(2*c1+jj), c] += sg*V2[ii*2+jj, s]
            for (r1,c1,sg) in cb:
                for ii in range(2):
                    for jj in range(2):
                        Wt[(2*r1+ii)*4+(2*c1+jj), c] += sg*W2[ii*2+jj, s]
    
    blk(U4,V4,W4,[(0,0,1),(1,1,1)],[(0,0,1),(1,1,1)],[(0,0,1),(1,1,1)],0)
    blk(U4,V4,W4,[(1,0,1),(1,1,1)],[(0,0,1)],[(1,0,1),(1,1,1)],7)
    blk(U4,V4,W4,[(0,0,1)],[(0,1,1),(1,1,-1)],[(0,1,1),(1,1,1)],14)
    blk(U4,V4,W4,[(1,1,1)],[(1,0,1),(0,0,-1)],[(0,0,1),(1,1,1)],21)
    blk(U4,V4,W4,[(0,0,1),(0,1,1)],[(1,1,1)],[(0,0,1),(0,1,-1)],28)
    blk(U4,V4,W4,[(1,0,1),(0,0,-1)],[(0,0,1),(0,1,1)],[(1,0,1)],35)
    blk(U4,V4,W4,[(0,1,1),(1,1,-1)],[(1,0,1),(1,1,1)],[(0,0,1),(0,1,1)],42)
    
    # Schoolbook 4x4x1
    U1 = np.zeros((16,16))
    V1 = np.zeros((4,16))
    W1 = np.zeros((4,16))
    for i in range(4):
        for j in range(4):
            c = i*4+j
            U1[i*4+j,c]=1; V1[j,c]=1; W1[i,c]=1
    
    # Assemble 4x4x9: rank 49+49+16=114
    rank = 114
    U = np.zeros((16, rank))
    V = np.zeros((36, rank))
    W = np.zeros((36, rank))
    
    # B cols 0-3
    U[:,0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j*9+k, 0:49] = V4[j*4+k, :]
    for k in range(4):
        for i in range(4):
            W[k*4+i, 0:49] = W4[k*4+i, :]
    
    # B cols 4-7
    U[:,49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j*9+4+k, 49:98] = V4[j*4+k, :]
    for k in range(4):
        for i in range(4):
            W[(4+k)*4+i, 49:98] = W4[k*4+i, :]
    
    # B col 8
    U[:,98:114] = U1
    for j in range(4):
        V[j*9+8, 98:114] = V1[j, :]
    for i in range(4):
        W[8*4+i, 98:114] = W1[i, :]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
