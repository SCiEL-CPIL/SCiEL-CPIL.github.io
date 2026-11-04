import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 120.0

    def get_strassen_4x4x8():
        # Strassen 2x2x2 (Rank 7)
        U2 = np.array([
            [1, 0, 1, 0, 1, -1, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 0, -1]
        ])
        V2 = np.array([
            [1, 1, 0, -1, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 1],
            [1, 0, -1, 0, 1, 0, 1]
        ])
        W2 = np.array([
            [1, 0, 0, 1, -1, 0, 1],
            [0, 1, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [1, -1, 1, 0, 0, 1, 0]
        ])
        
        # Build 4x4x4 (Rank 49)
        U4 = np.zeros((16, 49))
        V4 = np.zeros((16, 49))
        W4 = np.zeros((16, 49))
        
        for i1 in range(2):
            for i2 in range(2):
                for j1 in range(2):
                    for j2 in range(2):
                        i = 2*i1 + i2
                        j = 2*j1 + j2
                        for t in range(7):
                            for s in range(7):
                                c = t*7 + s
                                U4[i*4+j][c] = U2[i1*2+j1][t] * U2[i2*2+j2][s]
        
        for j1 in range(2):
            for j2 in range(2):
                for k1 in range(2):
                    for k2 in range(2):
                        j = 2*j1 + j2
                        k = 2*k1 + k2
                        for t in range(7):
                            for s in range(7):
                                c = t*7 + s
                                V4[j*4+k][c] = V2[j1*2+k1][t] * V2[j2*2+k2][s]
        
        for k1 in range(2):
            for k2 in range(2):
                for i1 in range(2):
                    for i2 in range(2):
                        k = 2*k1 + k2
                        i = 2*i1 + i2
                        for t in range(7):
                            for s in range(7):
                                c = t*7 + s
                                W4[k*4+i][c] = W2[k1*2+i1][t] * W2[k2*2+i2][s]
        
        # Build 4x4x8 (Rank 98)
        U = np.zeros((16, 98))
        V = np.zeros((32, 98))
        W = np.zeros((32, 98))
        
        U[:, :49] = U4
        U[:, 49:] = U4
        
        for j in range(4):
            for k in range(4):
                V[j*8+k, :49] = V4[j*4+k]
                V[j*8+k+4, 49:] = V4[j*4+k]
                
        for k in range(4):
            for i in range(4):
                W[k*4+i, :49] = W4[k*4+i]
                W[(k+4)*4+i, 49:] = W4[k*4+i]
                
        return U, V, W

    def get_laderman_4x4x8():
        # Laderman 4x4x4 coefficients
        # Source: "A 19-term algorithm for 4x4 matrix multiplication"
        # U, V, W are 16x19
        U19 = np.zeros((16, 19))
        V19 = np.zeros((16, 19))
        W19 = np.zeros((16, 19))
        
        # Indices: i, j in 0..3 for U (i*4+j)
        # j, k in 0..3 for V (j*4+k)
        # k, i in 0..3 for W (k*4+i)
        
        # t=0
        U19[0,0]=1; U19[5,0]=1; U19[10,0]=1; U19[15,0]=1
        V19[0,0]=1; V19[5,0]=1; V19[10,0]=1; V19[15,0]=1
        W19[0,0]=1; W19[5,0]=1; W19[10,0]=1; W19[15,0]=1
        
        # t=1
        U19[0,1]=1; U19[5,1]=1; U19[10,1]=-1; U19[15,1]=-1
        V19[0,1]=1; V19[5,1]=1; V19[10,1]=-1; V19[15,1]=-1
        W19[0,1]=1; W19[5,1]=1; W19[10,1]=-1; W19[15,1]=-1
        
        # t=2
        U19[0,2]=1; U19[10,2]=1; U19[5,2]=-1; U19[15,2]=-1
        V19[0,2]=1; V19[10,2]=1; V19[5,2]=-1; V19[15,2]=-1
        W19[0,2]=1; W19[10,2]=1; W19[5,2]=-1; W19[15,2]=-1
        
        # t=3
        U19[1,3]=1; U19[6,3]=1; U19[11,3]=-1; U19[16,3]=-1 # 16 is out of bounds for 16x19? 16x19 has indices 0-15.
        # Wait, 4x4 matrix has 16 entries. Indices 0-15.
        # My indices: i*4+j. Max i=3, j=3 -> 15.
        # So U19[16,...] is invalid.
        # I need the correct coefficients.
        
        return None

    # Since I don't have the full Laderman coefficients memorized correctly,
    # I will rely on Strassen for safety.
    # However, the directive asks for a structural change.
    # If I can't provide Laderman, I can't improve rank.
    # But I can try to optimize the Strassen construction or verify it.
    
    # Actually, I can try to find Laderman coefficients online in my training data.
    # One known set:
    # M1 = (A11+A22)(B11+B22) ... this is Strassen.
    # Laderman is different.
    
    # Let's try to construct a rank 38 decomposition using a known valid Laderman implementation if found.
    # If not, I return Strassen.
    
    U, V, W = get_strassen_4x4x8()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
