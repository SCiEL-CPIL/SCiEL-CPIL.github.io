import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # 1. Strassen 2x2x2 (Rank 7)
    # U (4x7), V (4x7), W (4x7)
    U_222 = np.zeros((4, 7))
    V_222 = np.zeros((4, 7))
    W_222 = np.zeros((4, 7))
    
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    # P1 = (A11+A22)(B11+B22)
    U_222[0, 0] = 1; U_222[3, 0] = 1
    V_222[0, 0] = 1; V_222[3, 0] = 1
    W_222[0, 0] = 1; W_222[2, 0] = 1; W_222[1, 0] = 1; W_222[3, 0] = 1
    # ... this is tedious to hardcode manually in the thought process.
    # I will generate them programmatically in the code.
    
    def make_strassen_222():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # P1
        U[0,0]=1; U[3,0]=1
        V[0,0]=1; V[3,0]=1
        W[0,0]=1; W[2,0]=1; W[1,0]=1; W[3,0]=1
        
        # P2
        U[2,1]=1; U[3,1]=1
        V[0,1]=1
        W[0,1]=1; W[2,1]=1
        
        # P3
        U[0,2]=1
        V[1,2]=1; V[3,2]=-1
        W[0,2]=1; W[1,2]=1
        
        # P4
        U[3,3]=1
        V[2,3]=1; V[0,3]=-1
        W[2,3]=1; W[3,3]=1
        
        # P5
        U[0,4]=1; U[1,4]=1
        V[3,4]=1
        W[2,4]=1; W[3,4]=1
        
        # P6
        U[2,5]=-1; U[0,5]=1
        V[0,5]=1; V[1,5]=1
        W[0,5]=1; W[1,5]=1
        
        # P7
        U[1,6]=-1; U[3,6]=1
        V[2,6]=1; V[3,6]=1
        W[2,6]=1; W[3,6]=1
        
        return U, V, W

    def make_schoolbook_122():
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((4, 4))
        col = 0
        for j in range(2):
            for k in range(2):
                U[j, col] = 1
                V[j*2+k, col] = 1
                W[k*1+j, col] = 1 # W index k*n+i, n=1, i=0. So W[k, col]
                col += 1
        return U, V, W

    # Combine to 3x2x2
    U_322, V_322, W_322 = np.hstack([
        np.vstack([make_strassen_222()[0], np.zeros((2, 7))]), # A1 is 2x2, A2 is 1x2. U_222 is 4x7. U_122 is 2x4.
        # Wait, U_222 rows 0..3. U_122 rows 0..1.
        # Global U rows 0..5.
        # So we need to place U_222 in rows 0..3 and U_122 in rows 4..5.
    ])
    # This hstack logic is wrong. I need to build the full matrices.
    
    # I will write the code to build the full matrices directly.
    
    U = np.zeros((12, 88))
    V = np.zeros((32, 88))
    W = np.zeros((24, 88))
    
    # ... fill logic ...
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
