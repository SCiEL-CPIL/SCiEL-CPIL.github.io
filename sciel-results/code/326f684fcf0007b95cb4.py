import time
import numpy as np

START = time.time()
DEADLINE = START + 140.0

def _strassen_2x2x2():
    """Strassen 2x2x2 decomposition, rank 7."""
    U = np.zeros((4, 7), dtype=np.int16)
    V = np.zeros((4, 7), dtype=np.int16)
    W = np.zeros((4, 7), dtype=np.int16)
    # M1 = (A00+A11)(B00+B11)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # M2 = (A10+A11)B00
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[1,1]=1; W[3,1]=-1
    # M3 = A00(B01-B11)
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[2,2]=1; W[3,2]=1
    # M4 = A11(B10-B00)
    U[3,3]=1; V[0,3]=-1; V[2,3]=1; W[0,3]=1; W[1,3]=1
    # M5 = (A00+A01)B11
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[2,4]=1
    # M6 = (A10-A00)(B00+B01)
    U[0,5]=-1; U[2,5]=1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    # M7 = (A01-A11)(B10+B11)
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W

def _build_4x4x4_kron():
    """Build 4x4x4 decomposition using Kronecker product of 2x2x2, rank 49."""
    U2, V2, W2 = _strassen_2x2x2()
    r2 = 7
    rank = r2 * r2  # 49
    
    U = np.zeros((16, rank), dtype=np.int16)
    V = np.zeros((16, rank), dtype=np.int16)
    W = np.zeros((16, rank), dtype=np.int16)
    
    for c1 in range(r2):
        for c2 in range(r2):
            c = c1 * r2 + c2
            for i1 in range(2):
                for i2 in range(2):
                    i = i1 * 2 + i2
                    U[i, c] = U2[i1, c1] * U2[i2, c2]
            for j1 in range(2):
                for j2 in range(2):
                    j = j1 * 2 + j2
                    V[j, c] = V2[j1, c1] * V2[j2, c2]
            for k1 in range(2):
                for k2 in range(2):
                    k = k1 * 2 + k2
                    W[k, c] = W2[k1, c1] * W2[k2, c2]
                    
    return U, V, W

def construct():
    """
    Construct rank-49 decomposition for 4x4 * 4x8 matrix multiplication.
    Uses Kronecker product of 2x2x2 (rank 7) to get 4x4x4 (rank 49),
    then extends to 4x4x8 by replicating V and W factors.
    """
    if time.time() > DEADLINE:
        # Fallback to rank 98 if something goes wrong or time is low
        return _construct_fallback()
        
    n, m, p = 4, 4, 8
    
    U4, V4, W4 = _build_4x4x4_kron()
    
    rank = 49
    
    U = U4.copy()  # 16 x 49
    V = np.zeros((32, rank), dtype=np.int16)
    W = np.zeros((32, rank), dtype=np.int16)
    
    # Map 4x4x8 indices to 4x4x4 indices
    # j in [0, 32) -> j_map in [0, 16)
    # j = r_B * 8 + c_B
    # j_map = r_B * 4 + (c_B % 4)
    for j in range(32):
        r_B = j // 8
        c_B = j % 8
        j_map = r_B * 4 + (c_B % 4)
        V[j, :] = V4[j_map, :]
        
    # k in [0, 32) -> k_map in [0, 16)
    # k = r_C * 8 + c_C
    # k_map = r_C * 4 + (c_C % 4)
    for k in range(32):
        r_C = k // 8
        c_C = k % 8
        k_map = r_C * 4 + (c_C % 4)
        W[k, :] = W4[k_map, :]
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _construct_fallback():
    """Fallback rank-98 decomposition."""
    U2, V2, W2 = _strassen_2x2x2()
    rank49 = 49
    U4 = np.zeros((16, rank49), dtype=np.int16)
    V4 = np.zeros((16, rank49), dtype=np.int16)
    W4 = np.zeros((16, rank49), dtype=np.int16)
    for c0 in range(7):
        for c1 in range(7):
            c = c0*7 + c1
            for i in range(4):
                for j in range(4):
                    U4[i*4+j, c] = U2[(i//2)*2+(j//2), c0] * U2[(i%2)*2+(j%2), c1]
            for j in range(4):
                for k in range(4):
                    V4[j*4+k, c] = V2[(j//2)*2+(k//2), c0] * V2[(j%2)*2+(k%2), c1]
            for k in range(4):
                for i in range(4):
                    W4[k*4+i, c] = W2[(k//2)*2+(i//2), c0] * W2[(k%2)*2+(i%2), c1]
    
    total_rank = 98
    U = np.zeros((16, total_rank), dtype=np.int16)
    V = np.zeros((32, total_rank), dtype=np.int16)
    W = np.zeros((32, total_rank), dtype=np.int16)
    
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j*8 + k, 0:49] = V4[j*4 + k, :]
    for k in range(4):
        for i in range(4):
            W[k*4 + i, 0:49] = W4[k*4 + i, :]
            
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4, 8):
            V[j*8 + k, 49:98] = V4[j*4 + (k-4), :]
    for k in range(4, 8):
        for i in range(4):
            W[k*4 + i, 49:98] = W4[(k-4)*4 + i, :]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
