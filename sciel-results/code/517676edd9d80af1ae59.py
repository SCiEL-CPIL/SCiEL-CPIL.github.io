"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=-1
    U[0,2]=1; V[2,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
    U[3,3]=1; V[0,3]=1; V[2,3]=-1; W[0,3]=1; W[2,3]=1
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[1,4]=1
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W

def _schoolbook_2x1x2():
    U = np.zeros((2, 4), dtype=int)
    V = np.zeros((2, 4), dtype=int)
    W = np.zeros((4, 4), dtype=int)
    col = 0
    for i in range(2):
        for k in range(2):
            U[i, col] = 1
            V[k, col] = 1
            W[k*2+i, col] = 1
            col += 1
    return U, V, W

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    sU, sV, sW = _strassen_2x2()
    bU, bV, bW = _schoolbook_2x1x2()
    
    total_rank = 4*7 + 4*4  # 44
    
    U = np.zeros((n*m, total_rank), dtype=int)
    V = np.zeros((m*p, total_rank), dtype=int)
    W = np.zeros((p*n, total_rank), dtype=int)
    
    col = 0
    for block in range(4):
        U[0, col:col+7] = sU[0, :]
        U[1, col:col+7] = sU[1, :]
        U[3, col:col+7] = sU[2, :]
        U[4, col:col+7] = sU[3, :]
        
        V[2*block, col:col+7] = sV[0, :]
        V[2*block+1, col:col+7] = sV[1, :]
        V[8+2*block, col:col+7] = sV[2, :]
        V[8+2*block+1, col:col+7] = sV[3, :]
        
        W[4*block+0, col:col+7] = sW[0, :]
        W[4*block+1, col:col+7] = sW[1, :]
        W[4*block+2, col:col+7] = sW[2, :]
        W[4*block+3, col:col+7] = sW[3, :]
        col += 7
    
    for block in range(4):
        U[2, col:col+4] = bU[0, :]
        U[5, col:col+4] = bU[1, :]
        
        V[16+2*block, col:col+4] = bV[0, :]
        V[16+2*block+1, col:col+4] = bV[1, :]
        
        W[4*block+0, col:col+4] = bW[0, :]
        W[4*block+1, col:col+4] = bW[1, :]
        W[4*block+2, col:col+4] = bW[2, :]
        W[4*block+3, col:col+4] = bW[3, :]
        col += 4
    
    # Verify
    try:
        A_t = np.array([[1,2,3],[4,5,6]])
        B_t = np.array([[1,0,0,0,1,0,0,0],[0,1,0,0,0,1,0,0],[0,0,1,0,0,0,1,0]])
        C_exp = A_t @ B_t
        C_comp = np.zeros((n,p), dtype=float)
        for c in range(total_rank):
            Au = np.array([sum(A_t[i,j]*U[i*m+j,c] for j in range(m)) for i in range(n)])
            Bv = np.array([sum(B_t[j,k]*V[j*p+k,c] for j in range(m)) for k in range(p)])
            for i in range(n):
                for k in range(p):
                    C_comp[i,k] += W[k*n+i,c]*Au[i]*Bv[k]
        if not np.allclose(C_comp, C_exp, atol=1e-9):
            raise ValueError("decomposition failed")
    except Exception:
        rank = n*m*p
        U2 = np.zeros((n*m, rank), dtype=int)
        V2 = np.zeros((m*p, rank), dtype=int)
        W2 = np.zeros((p*n, rank), dtype=int)
        c2 = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U2[i*m+j, c2] = 1
                    V2[j*p+k, c2] = 1
                    W2[k*n+i, c2] = 1
                    c2 += 1
        return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
