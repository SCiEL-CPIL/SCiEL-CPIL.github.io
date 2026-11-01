"""Tensor decomposition for 2x2x5 matrix multiplication."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
DEADLINE = 145.0
import numpy as np

def _verify(U, V, W, n=2, m=2, p=5):
    """Verify exact decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 5
    # W index: W[k*n + i] for output C[k,i]
    # U index: U[i*m + j] for input A[i,j]
    # V index: V[j*p + k] for input B[j,k]
    
    # Rank-19: Strassen(7) for cols 0,1 + schoolbook(12) for cols 2,3,4
    rank = 19
    U = [[0.0]*rank for _ in range(4)]
    V = [[0.0]*rank for _ in range(10)]
    W = [[0.0]*rank for _ in range(10)]
    
    # Strassen for 2x2 block (cols 0,1 of B)
    # m1=(a00+a11)(b00+b11): U[0],U[3]; V[0],V[6]; contributes to C[0,0](W[0]) and C[1,1](W[3])
    U[0][0]=1; U[3][0]=1; V[0][0]=1; V[6][0]=1; W[0][0]=1; W[3][0]=1
    
    # m2=(a10+a11)b00: U[2],U[3]; V[0]; contributes to C[1,0](W[1]) and C[1,1](W[3])
    U[2][1]=1; U[3][1]=1; V[0][1]=1; W[1][1]=1; W[3][1]=-1
    
    # m3=a00(b01-b11): U[0]; V[1],V[6]; contributes to C[0,1](W[2]) and C[1,1](W[3])
    U[0][2]=1; V[1][2]=1; V[6][2]=-1; W[2][2]=1; W[3][2]=-1
    
    # m4=a11(b10-b00): U[3]; V[5],V[0]; contributes to C[1,0](W[1]) and C[0,0](W[0])
    U[3][3]=1; V[5][3]=1; V[0][3]=-1; W[1][3]=1; W[0][3]=-1
    
    # m5=(a00+a10)b11: U[0],U[2]; V[6]; contributes to C[0,1](W[2]) and C[0,0](W[0])
    U[0][4]=1; U[2][4]=1; V[6][4]=1; W[2][4]=1; W[0][4]=-1
    
    # m6=(a01-a10)(b00+b10): U[1],U[2]; V[0],V[5]; contributes to C[0,0](W[0])
    U[1][5]=1; U[2][5]=-1; V[0][5]=1; V[5][5]=1; W[0][5]=-1
    
    # m7=(a01-a11)(b01+b11): U[1],U[3]; V[1],V[6]; contributes to C[0,1](W[2])
    U[1][6]=1; U[3][6]=-1; V[1][6]=1; V[6][6]=1; W[2][6]=1
    
    # Schoolbook for cols 2,3,4
    col = 7
    for k in [2, 3, 4]:
        # C[0,k] = a00*b0k + a01*b1k
        U[0][col]=1; V[k][col]=1; W[k*n+0][col]=1; col+=1
        U[1][col]=1; V[k+5][col]=1; W[k*n+0][col]=1; col+=1
        # C[1,k] = a10*b0k + a11*b1k
        U[2][col]=1; V[k][col]=1; W[k*n+1][col]=1; col+=1
        U[3][col]=1; V[k+5][col]=1; W[k*n+1][col]=1; col+=1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
