import time
import numpy as np

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the optimal rank-7 Strassen decomposition for 2x2 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    rank = 7
    
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # Strassen products:
    # r=0: M1 = (a+d)(e+h)
    # r=1: M2 = (c+d)e
    # r=2: M3 = a(f-h)
    # r=3: M4 = d(g-e)
    # r=4: M5 = (a+b)h
    # r=5: M6 = (c-a)(e+f)
    # r=6: M7 = (b-d)(g+h)
    
    # U[i*m+j, r]: a=U[0], b=U[1], c=U[2], d=U[3]
    U[0,0]=1; U[3,0]=1    # M1: a+d
    U[2,1]=1; U[3,1]=1    # M2: c+d
    U[0,2]=1              # M3: a
    U[3,3]=1              # M4: d
    U[0,4]=1; U[1,4]=1    # M5: a+b
    U[2,5]=1; U[0,5]=-1   # M6: c-a
    U[1,6]=1; U[3,6]=-1   # M7: b-d
    
    # V[j*p+k, r]: e=V[0], f=V[1], g=V[2], h=V[3]
    V[0,0]=1; V[3,0]=1    # M1: e+h
    V[0,1]=1              # M2: e
    V[1,2]=1; V[3,2]=-1   # M3: f-h
    V[2,3]=1; V[0,3]=-1   # M4: g-e
    V[3,4]=1              # M5: h
    V[0,5]=1; V[1,5]=1    # M6: e+f
    V[2,6]=1; V[3,6]=1    # M7: g+h
    
    # W[k*n+i, r]: C[0,0]=W[0], C[1,0]=W[1], C[0,1]=W[2], C[1,1]=W[3]
    # C[0,0] = M1 + M4 - M5 + M7
    W[0,0]=1; W[0,3]=1; W[0,4]=-1; W[0,6]=1
    # C[1,0] = M2 + M4
    W[1,1]=1; W[1,3]=1
    # C[0,1] = M3 + M5
    W[2,2]=1; W[2,4]=1
    # C[1,1] = M1 - M2 + M3 + M6
    W[3,0]=1; W[3,1]=-1; W[3,2]=1; W[3,5]=1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
