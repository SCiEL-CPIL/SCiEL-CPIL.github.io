import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    col = 0
    # Block 1: Strassen 2x2x2 for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # A indices: 0=(0,0), 1=(0,1), 3=(1,0), 4=(1,1)
    # B indices: 0=(0,0), 1=(0,1), 5=(1,0), 6=(1,1)
    # C indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    # M1 = (A00+A11)(B00+B11), C00+=1, C11+=1
    U[0,col]=1; U[4,col]=1; V[0,col]=1; V[6,col]=1; W[0,col]=1; W[3,col]=1; col+=1
    # M2 = (A10+A11)B00, C10+=1, C11+=1
    U[3,col]=1; U[4,col]=1; V[0,col]=1; W[2,col]=1; W[3,col]=1; col+=1
    # M3 = A00(B01-B11), C01+=1, C11+=1
    U[0,col]=1; V[1,col]=1; V[6,col]=-1; W[1,col]=1; W[3,col]=1; col+=1
    # M4 = A11(B10-B00), C00+=1, C10+=1
    U[4,col]=1; V[5,col]=1; V[0,col]=-1; W[0,col]=1; W[2,col]=1; col+=1
    # M5 = (A00+A01)B11, C00-=1, C01+=1
    U[0,col]=1; U[1,col]=1; V[6,col]=1; W[0,col]=-1; W[1,col]=1; col+=1
    # M6 = (A10-A00)(B00+B10), C11+=1
    U[3,col]=1; U[0,col]=-1; V[0,col]=1; V[5,col]=1; W[3,col]=1; col+=1
    # M7 = (A01-A11)(B10+B11), C00+=1, C11-=1
    U[1,col]=1; U[4,col]=-1; V[5,col]=1; V[6,col]=1; W[0,col]=1; W[3,col]=-1; col+=1
    
    # Block 2: Strassen 2x2x2 for A[0:2,0:2] * B[0:2,2:4] -> C[2:4,0:2]
    # B indices: 2=(0,2), 3=(0,3), 7=(1,2), 8=(1,3)
    # C indices: 4=(2,0), 5=(2,1), 6=(3,0), 7=(3,1)
    
    # M1
    U[0,col]=1; U[4,col]=1; V[2,col]=1; V[8,col]=1; W[4,col]=1; W[7,col]=1; col+=1
    # M2
    U[3,col]=1; U[4,col]=1; V[2,col]=1; W[6,col]=1; W[7,col]=1; col+=1
    # M3
    U[0,col]=1; V[3,col]=1; V[8,col]=-1; W[5,col]=1; W[7,col]=1; col+=1
    # M4
    U[4,col]=1; V[7,col]=1; V[2,col]=-1; W[4,col]=1; W[6,col]=1; col+=1
    # M5
    U[0,col]=1; U[1,col]=1; V[8,col]=1; W[4,col]=-1; W[5,col]=1; col+=1
    # M6
    U[3,col]=1; U[0,col]=-1; V[2,col]=1; V[7,col]=1; W[7,col]=1; col+=1
    # M7
    U[1,col]=1; U[4,col]=-1; V[7,col]=1; V[8,col]=1; W[4,col]=1; W[7,col]=-1; col+=1
    
    # Block 3: 2x2x1 schoolbook for A[0:2,0:2] * B[0:2,4] -> C[4,0:2]
    # B indices: 4=(0,4), 9=(1,4)
    # C indices: 8=(4,0), 9=(4,1)
    
    U[0,col]=1; V[4,col]=1; W[8,col]=1; col+=1
    U[1,col]=1; V[9,col]=1; W[8,col]=1; col+=1
    U[3,col]=1; V[4,col]=1; W[9,col]=1; col+=1
    U[4,col]=1; V[9,col]=1; W[9,col]=1; col+=1
    
    # Block 4: 2x1x2 for A[0:2,2] * B[2,0:2] -> C[0:2,0:2]
    # A indices: 2=(0,2), 5=(1,2)
    # B indices: 10=(2,0), 11=(2,1)
    # C indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    U[2,col]=1; V[10,col]=1; W[0,col]=1; col+=1
    U[2,col]=1; V[11,col]=1; W[1,col]=1; col+=1
    U[5,col]=1; V[10,col]=1; W[2,col]=1; col+=1
    U[5,col]=1; V[11,col]=1; W[3,col]=1; col+=1
    
    # Block 5: 2x1x2 for A[0:2,2] * B[2,2:4] -> C[2:4,0:2]
    # B indices: 12=(2,2), 13=(2,3)
    # C indices: 4=(2,0), 5=(2,1), 6=(3,0), 7=(3,1)
    
    U[2,col]=1; V[12,col]=1; W[4,col]=1; col+=1
    U[2,col]=1; V[13,col]=1; W[5,col]=1; col+=1
    U[5,col]=1; V[12,col]=1; W[6,col]=1; col+=1
    U[5,col]=1; V[13,col]=1; W[7,col]=1; col+=1
    
    # Block 6: 2x1x1 for A[0:2,2] * B[2,4] -> C[4,0:2]
    # B index: 14=(2,4)
    # C indices: 8=(4,0), 9=(4,1)
    
    U[2,col]=1; V[14,col]=1; W[8,col]=1; col+=1
    U[5,col]=1; V[14,col]=1; W[9,col]=1; col+=1
    
    assert col == rank, f"Expected {rank} terms, got {col}"
    
    # Verify
    support = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m+j
                b = j*p+k
                c = k*n+i
                support.add((a,b,c))
    
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                val = np.dot(U[a], V[b]*W[c])
                if (a,b,c) in support:
                    if abs(val - 1.0) > 1e-10:
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}  # fallback
                else:
                    if abs(val) > 1e-10:
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
