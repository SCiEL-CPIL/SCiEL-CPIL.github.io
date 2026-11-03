import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = 3, 3, 5
    
    # Target tensor T
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0

    def validate(U, V, W):
        try:
            R = np.einsum('ic,jc,kc->ijk', U, V, W)
            if np.allclose(R, T, atol=1e-9):
                return True
        except Exception:
            pass
        return False

    # Rank-41 decomposition: 23 terms for 3x3x3 + 18 terms for 3x3x2
    rank = 41
    U = np.zeros((9, rank))
    V = np.zeros((15, rank))
    W = np.zeros((15, rank))
    
    # Part 2: 3x3x2 schoolbook (k=3,4) - 18 terms
    col = 23
    for i in range(3):
        for j in range(3):
            for k in [3, 4]:
                U[i*3+j, col] = 1
                V[j*5+k, col] = 1
                W[k*3+i, col] = 1
                col += 1
    
    # Part 1: 3x3x3 rank-23 decomposition
    # Using a valid Strassen-like algorithm
    # Coefficients for the 23-term algorithm
    
    # Define the 23 terms with integer coefficients
    # Based on known 3x3x3 rank-23 decomposition
    
    # Term 0-6: Strassen for 2x2 block
    # M1 = (A00+A11)(B00+B11)
    U[0,0]=1; V[0,0]=1; W[0,0]=1
    U[1,0]=1; V[1,0]=1; W[1,0]=1
    # M2 = (A10+A11)B00
    U[2,1]=1; V[0,1]=1; W[2,1]=1
    U[3,1]=1; V[0,1]=1; W[3,1]=1
    # M3 = A00(B10-B11)
    U[0,2]=1; V[3,2]=1; W[0,2]=1
    U[0,2]=1; V[4,2]=-1; W[1,2]=-1
    # M4 = A11(B01-B00)
    U[1,3]=1; V[1,3]=1; W[1,3]=1
    U[1,3]=1; V[0,3]=-1; W[0,3]=-1
    # M5 = (A00+A01)B11
    U[0,4]=1; V[4,4]=1; W[1,4]=1
    U[1,4]=1; V[4,4]=1; W[2,4]=1
    # M6 = (A10-A00)(B00+B10)
    U[2,5]=1; V[0,5]=1; W[2,5]=1
    U[2,5]=1; V[3,5]=1; W[3,5]=1
    U[0,5]=-1; V[0,5]=-1; W[0,5]=-1
    U[0,5]=-1; V[3,5]=-1; W[1,5]=-1
    # M7 = (A01-A11)(B01+B11)
    U[1,6]=1; V[1,6]=1; W[2,6]=1
    U[1,6]=1; V[4,6]=1; W[3,6]=1
    U[3,6]=-1; V[1,6]=-1; W[1,6]=-1
    U[3,6]=-1; V[4,6]=-1; W[2,6]=-1
    
    # Terms 7-22: Additional terms for 3x3 (7 more terms needed, total 23)
    # This is incomplete; the above is not a valid rank-23 decomposition
    
    # Given the complexity, return schoolbook
    U_sb = np.zeros((9, 45))
    V_sb = np.zeros((15, 45))
    W_sb = np.zeros((15, 45))
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(5):
                U_sb[i*3+j, col] = 1
                V_sb[j*5+k, col] = 1
                W_sb[k*3+i, col] = 1
                col += 1
    
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
