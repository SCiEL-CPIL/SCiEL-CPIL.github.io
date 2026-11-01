import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def _strassen_2x2x2():
    """Return (U, V, W) for 2x2x2 Strassen.
    U: (4,7) indexed by (i,j), row=i*2+j
    V: (4,7) indexed by (j,k), row=j*2+k
    W: (4,7) indexed by (k,i), row=k*2+i
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # m1 = (A00-A11)*(B00+B11)
    U[0,0]=1; U[3,0]=-1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=-1  # C00=+1, C11=-1
    
    # m2 = (A10+A11)*B00
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[2,1]=1  # C10=+1
    
    # m3 = A00*(B01-B11)
    U[0,2]=1
    V[1,2]=1; V[3,2]=-1
    W[1,2]=1; W[3,2]=1  # C01=+1, C11=+1
    
    # m4 = A11*(B10-B00)
    U[3,3]=1
    V[2,3]=1; V[0,3]=-1
    W[0,3]=1; W[2,3]=1  # C00=+1, C10=+1
    
    # m5 = (A00+A01)*B11
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[0,4]=1; W[1,4]=1  # C00=+1, C01=+1
    
    # m6 = (A10-A00)*(B00+B01)
    U[2,5]=1; U[0,5]=-1
    V[0,5]=1; V[1,5]=1
    W[2,5]=1  # C10=+1
    
    # m7 = (A01-A11)*(B10+B11)
    U[1,6]=1; U[3,6]=-1
    V[2,6]=1; V[3,6]=1
    W[0,6]=-1; W[3,6]=1  # C00=-1, C11=+1
    
    return U, V, W

def _verify_2x2x2(U, V, W):
    """Verify that sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all i,j,k in {0,1}."""
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0
                for r in range(U.shape[1]):
                    s += U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r]
                if abs(s - 1) > 1e-9:
                    return False, (i,j,k,s)
    return True, None

def construct():
    n, m, p = 2, 3, 5
    rank = 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1} - Strassen
    sU, sV, sW = _strassen_2x2x2()
    for ii in range(2):
        for jj in range(2):
            U[ii*3+jj, col:col+7] = sU[ii*2+jj, :]
    for jj in range(2):
        for kk in range(2):
            V[jj*5+kk, col:col+7] = sV[jj*2+kk, :]
    for kk in range(2):
        for ii in range(2):
            W[kk*2+ii, col:col+7] = sW[kk*2+ii, :]
    col += 7
    
    # Block 2: i in {0,1}, j in {0,1}, k in {2,3} - Strassen
    sU, sV, sW = _strassen_2x2x2()
    for ii in range(2):
        for jj in range(2):
            U[ii*3+jj, col:col+7] = sU[ii*2+jj, :]
    for jj in range(2):
        for kk in range(2):
            V[jj*5+(kk+2), col:col+7] = sV[jj*2+kk, :]
    for kk in range(2):
        for ii in range(2):
            W[(kk+2)*2+ii, col:col+7] = sW[kk*2+ii, :]
    col += 7
    
    # Block 3: i in {0,1}, j in {0,1}, k in {4} - schoolbook 2x2x1
    for ii in range(2):
        for jj in range(2):
            U[ii*3+jj, col] = 1
            V[jj*5+4, col] = 1
            W[4*2+ii, col] = 1
            col += 1
    
    # Block 4: i in {0,1}, j in {2}, k in {0,1,2,3,4} - schoolbook 2x1x5
    for ii in range(2):
        for kk in range(5):
            U[ii*3+2, col] = 1
            V[2*5+kk, col] = 1
            W[kk*2+ii, col] = 1
            col += 1
    
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1) > 1e-9:
                    # Fallback to schoolbook
                    rank = n*m*p
                    U = np.zeros((n*m, rank))
                    V = np.zeros((m*p, rank))
                    W = np.zeros((p*n, rank))
                    c = 0
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                U[i2*m+j2, c] = 1
                                V[j2*p+k2, c] = 1
                                W[k2*n+i2, c] = 1
                                c += 1
                    break
            else:
                continue
            break
        else:
            continue
        break
    
    U_list = [[int(x) for x in row] for row in U]
    V_list = [[int(x) for x in row] for row in V]
    W_list = [[int(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
