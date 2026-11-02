import time
import numpy as np

_DEADLINE = time.time() + 145  # 5s safety margin before 150s kill

def _verify(U, V, W, n, m, p):
    """Verify decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(U.shape[1]):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = 2, 2, 7
    
    # Build rank-25 decomposition: 3 Strassen blocks + 1 schoolbook column
    rank = 25
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    def add_strassen(col_start, k_start):
        # M1 = (A00+A11)*(B0,ks + B1,ks+1) -> C0,ks += M1; C1,ks+1 += M1
        U[0, col_start+0] += 1; U[3, col_start+0] += 1
        V[7*0+k_start, col_start+0] += 1; V[7*1+(k_start+1), col_start+0] += 1
        W[2*k_start+0, col_start+0] += 1; W[2*(k_start+1)+1, col_start+0] += 1
        # M2 = (A10+A11)*B0,ks -> C1,ks += M2; C1,ks+1 -= M2
        U[2, col_start+1] += 1; U[3, col_start+1] += 1
        V[7*0+k_start, col_start+1] += 1
        W[2*k_start+1, col_start+1] += 1; W[2*(k_start+1)+1, col_start+1] -= 1
        # M3 = A00*(B0,ks+1 - B1,ks+1) -> C0,ks+1 += M3; C1,ks+1 += M3
        U[0, col_start+2] += 1
        V[7*0+(k_start+1), col_start+2] += 1; V[7*1+(k_start+1), col_start+2] -= 1
        W[2*(k_start+1)+0, col_start+2] += 1; W[2*(k_start+1)+1, col_start+2] += 1
        # M4 = A11*(B1,ks - B0,ks) -> C0,ks += M4; C1,ks += M4
        U[3, col_start+3] += 1
        V[7*1+k_start, col_start+3] += 1; V[7*0+k_start, col_start+3] -= 1
        W[2*k_start+0, col_start+3] += 1; W[2*k_start+1, col_start+3] += 1
        # M5 = (A00+A01)*B1,ks+1 -> C0,ks -= M5; C0,ks+1 += M5
        U[0, col_start+4] += 1; U[1, col_start+4] += 1
        V[7*1+(k_start+1), col_start+4] += 1
        W[2*k_start+0, col_start+4] -= 1; W[2*(k_start+1)+0, col_start+4] += 1
        # M6 = (A10-A00)*(B0,ks+B0,ks+1) -> C1,ks+1 += M6
        U[2, col_start+5] += 1; U[0, col_start+5] -= 1
        V[7*0+k_start, col_start+5] += 1; V[7*0+(k_start+1), col_start+5] += 1
        W[2*(k_start+1)+1, col_start+5] += 1
        # M7 = (A01-A11)*(B1,ks+B1,ks+1) -> C0,ks += M7
        U[1, col_start+6] += 1; U[3, col_start+6] -= 1
        V[7*1+k_start, col_start+6] += 1; V[7*1+(k_start+1), col_start+6] += 1
        W[2*k_start+0, col_start+6] += 1

    add_strassen(0, 0)
    add_strassen(7, 2)
    add_strassen(14, 4)
    
    # Schoolbook for column 6
    cs = 21
    U[0, cs+0] += 1; V[7*0+6, cs+0] += 1; W[2*6+0, cs+0] += 1
    U[1, cs+1] += 1; V[7*1+6, cs+1] += 1; W[2*6+0, cs+1] += 1
    U[2, cs+2] += 1; V[7*0+6, cs+2] += 1; W[2*6+1, cs+2] += 1
    U[3, cs+3] += 1; V[7*1+6, cs+3] += 1; W[2*6+1, cs+3] += 1
    
    # Verify
    if time.time() < _DEADLINE - 5:
        if not _verify(U, V, W, n, m, p):
            # Fallback to trivial rank-28
            r28 = n*m*p
            U28 = np.zeros((n*m, r28)); V28 = np.zeros((m*p, r28)); W28 = np.zeros((p*n, r28))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        r = i*m*p + j*p + k
                        U28[i*m+j, r] = 1; V28[j*p+k, r] = 1; W28[k*n+i, r] = 1
            return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
