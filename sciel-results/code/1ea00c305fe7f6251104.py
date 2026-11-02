import numpy as np

def construct():
    n, m, p = 2, 2, 7
    
    # Rank 25 decomposition
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

    # Apply Strassen to columns (0,1), (2,3), (4,5)
    add_strassen(0, 0)
    add_strassen(7, 2)
    add_strassen(14, 4)
    
    # Schoolbook for column 6 (indices 21 to 24)
    cs = 21
    # C06 = A00*B06 + A01*B16
    U[0, cs+0] += 1; V[7*0+6, cs+0] += 1; W[2*6+0, cs+0] += 1
    U[1, cs+1] += 1; V[7*1+6, cs+1] += 1; W[2*6+0, cs+1] += 1
    # C16 = A10*B06 + A11*B16
    U[2, cs+2] += 1; V[7*0+6, cs+2] += 1; W[2*6+1, cs+2] += 1
    U[3, cs+3] += 1; V[7*1+6, cs+3] += 1; W[2*6+1, cs+3] += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
