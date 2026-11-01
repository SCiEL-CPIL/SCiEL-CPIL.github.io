"""Program for matrix multiplication 2,2,13 - rank-minimal decomposition."""
import time
import numpy as np

DEADLINE = time.time() + 140.0

def construct():
    """
    Find rank-minimal exact tensor decomposition for 2x2 @ 2x13.
    
    Uses Strassen decomposition for 6 pairs of columns (rank 7 each)
    plus direct decomposition for the last column (rank 4).
    Total rank: 6*7 + 4 = 46.
    """
    n, m, p = 2, 2, 13
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    # Process 6 chunks of 2 columns each (columns 0-11)
    for t in range(6):
        col0 = 2 * t
        col1 = 2 * t + 1
        
        # Strassen 7 products
        # U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # V indices: 0*13+k, 1*13+k
        # W indices: k*2+i
        
        # p1 = (a00+a11)*(b00+b11)
        U1 = [1, 0, 0, 1]
        V1 = [0] * 26
        V1[0 * p + col0] = 1
        V1[1 * p + col1] = 1
        W1 = [0] * 26
        W1[col0 * n + 0] = 1
        W1[col1 * n + 1] = 1
        
        U_rows.append(U1)
        V_rows.append(V1)
        W_rows.append(W1)
        
        # p2 = (a10+a11)*b00
        U2 = [0, 0, 1, 1]
        V2 = [0] * 26
        V2[0 * p + col0] = 1
        W2 = [0] * 26
        W2[col0 * n + 1] = 1
        W2[col1 * n + 1] = -1
        
        U_rows.append(U2)
        V_rows.append(V2)
        W_rows.append(W2)
        
        # p3 = a00*(b10-b00)
        U3 = [1, 0, 0, 0]
        V3 = [0] * 26
        V3[1 * p + col0] = 1
        V3[0 * p + col0] = -1
        W3 = [0] * 26
        W3[col1 * n + 0] = 1
        W3[col1 * n + 1] = 1
        
        U_rows.append(U3)
        V_rows.append(V3)
        W_rows.append(W3)
        
        # p4 = a11*(b01-b11)
        U4 = [0, 0, 0, 1]
        V4 = [0] * 26
        V4[0 * p + col1] = 1
        V4[1 * p + col1] = -1
        W4 = [0] * 26
        W4[col0 * n + 0] = 1
        W4[col0 * n + 1] = 1
        
        U_rows.append(U4)
        V_rows.append(V4)
        W_rows.append(W4)
        
        # p5 = (a00+a01)*b11
        U5 = [1, 1, 0, 0]
        V5 = [0] * 26
        V5[1 * p + col1] = 1
        W5 = [0] * 26
        W5[col0 * n + 0] = -1
        W5[col1 * n + 0] = 1
        
        U_rows.append(U5)
        V_rows.append(V5)
        W_rows.append(W5)
        
        # p6 = (a10-a00)*(b00+b01)
        U6 = [-1, 0, 1, 0]
        V6 = [0] * 26
        V6[0 * p + col0] = 1
        V6[0 * p + col1] = 1
        W6 = [0] * 26
        W6[col1 * n + 1] = 1
        
        U_rows.append(U6)
        V_rows.append(V6)
        W_rows.append(W6)
        
        # p7 = (a01-a11)*(b10+b11)
        U7 = [0, 1, 0, -1]
        V7 = [0] * 26
        V7[1 * p + col0] = 1
        V7[1 * p + col1] = 1
        W7 = [0] * 26
        W7[col0 * n + 0] = 1
        
        U_rows.append(U7)
        V_rows.append(V7)
        W_rows.append(W7)
        
    # Last column (12)
    col_last = 12
    for i in range(n):
        for j in range(m):
            U_term = [0] * 4
            U_term[i * m + j] = 1
            V_term = [0] * 26
            V_term[j * p + col_last] = 1
            W_term = [0] * 26
            W_term[col_last * n + i] = 1
            U_rows.append(U_term)
            V_rows.append(V_term)
            W_rows.append(W_term)
            
    U = np.array(U_rows, dtype=float).T
    V = np.array(V_rows, dtype=float).T
    W = np.array(W_rows, dtype=float).T
    
    # Verify
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                error += abs(val - 1.0)
                
    if error > 1e-10:
        # Fallback
        rank_school = n * m * p
        U_school = np.zeros((n * m, rank_school))
        V_school = np.zeros((m * p, rank_school))
        W_school = np.zeros((p * n, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j, col] = 1
                    V_school[j * p + k, col] = 1
                    W_school[k * n + i, col] = 1
                    col += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
