import numpy as np
import time

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank-21 Strassen-based decomposition
    rank = 21
    U = np.zeros((4, rank))
    V = np.zeros((12, rank))
    W = np.zeros((12, rank))
    
    for t in range(3):
        base = t * 7
        # M1 = (A00+A11)*(B00+B11)
        U[0, base] = 1
        U[3, base] = 1
        V[2*t, base] = 1
        V[6+2*t, base] = 1
        W[4*t, base] = 1
        W[4*t+3, base] = 1
        
        # M2 = (A10+A11)*B00
        U[2, base+1] = 1
        U[3, base+1] = 1
        V[2*t, base+1] = 1
        W[4*t+1, base+1] = 1
        W[4*t+3, base+1] = -1
        
        # M3 = A00*(B01-B11)
        U[0, base+2] = 1
        V[2*t+1, base+2] = 1
        V[6+2*t+1, base+2] = -1
        W[4*t+2, base+2] = 1
        W[4*t+3, base+2] = 1
        
        # M4 = A11*(B10-B00)
        U[3, base+3] = 1
        V[6+2*t, base+3] = 1
        V[2*t, base+3] = -1
        W[4*t, base+3] = 1
        W[4*t+1, base+3] = 1
        
        # M5 = (A00+A01)*B11
        U[0, base+4] = 1
        U[1, base+4] = 1
        V[6+2*t+1, base+4] = 1
        W[4*t, base+4] = -1
        W[4*t+2, base+4] = 1
        
        # M6 = (A10-A00)*(B00+B01)
        U[2, base+5] = 1
        U[0, base+5] = -1
        V[2*t, base+5] = 1
        V[2*t+1, base+5] = 1
        W[4*t+3, base+5] = 1
        
        # M7 = (A01-A11)*(B10+B11)
        U[1, base+6] = 1
        U[3, base+6] = -1
        V[6+2*t, base+6] = 1
        V[6+2*t+1, base+6] = 1
        W[4*t, base+6] = 1
    
    U_list = [[int(U[i,j]) for j in range(rank)] for i in range(4)]
    V_list = [[int(V[i,j]) for j in range(rank)] for i in range(12)]
    W_list = [[int(W[i,j]) for j in range(rank)] for i in range(12)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
