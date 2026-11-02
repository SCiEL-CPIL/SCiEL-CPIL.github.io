import numpy as np
import time

def construct():
    _deadline = time.time() + 110  # safety margin before 120s limit
    n, m, p = 2, 2, 16
    num_blocks = p // 2  # 8
    
    # Strassen factors for 2x2x2
    # U: 4x7, V: 4x7, W: 4x7
    # Rows of U: (0,0), (1,0), (0,1), (1,1) -> indices 0, 1, 2, 3 ?
    # Standard ordering: i*m + j.
    # i=0, j=0 -> 0
    # i=0, j=1 -> 1
    # i=1, j=0 -> 2
    # i=1, j=1 -> 3
    
    # Let's define the 7 factors for Strassen 2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # U entries (i, j, coeff)
    # Row index = i*2 + j
    def add_u(i, j, coeff, col):
        U_s[i*2 + j, col] += coeff
        
    def add_v(j, k, coeff, col):
        # V row index for 2x2 block: j*2 + k
        V_s[j*2 + k, col] += coeff
        
    def add_w(i, k, coeff, col):
        # W row index for 2x2 block: k*2 + i  (Output C is n x p, here 2x2. Row k, Col i? No, C[i,k]. Trace convention W[k*n+i]. So row = k*2 + i)
        W_s[k*2 + i, col] += coeff

    # Factor 1
    add_u(0,0,1,0); add_u(1,1,1,0)
    add_v(0,0,1,0); add_v(1,1,1,0)
    add_w(0,0,1,0); add_w(1,1,1,0)
    
    # Factor 2
    add_u(1,0,1,1); add_u(1,1,1,1)
    add_v(0,0,1,1)
    add_w(1,0,1,1); add_w(1,1,-1,1)
    
    # Factor 3
    add_u(0,0,1,2)
    add_v(0,1,1,2); add_v(1,1,-1,2)
    add_w(0,1,1,2); add_w(1,1,1,2)
    
    # Factor 4
    add_u(1,1,1,3)
    add_v(1,0,1,3); add_v(0,0,-1,3)
    add_w(0,0,1,3); add_w(1,0,1,3)
    
    # Factor 5
    add_u(0,0,1,4); add_u(0,1,1,4)
    add_v(1,1,1,4)
    add_w(0,0,-1,4); add_w(0,1,1,4)
    
    # Factor 6
    add_u(1,0,1,5); add_u(0,0,-1,5)
    add_v(0,0,1,5); add_v(0,1,1,5)
    add_w(1,1,1,5)
    
    # Factor 7
    add_u(0,1,1,6); add_u(1,1,-1,6)
    add_v(1,0,1,6); add_v(1,1,1,6)
    add_w(0,0,1,6)
    
    # Now expand to 2x2x16
    # U: 4x7 -> 4x56
    # V: 4x7 -> 32x56
    # W: 4x7 -> 32x56
    
    # U is just repeated 8 times?
    # U_global[i, 7t+s] = U_s[i, s]
    U = np.tile(U_s, (1, num_blocks))
    
    # V_global[j*16 + (2t+k), 7t+s] = V_s[j*2+k, s]
    # V_s rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V_global rows:
    # t=0:
    #   j=0, k=0 -> row 0. V_s row 0.
    #   j=0, k=1 -> row 1. V_s row 1.
    #   j=1, k=0 -> row 16. V_s row 2.
    #   j=1, k=1 -> row 17. V_s row 3.
    # t=1:
    #   j=0, k=0 -> row 2. V_s row 0.
    #   j=0, k=1 -> row 3. V_s row 1.
    #   j=1, k=0 -> row 18. V_s row 2.
    #   j=1, k=1 -> row 19. V_s row 3.
    
    # This is not a simple Kronecker product with I_8.
    # V_s is 4x7.
    # We want V to be 32x56.
    # Let's construct V by blocks.
    V = np.zeros((m * p, 7 * num_blocks))
    W = np.zeros((p * n, 7 * num_blocks))
    
    for t in range(num_blocks):
        # V block for t
        # Rows in V_s: 0,1,2,3 correspond to (0,0), (0,1), (1,0), (1,1)
        # Rows in V_global:
        # (0, 2t) -> 0*16 + 2t = 2t
        # (0, 2t+1) -> 0*16 + 2t+1 = 2t+1
        # (1, 2t) -> 1*16 + 2t = 16+2t
        # (1, 2t+1) -> 1*16 + 2t+1 = 17+2t
        
        # V_s row 0 -> V_global row 2t
        # V_s row 1 -> V_global row 2t+1
        # V_s row 2 -> V_global row 16+2t
        # V_s row 3 -> V_global row 17+2t
        
        cols = slice(7*t, 7*t+7)
        
        V[2*t, cols] = V_s[0, :]
        V[2*t+1, cols] = V_s[1, :]
        V[16+2*t, cols] = V_s[2, :]
        V[17+2*t, cols] = V_s[3, :]
        
        # W block for t
        # W_s rows: 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1) ?
        # W_s row index = k*2 + i.
        # k=0, i=0 -> 0
        # k=0, i=1 -> 1
        # k=1, i=0 -> 2
        # k=1, i=1 -> 3
        
        # W_global row index = k_global * n + i = (2t+k)*2 + i = 4t + 2k + i.
        # W_s row 0 (k=0, i=0) -> W_global row 4t + 0 + 0 = 4t
        # W_s row 1 (k=0, i=1) -> W_global row 4t + 0 + 1 = 4t+1
        # W_s row 2 (k=1, i=0) -> W_global row 4t + 2 + 0 = 4t+2
        # W_s row 3 (k=1, i=1) -> W_global row 4t + 2 + 1 = 4t+3
        
        W[4*t, cols] = W_s[0, :]
        W[4*t+1, cols] = W_s[1, :]
        W[4*t+2, cols] = W_s[2, :]
        W[4*t+3, cols] = W_s[3, :]
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
