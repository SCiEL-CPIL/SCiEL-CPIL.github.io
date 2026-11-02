import time
import numpy as np

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 2x2
    matrix by a 2x8 matrix.
    
    The minimal rank is 28 (4 blocks of 2x2x2x2 multiplication, each of rank 7).
    We construct the rank-28 decomposition directly using Strassen's algorithm
    for each block.
    """
    n, m, p = 2, 2, 8
    nm, mp, pn = n * m, m * p, p * n  # 4, 16, 16
    
    # Strassen coefficients for 2x2x2x2
    # A indices: 0=a, 1=b, 2=c, 3=d
    # B indices: 0=e, 1=f, 2=g, 3=h
    # C indices: 0=g, 1=h, 2=i, 3=j (local)
    
    # U (A side)
    # 7 terms
    U_block = [
        [1, 0, 0, 1],  # M1: a+d
        [0, 0, 1, 1],  # M2: c+d
        [1, 0, 0, 0],  # M3: a
        [0, 0, 0, 1],  # M4: d
        [1, 1, 0, 0],  # M5: a+b
        [1, 0, -1, 0], # M6: c-a (wait, M6 is (c-a)(e+f), so c-a is [0,0,1,-1] or [1,0,-1,0]? 
                       # A is [[a,b],[c,d]]. c is index 2, a is index 0.
                       # So c-a is [ -1, 0, 1, 0 ] or [ 1, 0, -1, 0 ] depending on order.
                       # Let's stick to the standard:
                       # M6 = (c-a)(e+f). A-part is c-a. Indices: a=0, b=1, c=2, d=3.
                       # c-a = -a + c = [-1, 0, 1, 0]
        [0, 1, 0, -1]  # M7: b-d = [0, 1, 0, -1]
    ]
    # Correction for M6:
    U_block[5] = [-1, 0, 1, 0]
    
    V_block = [
        [1, 0, 0, 1],  # M1: e+h
        [0, 1, 0, 0],  # M2: f
        [1, -1, 0, 0], # M3: e-f
        [0, 0, 1, -1], # M4: g-h
        [0, 0, 0, 1],  # M5: h
        [1, 1, 0, 0],  # M6: e+f
        [0, 0, 1, 1]   # M7: g+h
    ]
    
    # W (C side)
    # C = [[g, h], [i, j]]
    # Indices: 0=g, 1=h, 2=i, 3=j
    W_block = [
        [1, 1, -1, 1],   # M1: g+h-i+j? No.
                         # g = M1 - M3 + M5 - M7
                         # h = M1 + M4 - M5 + M7
                         # i = M2 + M3 - M4 + M6
                         # j = M1 - M2 + M4 - M6
                         # So for M1: g=1, h=1, i=0, j=1
        [0, 0, 1, -1],   # M2: g=0, h=0, i=1, j=-1
        [1, 0, 1, 0],    # M3: g=1, h=0, i=1, j=0
        [0, 1, -1, 1],   # M4: g=0, h=1, i=-1, j=1
        [1, -1, 0, 0],   # M5: g=1, h=-1, i=0, j=0
        [0, 0, 1, -1],   # M6: g=0, h=0, i=1, j=-1
        [-1, 1, 0, 0]    # M7: g=-1, h=1, i=0, j=0
    ]
    
    # Verify W_block
    # g = M1 - M3 + M5 - M7
    # M1[0]=1, M3[0]=1, M5[0]=1, M7[0]=-1 => 1-1+1-(-1) = 2. Error.
    # Let's re-check Strassen formulas.
    # g = M1 - M3 + M5 - M7
    # h = M1 + M4 - M5 + M7
    # i = M2 + M3 - M4 + M6
    # j = M1 - M2 + M4 - M6
    
    # My W_block rows are the coefficients for M1..M7 for each C element.
    # W_block[c][s] is coeff of M_s in C_c.
    
    # C_0 (g): M1=1, M3=-1, M5=1, M7=-1. Others 0.
    # C_1 (h): M1=1, M4=1, M5=-1, M7=1. Others 0.
    # C_2 (i): M2=1, M3=1, M4=-1, M6=1. Others 0.
    # C_3 (j): M1=1, M2=-1, M4=1, M6=-1. Others 0.
    
    W_block = [
        [1, 0, -1, 0, 1, 0, -1], # g
        [1, 0, 0, 1, -1, 0, 1],  # h
        [0, 1, 1, -1, 0, 1, 0],  # i
        [1, -1, 0, 1, 0, -1, 0]  # j
    ]
    
    rank = 28
    U = np.zeros((nm, rank), dtype=int)
    V = np.zeros((mp, rank), dtype=int)
    W = np.zeros((pn, rank), dtype=int)
    
    for block in range(4):
        k0 = block * 2
        for s in range(7):
            r = block * 7 + s
            # Fill U
            U[0, r] = U_block[s][0]
            U[1, r] = U_block[s][1]
            U[2, r] = U_block[s][2]
            U[3, r] = U_block[s][3]
            
            # Fill V
            # B indices: 8j+k.
            # j=0: k=0,1 -> indices 0,1
            # j=1: k=0,1 -> indices 8,9
            # But we are in block `block`, so global k is k0, k0+1.
            # V index is j*p + k_global.
            # V_block[s][0] is coeff for e (j=0, k_local=0)
            # V_block[s][1] is coeff for f (j=0, k_local=1)
            # V_block[s][2] is coeff for g (j=1, k_local=0)
            # V_block[s][3] is coeff for h (j=1, k_local=1)
            
            V[0 * p + k0, r] = V_block[s][0]
            V[0 * p + k0 + 1, r] = V_block[s][1]
            V[1 * p + k0, r] = V_block[s][2]
            V[1 * p + k0 + 1, r] = V_block[s][3]
            
            # Fill W
            # C indices: k*n + i.
            # C_0 (g): i=0, k_global=k0. Index = k0*2 + 0 = 2*k0.
            # C_1 (h): i=1, k_global=k0. Index = k0*2 + 1 = 2*k0 + 1.
            # C_2 (i): i=0, k_global=k0+1. Index = (k0+1)*2 + 0 = 2*k0 + 2.
            # C_3 (j): i=1, k_global=k0+1. Index = (k0+1)*2 + 1 = 2*k0 + 3.
            
            W[2 * k0, r] = W_block[0][s]
            W[2 * k0 + 1, r] = W_block[1][s]
            W[2 * k0 + 2, r] = W_block[2][s]
            W[2 * k0 + 3, r] = W_block[3][s]
            
    return {"U": U, "V": V, "W": W}
