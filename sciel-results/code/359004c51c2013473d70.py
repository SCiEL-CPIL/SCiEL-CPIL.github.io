import numpy as np
import time

def construct():
    # Matrix multiplication 2x2 * 2x10
    # A is 2x2, B is 2x10, C = A*B is 2x10
    # n=2 (rows of A), m=2 (cols of A / rows of B), p=10 (cols of B)
    # U: (n*m, rank) = (4, rank)
    # V: (m*p, rank) = (20, rank)
    # W: (p*n, rank) = (20, rank)
    # Convention: W[k*n + i, r] where k is column index of C (0..p-1), i is row index of C (0..n-1)
    # So W index = k*2 + i
    
    # The 2x10 matrix B can be viewed as 5 blocks of 2x2.
    # Each 2x2 * 2x2 multiplication has rank 7 (Strassen).
    # Total rank = 5 * 7 = 35.
    
    # Strassen decomposition for 2x2 * 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = A*B = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    #
    # Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U vectors (for A indices [a, b, c, d]):
    u_vecs = [
        np.array([1, 0, 0, 1]),   # M1: a+d
        np.array([0, 0, 1, 1]),   # M2: c+d
        np.array([1, 0, 0, 0]),   # M3: a
        np.array([0, 0, 0, 1]),   # M4: d
        np.array([1, 1, 0, 0]),   # M5: a+b
        np.array([-1, 0, 1, 0]),  # M6: c-a
        np.array([0, 1, 0, -1])   # M7: b-d
    ]
    
    # V vectors (for B indices [e, f, g, h]):
    v_vecs = [
        np.array([1, 0, 0, 1]),   # M1: e+h
        np.array([1, 0, 0, 0]),   # M2: e
        np.array([0, 1, 0, -1]),  # M3: f-h
        np.array([-1, 0, 1, 0]),  # M4: g-e
        np.array([0, 0, 0, 1]),   # M5: h
        np.array([1, 1, 0, 0]),   # M6: e+f
        np.array([0, 0, 1, 1])    # M7: g+h
    ]
    
    # W vectors (for C indices [C11, C21, C12, C22]):
    w_vecs = [
        np.array([1, 0, 0, 1]),   # M1: C11 + C22
        np.array([0, 1, 0, -1]),  # M2: C21 - C22
        np.array([0, 0, 1, 1]),   # M3: C12 + C22
        np.array([1, 1, 0, 0]),   # M4: C11 + C21
        np.array([-1, 0, 1, 0]),  # M5: C12 - C11
        np.array([0, 0, 0, 1]),   # M6: C22
        np.array([1, 0, 0, 0])    # M7: C11
    ]
    
    rank = 35
    U = np.zeros((4, rank))
    V = np.zeros((20, rank))
    W = np.zeros((20, rank))
    
    # 5 blocks of 2x2 in B (columns 0-1, 2-3, 4-5, 6-7, 8-9)
    col = 0
    for r in range(5):
        for k in range(7):
            U[:, col] = u_vecs[k]
            
            # V indices for block r:
            # B[0, 2r] -> index 2r
            # B[0, 2r+1] -> index 2r+1
            # B[1, 2r] -> index 10 + 2r
            # B[1, 2r+1] -> index 10 + 2r + 1
            V[2*r, col] = v_vecs[k][0]
            V[2*r+1, col] = v_vecs[k][1]
            V[10 + 2*r, col] = v_vecs[k][2]
            V[10 + 2*r + 1, col] = v_vecs[k][3]
            
            # W indices for block r:
            # C[0, 2r] -> index (2r)*2 + 0 = 4r
            # C[1, 2r] -> index (2r)*2 + 1 = 4r+1
            # C[0, 2r+1] -> index (2r+1)*2 + 0 = 4r+2
            # C[1, 2r+1] -> index (2r+1)*2 + 1 = 4r+3
            W[4*r, col] = w_vecs[k][0]
            W[4*r+1, col] = w_vecs[k][1]
            W[4*r+2, col] = w_vecs[k][2]
            W[4*r+3, col] = w_vecs[k][3]
            
            col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
