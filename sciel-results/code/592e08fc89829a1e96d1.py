import time
import numpy as np

def construct():
    n, m, p = 4, 4, 4
    rank = 49

    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)

    # Strassen factors
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1

    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1

    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1

    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1

    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1

    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1

    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    U_arr = np.zeros((16, 49), dtype=int)
    V_arr = np.zeros((16, 49), dtype=int)
    W_arr = np.zeros((16, 49), dtype=int)
    
    # Vectorized construction
    for s in range(7):
        for t in range(7):
            r = s * 7 + t
            
            # U
            for i in range(4):
                for j in range(4):
                    coeff = U2[(i//2)*2 + (j//2), s] * U2[(i%2)*2 + (j%2), t]
                    if coeff != 0:
                        U_arr[i*4 + j, r] = coeff
            
            # V
            for k in range(4):
                for l in range(4):
                    coeff = V2[(k//2)*2 + (l//2), s] * V2[(k%2)*2 + (l%2), t]
                    if coeff != 0:
                        V_arr[k*4 + l, r] = coeff
            
            # W
            for j in range(4):
                for i in range(4):
                    coeff = W2[(i//2)*2 + (j//2), s] * W2[(i%2)*2 + (j%2), t]
                    if coeff != 0:
                        W_arr[j*4 + i, r] = coeff

    return {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}
