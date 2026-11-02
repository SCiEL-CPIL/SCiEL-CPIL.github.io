import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Outer Strassen factors for 2x2 block matrices
    # U_outer[r] is 2x2, V_outer[r] is 2x2, W_outer[r] is 2x2
    # Indices 0,1 for block rows/cols
    
    U_outer = [
        np.array([[1, 0], [0, 1]]),  # M1: A00 + A11
        np.array([[0, 0], [1, 1]]),  # M2: A10 + A11
        np.array([[1, 0], [0, 0]]),  # M3: A00
        np.array([[0, 0], [0, 1]]),  # M4: A11
        np.array([[1, 1], [0, 0]]),  # M5: A00 + A01
        np.array([[-1, 0], [1, 0]]), # M6: A10 - A00
        np.array([[0, 1], [0, -1]])  # M7: A01 - A11
    ]
    
    V_outer = [
        np.array([[1, 0], [0, 1]]),  # M1: B00 + B11
        np.array([[1, 0], [0, 0]]),  # M2: B00
        np.array([[0, 1], [0, -1]]), # M3: B01 - B11
        np.array([[-1, 0], [1, 0]]), # M4: B10 - B00
        np.array([[0, 0], [0, 1]]),  # M5: B11
        np.array([[1, 1], [0, 0]]),  # M6: B00 + B01
        np.array([[0, 0], [1, 1]])   # M7: B10 + B11
    ]
    
    W_outer = [
        np.array([[1, 0], [0, 1]]),  # M1: C00 += 1, C11 += 1
        np.array([[0, 0], [1, 0]]),  # M2: C10 += 1
        np.array([[0, 1], [0, 0]]),  # M3: C01 += 1
        np.array([[1, 0], [1, 0]]),  # M4: C00 += 1, C10 += 1
        np.array([[-1, 1], [0, 0]]), # M5: C00 += -1, C01 += 1
        np.array([[0, 0], [0, 1]]),  # M6: C11 += 1
        np.array([[1, 0], [0, -1]])  # M7: C00 += 1, C11 += -1
    ]
    
    # Inner Strassen factors for 2x2 scalar matrices
    U_inner = [
        np.array([[1, 0], [0, 1]]),
        np.array([[0, 0], [1, 1]]),
        np.array([[1, 0], [0, 0]]),
        np.array([[0, 0], [0, 1]]),
        np.array([[1, 1], [0, 0]]),
        np.array([[-1, 0], [1, 0]]),
        np.array([[0, 1], [0, -1]])
    ]
    
    V_inner = [
        np.array([[1, 0], [0, 1]]),
        np.array([[1, 0], [0, 0]]),
        np.array([[0, 1], [0, -1]]),
        np.array([[-1, 0], [1, 0]]),
        np.array([[0, 0], [0, 1]]),
        np.array([[1, 1], [0, 0]]),
        np.array([[0, 0], [1, 1]])
    ]
    
    W_inner = [
        np.array([[1, 0], [0, 1]]),
        np.array([[0, 0], [1, 0]]),
        np.array([[0, 1], [0, 0]]),
        np.array([[1, 0], [1, 0]]),
        np.array([[-1, 1], [0, 0]]),
        np.array([[0, 0], [0, 1]]),
        np.array([[1, 0], [0, -1]])
    ]
    
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    col = 0
    for r in range(7):
        for s in range(7):
            # Construct U_total
            U_tot = np.zeros((4, 4))
            for I in range(2):
                for J in range(2):
                    coeff = U_outer[r][I, J]
                    if coeff != 0:
                        U_tot[I*2:(I+1)*2, J*2:(J+1)*2] = coeff * U_inner[s]
                        
            V_tot = np.zeros((4, 4))
            for J in range(2):
                for K in range(2):
                    coeff = V_outer[r][J, K]
                    if coeff != 0:
                        V_tot[J*2:(J+1)*2, K*2:(K+1)*2] = coeff * V_inner[s]
                        
            W_tot = np.zeros((4, 4))
            for K in range(2):
                for I in range(2):
                    coeff = W_outer[r][K, I]
                    if coeff != 0:
                        W_tot[K*2:(K+1)*2, I*2:(I+1)*2] = coeff * W_inner[s]
                        
            # Flatten and assign
            # U row index: i*4 + j
            U[:, col] = U_tot.flatten(order='C')
            # V row index: j*4 + k
            V[:, col] = V_tot.flatten(order='C')
            # W row index: k*4 + i
            W[:, col] = W_tot.flatten(order='C')
            
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
