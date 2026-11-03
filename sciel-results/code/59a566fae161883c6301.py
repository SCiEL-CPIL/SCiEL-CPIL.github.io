import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2 factors
    # U: 4x7, V: 4x7, W: 4x7
    sU = np.zeros((4, 7))
    sV = np.zeros((4, 7))
    sW = np.zeros((4, 7))
    
    # P1 = (a+d)(e+h)
    sU[0, 0] = 1; sU[3, 0] = 1
    sV[0, 0] = 1; sV[3, 0] = 1
    sW[0, 0] = 1 # C11
    
    # P2 = (c+d)e
    sU[2, 1] = 1; sU[3, 1] = 1
    sV[0, 1] = 1
    sW[2, 1] = 1 # C21
    
    # P3 = a(f-h)
    sU[0, 2] = 1
    sV[1, 2] = 1; sV[3, 2] = -1
    sW[1, 2] = 1 # C12
    
    # P4 = d(g-e)
    sU[3, 3] = 1
    sV[2, 3] = 1; sV[0, 3] = -1
    sW[0, 3] = 1 # C11
    
    # P5 = (a+b)h
    sU[0, 4] = 1; sU[1, 4] = 1
    sV[3, 4] = 1
    sW[0, 4] = -1 # C11
    
    # P6 = (c-a)(e+f)
    sU[2, 5] = 1; sU[0, 5] = -1
    sV[0, 5] = 1; sV[1, 5] = 1
    sW[3, 5] = 1 # C22
    
    # P7 = (b-d)(g+h)
    sU[1, 6] = 1; sU[3, 6] = -1
    sV[2, 6] = 1; sV[3, 6] = 1
    sW[3, 6] = 1 # C22
    sW[0, 6] = 1 # C11
    
    # Strassen 4x4x4 macro factors
    # M_U, M_V, M_W are 7x4
    M_U = np.zeros((7, 4))
    M_V = np.zeros((7, 4))
    M_W = np.zeros((7, 4))
    
    # L1 = A1 + A4
    M_U[0, 0] = 1; M_U[0, 3] = 1
    # L2 = A3 + A4
    M_U[1, 2] = 1; M_U[1, 3] = 1
    # L3 = A1
    M_U[2, 0] = 1
    # L4 = A4
    M_U[3, 3] = 1
    # L5 = A1 + A2
    M_U[4, 0] = 1; M_U[4, 1] = 1
    # L6 = A3 - A1
    M_U[5, 2] = 1; M_U[5, 0] = -1
    # L7 = A2 - A4
    M_U[6, 1] = 1; M_U[6, 3] = -1
    
    # R1 = B1 + B4
    M_V[0, 0] = 1; M_V[0, 3] = 1
    # R2 = B1
    M_V[1, 0] = 1
    # R3 = B2 - B4
    M_V[2, 1] = 1; M_V[2, 3] = -1
    # R4 = B3 - B1
    M_V[3, 2] = 1; M_V[3, 0] = -1
    # R5 = B4
    M_V[4, 3] = 1
    # R6 = B1 + B2
    M_V[5, 0] = 1; M_V[5, 1] = 1
    # R7 = B3 + B4
    M_V[6, 2] = 1; M_V[6, 3] = 1
    
    # C11 = P1 + P4 - P5 + P7
    M_W[0, 0] = 1; M_W[0, 3] = 1; M_W[0, 4] = -1; M_W[0, 6] = 1
    # C12 = P3 + P5
    M_W[1, 2] = 1; M_W[1, 4] = 1
    # C21 = P2 + P4
    M_W[2, 1] = 1; M_W[2, 3] = 1
    # C22 = P1 - P2 + P3 + P6
    M_W[3, 0] = 1; M_W[3, 1] = -1; M_W[3, 2] = 1; M_W[3, 5] = 1
    
    # Construct 4x4x4 factors (16x49)
    U_44 = np.zeros((16, 49))
    V_44 = np.zeros((16, 49))
    W_44 = np.zeros((16, 49))
    
    for m in range(7):
        for mp in range(7):
            col = m * 7 + mp
            # U
            for b in range(4):
                for l in range(4):
                    row = b * 4 + l
                    U_44[row, col] = M_U[m, b] * sU[l, mp]
            # V
            for b in range(4):
                for l in range(4):
                    row = b * 4 + l
                    V_44[row, col] = M_V[m, b] * sV[l, mp]
            # W
            for b in range(4):
                for l in range(4):
                    row = b * 4 + l
                    W_44[row, col] = M_W[m, b] * sW[l, mp]
    
    # Construct 4x4x8 factors
    # U: 16x98
    U = np.zeros((16, 98))
    U[:, 0:49] = U_44
    U[:, 49:98] = U_44
    
    # V: 32x98
    V = np.zeros((32, 98))
    # B1 is k=0..3 (rows 0..15)
    V[0:16, 0:49] = V_44
    # B2 is k=4..7 (rows 16..31)
    V[16:32, 49:98] = V_44
    
    # W: 32x98
    W = np.zeros((32, 98))
    # C1 is k=0..3 (rows 0..15)
    W[0:16, 0:49] = W_44
    # C2 is k=4..7 (rows 16..31)
    W[16:32, 49:98] = W_44
    
    # Convert to lists of lists with integers
    U_list = U.astype(int).tolist()
    V_list = V.astype(int).tolist()
    W_list = W.astype(int).tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
