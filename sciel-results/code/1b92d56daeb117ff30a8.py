import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # We are building a decomposition of rank R
    # Part 1: 2x4x4 using Strassen(2x2x2) x Schoolbook(1x2x2)
    # Part 2: 1x4x4 using Schoolbook(1x2x2) x Schoolbook(1x2x2)
    
    # Strassen 2x2x2
    # U: 4x7, V: 4x7, W: 4x7
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # Strassen terms
    # 1: (A11+A22)(B11+B22)
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    W_s[0, 0] = 1; W_s[3, 0] = 1
    
    # 2: (A21+A22)B11
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    W_s[2, 1] = 1; W_s[3, 1] = 1 # Wait, W indices for C21, C22?
    # C = AB.
    # C11 = M1 - M2 - M4 + M7
    # C12 = M3 + M4
    # C21 = M1 + M2 - M5 - M7
    # C22 = M3 + M5 - M1 + M6
    # Let's use the standard Strassen coefficients.
    
    # It is safer to generate the schoolbook and Strassen tensors explicitly.
    
    pass
