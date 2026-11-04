"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Construct a rank-49 tensor decomposition for 4x4 matrix multiplication
    using the Kronecker product of the 2x2 Strassen algorithm.
    """
    n, m, p = DIMENSIONS
    
    # Strassen 2x2 factors (4x7 matrices)
    # Indices for U: i*2 + j
    # Indices for V: j*2 + k
    # Indices for W: k*2 + i
    
    SU = np.zeros((4, 7), dtype=int)
    # M1: A00 + A11
    SU[0, 0] = 1
    SU[3, 0] = 1
    # M2: A10 + A11
    SU[2, 1] = 1
    SU[3, 1] = 1
    # M3: A00
    SU[0, 2] = 1
    # M4: A11
    SU[3, 3] = 1
    # M5: A00 + A01
    SU[0, 4] = 1
    SU[1, 4] = 1
    # M6: A10 - A00
    SU[2, 5] = 1
    SU[0, 5] = -1
    # M7: A01 - A11
    SU[1, 6] = 1
    SU[3, 6] = -1

    SV = np.zeros((4, 7), dtype=int)
    # M1: B00 + B11
    SV[0, 0] = 1
    SV[3, 0] = 1
    # M2: B00
    SV[0, 1] = 1
    # M3: B01 - B11
    SV[1, 2] = 1
    SV[3, 2] = -1
    # M4: B10 - B00
    SV[2, 3] = 1
    SV[0, 3] = -1
    # M5: B11
    SV[3, 4] = 1
    # M6: B00 + B01
    SV[0, 5] = 1
    SV[1, 5] = 1
    # M7: B10 + B11
    SV[2, 6] = 1
    SV[3, 6] = 1

    SW = np.zeros((4, 7), dtype=int)
    # Row 0: C11 (k=0, i=0) = M1 + M4 - M5 + M7
    SW[0, 0] = 1
    SW[0, 3] = 1
    SW[0, 4] = -1
    SW[0, 6] = 1
    # Row 1: C12 (k=0, i=1) = M3 + M5
    SW[1, 2] = 1
    SW[1, 4] = 1
    # Row 2: C21 (k=1, i=0) = M2 + M4
    SW[2, 1] = 1
    SW[2, 3] = 1
    # Row 3: C22 (k=1, i=1) = M1 - M2 + M3 + M6
    SW[3, 0] = 1
    SW[3, 1] = -1
    SW[3, 2] = 1
    SW[3, 5] = 1

    # Kronecker products to get 4x4 factors (16x49)
    U = np.kron(SU, SU)
    V = np.kron(SV, SV)
    W = np.kron(SW, SW)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
