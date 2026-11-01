import numpy as np

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the optimal rank-7 decomposition for 2x2 matrix multiplication
    (Strassen's algorithm). Rank 6 is impossible, so we return the known
    optimal solution immediately.
    
    Interface: construct() -> {'U', 'V', 'W'}
    Score is seed/rank. Rank 7 gives ~1.1429.
    """
    rank = 7
    
    # Initialize matrices as numpy arrays for easy manipulation
    U = np.zeros((4, rank), dtype=float)
    V = np.zeros((4, rank), dtype=float)
    W = np.zeros((4, rank), dtype=float)
    
    # Strassen's 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Indices:
    # U (A): 0:A00, 1:A01, 2:A10, 3:A11
    # V (B): 0:B00, 1:B01, 2:B10, 3:B11
    # W (C): 0:C00, 1:C10, 2:C01, 3:C11 (Convention: W[k*n+i, r] -> W[2*k+i, r])
    # Note: C[i,j] corresponds to W[j*2 + i]
    # C00 -> W[0]
    # C10 -> W[1]
    # C01 -> W[2]
    # C11 -> W[3]

    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    # Contributes to C00 and C11
    # C00 = ... + M1 ...
    # C11 = ... + M1 ...
    # Wait, standard Strassen:
    # C00 = M1 + M4 - M5 + M7
    # C10 = M2 + M4
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 + M6
    
    # Let's re-verify the W assignments based on the formulas above.
    
    # Column 0: M1
    W[0][0] = 1   # C00
    W[3][0] = 1   # C11
    
    # Column 1: M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    # C10 = M2 + M4
    # C11 = ... - M2 ...
    W[1][1] = 1   # C10
    W[3][1] = -1  # C11
    
    # Column 2: M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    # C01 = M3 + M5
    # C11 = ... + M3 ...
    W[2][2] = 1   # C01
    W[3][2] = 1   # C11
    
    # Column 3: M4 = A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    # C00 = ... + M4 ...
    # C10 = M2 + M4
    W[0][3] = 1   # C00
    W[1][3] = 1   # C10
    
    # Column 4: M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    # C00 = ... - M5 ...
    # C01 = M3 + M5
    W[0][4] = -1  # C00
    W[2][4] = 1   # C01
    
    # Column 5: M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    # C11 = ... + M6 ...
    W[3][5] = 1   # C11
    
    # Column 6: M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    # C00 = ... + M7 ...
    W[0][6] = 1   # C00
    
    # Convert to lists of lists to ensure compatibility with "integer/half-integer entries"
    # and standard Python types if required by the evaluator.
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# EVOLVE-BLOCK-END
