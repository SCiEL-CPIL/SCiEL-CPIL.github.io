import numpy as np

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the optimal rank-7 decomposition for 2x2 matrix multiplication
    (Strassen's algorithm). 
    
    This implementation structurally constructs the factor matrices by explicitly
    defining the linear combinations for Strassen's 7 products.
    
    Interface: construct() -> {'U', 'V', 'W'}
    Score is seed/rank. Rank 7 gives 1.1429.
    """
    rank = 7
    n = 2
    m = 2
    p = 2
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n*m)]
    V = [[0] * rank for _ in range(m*p)]
    W = [[0] * rank for _ in range(p*n)]
    
    # Helper to set values
    def set_u(row, col, val):
        U[row][col] = val
    def set_v(row, col, val):
        V[row][col] = val
    def set_w(row, col, val):
        W[row][col] = val
        
    # Strassen's 7 products definitions
    # M1 = (A00+A11)(B00+B11)
    set_u(0, 0, 1) # A00
    set_u(3, 0, 1) # A11
    set_v(0, 0, 1) # B00
    set_v(3, 0, 1) # B11
    
    # M2 = (A10+A11)B00
    set_u(2, 1, 1) # A10
    set_u(3, 1, 1) # A11
    set_v(0, 1, 1) # B00
    
    # M3 = A00(B01-B11)
    set_u(0, 2, 1) # A00
    set_v(1, 2, 1) # B01
    set_v(3, 2, -1) # B11
    
    # M4 = A11(B10-B00)
    set_u(3, 3, 1) # A11
    set_v(2, 3, 1) # B10
    set_v(0, 3, -1) # B00
    
    # M5 = (A00+A01)B11
    set_u(0, 4, 1) # A00
    set_u(1, 4, 1) # A01
    set_v(3, 4, 1) # B11
    
    # M6 = (A10-A00)(B00+B01)
    set_u(2, 5, 1) # A10
    set_u(0, 5, -1) # A00
    set_v(0, 5, 1) # B00
    set_v(1, 5, 1) # B01
    
    # M7 = (A01-A11)(B10+B11)
    set_u(1, 6, 1) # A01
    set_u(3, 6, -1) # A11
    set_v(2, 6, 1) # B10
    set_v(3, 6, 1) # B11
    
    # W reconstruction
    # C00 (Index 0) = M1 + M4 - M5 + M7
    set_w(0, 0, 1)
    set_w(0, 3, 1)
    set_w(0, 4, -1)
    set_w(0, 6, 1)
    
    # C10 (Index 1) = M2 + M4
    set_w(1, 1, 1)
    set_w(1, 3, 1)
    
    # C01 (Index 2) = M3 + M5
    set_w(2, 2, 1)
    set_w(2, 4, 1)
    
    # C11 (Index 3) = M1 - M2 + M3 + M6
    set_w(3, 0, 1)
    set_w(3, 1, -1)
    set_w(3, 2, 1)
    set_w(3, 5, 1)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
