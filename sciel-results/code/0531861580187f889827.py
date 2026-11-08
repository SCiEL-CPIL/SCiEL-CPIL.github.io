import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # We will implement rank-49 for 4x4x4 and use it twice.
    # First half: B cols 0-3. Second half: B cols 4-7.
    
    # Strassen terms for 4x4x4
    # A blocks: A11, A12, A21, A22 (2x2 each)
    # B blocks: B11, B12, B21, B22 (2x2 each)
    
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Rank-7 decomposition for 2x2x2
    # X, Y, Z are 2x2.
    # P1 = (X11+X22)(Y11+Y22) -> w=[1,0,0,1]
    # P2 = (X21+X22)Y11 -> w=[0,0,1,0]
    # P3 = X11(Y12-Y22) -> w=[0,1,0,0]
    # P4 = X22(Y21-Y11) -> w=[1,0,1,0]
    # P5 = (X11+X12)Y22 -> w=[-1,1,0,0]
    # P6 = (X21-X11)(Y11+Y12) -> w=[0,0,0,1]
    # P7 = (X12-X22)(Y21+Y22) -> w=[1,0,0,1]
    
    # u vectors:
    # u1 = [1,0,0,1]
    # u2 = [0,0,1,1]
    # u3 = [1,0,0,0]
    # u4 = [0,0,0,1]
    # u5 = [1,1,0,0]
    # u6 = [-1,0,1,0]
    # u7 = [0,1,0,-1]
    
    # v vectors:
    # v1 = [1,0,0,1]
    # v2 = [1,0,0,0]
    # v3 = [0,1,0,-1]
    # v4 = [0,0,1,-1]
    # v5 = [0,0,0,1]
    # v6 = [1,1,0,0]
    # v7 = [0,0,1,1]
    
    # We need to combine these for each M_l.
    # For M1, X = A11+A22, Y = B11+B22.
    # This is not a single 2x2x2 matmul with constant coefficients.
    # The coefficients depend on the block.
    
    # I will iterate through the 7 M_l.
    # For each, I will add 7 terms.
    # Total 49 terms.
    
    # I need to map the local indices (0..3) to global indices.
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
