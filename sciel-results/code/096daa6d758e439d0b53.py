import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 36
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add Strassen block for M_{2,2,2}
    # i_base, j_base, k_base are starting indices
    def add_strassen(i_base, j_base, k_base):
        nonlocal col
        # P1
        U[i_base * m + j_base, col] = 1
        U[(i_base+1) * m + (j_base+1), col] = 1
        V[j_base * p + k_base, col] = 1
        V[(j_base+1) * p + (k_base+1), col] = 1
        W[k_base * n + i_base, col] = 1
        W[(k_base+1) * n + (i_base+1), col] = 1
        col += 1
        
        # P2
        U[(i_base+1) * m + j_base, col] = 1
        U[(i_base+1) * m + (j_base+1), col] = 1
        V[j_base * p + k_base, col] = 1
        W[k_base * n + (i_base+1), col] = 1
        W[(k_base+1) * n + (i_base+1), col] = -1
        col += 1
        
        # P3
        U[i_base * m + j_base, col] = 1
        V[(j_base+1) * p + k_base, col] = 1
        V[(j_base+1) * p + (k_base+1), col] = -1
        W[k_base * n + i_base, col] = 1
        W[(k_base+1) * n + i_base, col] = 1
        col += 1
        
        # P4
        U[(i_base+1) * m + (j_base+1), col] = 1
        V[j_base * p + (k_base+1), col] = 1
        V[j_base * p + k_base, col] = -1
        W[k_base * n + i_base, col] = 1
        W[k_base * n + (i_base+1), col] = 1
        col += 1
        
        # P5
        U[i_base * m + j_base, col] = 1
        U[i_base * m + (j_base+1), col] = 1
        V[(j_base+1) * p + (k_base+1), col] = 1
        W[(k_base+1) * n + i_base, col] = -1
        W[(k_base+1) * n + (i_base+1), col] = 1
        col += 1
        
        # P6
        U[(i_base+1) * m + j_base, col] = 1
        U[i_base * m + j_base, col] = -1
        V[j_base * p + k_base, col] = 1
        V[j_base * p + (k_base+1), col] = 1
        W[(k_base+1) * n + (i_base+1), col] = 1
        col += 1
        
        # P7
        U[i_base * m + (j_base+1), col] = 1
        U[(i_base+1) * m + (j_base+1), col] = -1
        V[(j_base+1) * p + k_base, col] = 1
        V[(j_base+1) * p + (k_base+1), col] = 1
        W[(k_base+1) * n + i_base, col] = 1
        W[k_base * n + i_base, col] = -1
        col += 1

    # Helper to add Schoolbook block for M_{2,2,1}
    def add_schoolbook_221(i_base, j_base, k_base):
        nonlocal col
        # a00 b00 -> c0
        U[i_base * m + j_base, col] = 1
        V[j_base * p + k_base, col] = 1
        W[k_base * n + i_base, col] = 1
        col += 1
        
        # a01 b10 -> c0
        U[i_base * m + (j_base+1), col] = 1
        V[(j_base+1) * p + k_base, col] = 1
        W[k_base * n + i_base, col] = 1
        col += 1
        
        # a10 b00 -> c1
        U[(i_base+1) * m + j_base, col] = 1
        V[j_base * p + k_base, col] = 1
        W[k_base * n + (i_base+1), col] = 1
        col += 1
        
        # a11 b10 -> c1
        U[(i_base+1) * m + (j_base+1), col] = 1
        V[(j_base+1) * p + k_base, col] = 1
        W[k_base * n + (i_base+1), col] = 1
        col += 1

    # Block 1: j in {0, 1}
    # Sub-block 1a: k in {0, 1} -> Strassen
    add_strassen(0, 0, 0)
    # Sub-block 1b: k in {2, 3, 4} -> M_{2,2,3}
    #   Sub-block 1b1: k in {2, 3} -> Strassen
    add_strassen(0, 0, 2)
    #   Sub-block 1b2: k in {4} -> Schoolbook 2x2x1
    add_schoolbook_221(0, 0, 4)

    # Block 2: j in {2, 3}
    # Sub-block 2a: k in {0, 1} -> Strassen
    add_strassen(0, 2, 0)
    # Sub-block 2b: k in {2, 3, 4} -> M_{2,2,3}
    #   Sub-block 2b1: k in {2, 3} -> Strassen
    add_strassen(0, 2, 2)
    #   Sub-block 2b2: k in {4} -> Schoolbook 2x2x1
    add_schoolbook_221(0, 2, 4)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
