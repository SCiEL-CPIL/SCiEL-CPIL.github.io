import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=5, p=6
    
    # Rank calculation:
    # Part 1: 2x1x6 -> 12
    # Part 2: 2x4x6 -> 2 * (3 * 7) = 42
    # Total Rank = 54
    rank = 54
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen block
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # Local indices: i in 0,1; j in 0,1; k in 0,1
        # Global: I = i + i_off, J = j + j_off, K = k + k_off
        
        # U indices: I * m + J
        # V indices: J * p + K
        # W indices: K * n + I
        
        # M1: (A00+A11)(B00+B11)
        # U: (0,0)=1, (1,1)=1
        # V: (0,0)=1, (1,1)=1
        # W: C00=1, C11=1
        U[(0 + i_off) * m + (0 + j_off), col] = 1
        U[(1 + i_off) * m + (1 + j_off), col] = 1
        V[(0 + j_off) * p + (0 + k_off), col] = 1
        V[(1 + j_off) * p + (1 + k_off), col] = 1
        W[(0 + k_off) * n + (0 + i_off), col] = 1
        W[(1 + k_off) * n + (1 + i_off), col] = 1
        col += 1
        
        # M2: (A10+A11)B00
        # U: (1,0)=1, (1,1)=1
        # V: (0,0)=1
        # W: C10=1, C11=-1
        U[(1 + i_off) * m + (0 + j_off), col] = 1
        U[(1 + i_off) * m + (1 + j_off), col] = 1
        V[(0 + j_off) * p + (0 + k_off), col] = 1
        W[(0 + k_off) * n + (1 + i_off), col] = 1
        W[(1 + k_off) * n + (1 + i_off), col] = -1
        col += 1
        
        # M3: A00(B01-B11)
        # U: (0,0)=1
        # V: (0,1)=1, (1,1)=-1
        # W: C01=1, C11=1
        U[(0 + i_off) * m + (0 + j_off), col] = 1
        V[(0 + j_off) * p + (1 + k_off), col] = 1
        V[(1 + j_off) * p + (1 + k_off), col] = -1
        W[(1 + k_off) * n + (0 + i_off), col] = 1
        W[(1 + k_off) * n + (1 + i_off), col] = 1
        col += 1
        
        # M4: A11(B10-B00)
        # U: (1,1)=1
        # V: (1,0)=1, (0,0)=-1
        # W: C00=1, C10=1
        U[(1 + i_off) * m + (1 + j_off), col] = 1
        V[(1 + j_off) * p + (0 + k_off), col] = 1
        V[(0 + j_off) * p + (0 + k_off), col] = -1
        W[(0 + k_off) * n + (0 + i_off), col] = 1
        W[(0 + k_off) * n + (1 + i_off), col] = 1
        col += 1
        
        # M5: (A00+A01)B11
        # U: (0,0)=1, (0,1)=1
        # V: (1,1)=1
        # W: C00=-1, C01=1
        U[(0 + i_off) * m + (0 + j_off), col] = 1
        U[(0 + i_off) * m + (1 + j_off), col] = 1
        V[(1 + j_off) * p + (1 + k_off), col] = 1
        W[(0 + k_off) * n + (0 + i_off), col] = -1
        W[(1 + k_off) * n + (0 + i_off), col] = 1
        col += 1
        
        # M6: (A10-A00)(B00+B01)
        # U: (1,0)=1, (0,0)=-1
        # V: (0,0)=1, (0,1)=1
        # W: C11=1
        U[(1 + i_off) * m + (0 + j_off), col] = 1
        U[(0 + i_off) * m + (0 + j_off), col] = -1
        V[(0 + j_off) * p + (0 + k_off), col] = 1
        V[(0 + j_off) * p + (1 + k_off), col] = 1
        W[(1 + k_off) * n + (1 + i_off), col] = 1
        col += 1
        
        # M7: (A01-A11)(B10+B11)
        # U: (0,1)=1, (1,1)=-1
        # V: (1,0)=1, (1,1)=1
        # W: C00=1
        U[(0 + i_off) * m + (1 + j_off), col] = 1
        U[(1 + i_off) * m + (1 + j_off), col] = -1
        V[(1 + j_off) * p + (0 + k_off), col] = 1
        V[(1 + j_off) * p + (1 + k_off), col] = 1
        W[(0 + k_off) * n + (0 + i_off), col] = 1
        col += 1

    # Part 1: 2x1x6 (j=4)
    # A is 2x1, B is 1x6.
    # C_ik = A_i4 * B_4k
    for i in range(n):
        for k in range(p):
            U[i * m + 4, col] = 1
            V[4 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
            
    # Part 2: 2x4x6 (j=0,1,2,3)
    # Split j into {0,1} and {2,3}
    # For each j_block in [0, 2]:
    for j_block in [0, 2]:
        # Split k into {0,1}, {2,3}, {4,5}
        for k_block in [0, 2, 4]:
            add_strassen(0, j_block, k_block)
            
    return {"U": U, "V": V, "W": W}
