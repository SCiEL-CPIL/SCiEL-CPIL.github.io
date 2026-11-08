import numpy as np

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 2 by 3 matrix by a 3 by 8 matrix.
    
    The decomposition is based on:
    1. Partitioning the 3x8 matrix B into B1 (2x8) and B2 (1x8).
    2. Partitioning the 2x3 matrix A into A1 (2x2) and A2 (2x1).
    3. Computing A1 * B1 using 4 Strassen 2x2x2 decompositions (Rank 28).
    4. Computing A2 * B2 using schoolbook multiplication (Rank 16).
    Total Rank = 44.
    
    Interface:
    U: shape (n*m, rank) = (6, 44)
    V: shape (m*p, rank) = (24, 44)
    W: shape (p*n, rank) = (16, 44)
    """
    n, m, p = 2, 3, 8
    rank = 44
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen 2x2x2 coefficients
    # Indices for U (i,a), V (a,k), W (k,i) where i,a,k in {0,1}
    # Flattened indices: i*2+a, a*2+k, k*2+i
    S_U = np.array([
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ])
    S_V = np.array([
        [1, 0, 0, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1],  # M5
        [1, 1, 0, 0],  # M6
        [0, 0, 1, 1]   # M7
    ])
    S_W = np.array([
        [1, 0, 0, 1],  # M1
        [0, 1, 0, -1], # M2
        [0, 0, 1, 1],  # M3
        [1, 1, 0, 0],  # M4
        [-1, 0, 1, 0], # M5
        [0, 0, 0, 1],  # M6
        [1, 0, 0, 0]   # M7
    ])
    
    col = 0
    
    # Part 1: A1 (2x2) * B1 (2x8)
    # B1 uses rows 0,1 of B (indices 0..7 in flattened B, corresponding to a=0,1)
    # A1 uses cols 0,1 of A (indices 0..3 in flattened A, corresponding to a=0,1)
    # Output uses cols 0..7 of C (indices 0..15 in flattened C, corresponding to k=0..7, i=0,1)
    
    for block_idx in range(4):
        k0 = block_idx * 2
        for r in range(7):
            # U: A1 indices (i, a) where i in {0,1}, a in {0,1}
            # Flattened A index: i*m + a = i*3 + a
            for i in range(2):
                for a in range(2):
                    val = S_U[r, i*2 + a]
                    if val != 0:
                        U[i*3 + a, col] = val
            
            # V: B1 indices (a, k) where a in {0,1}, k in {k0, k0+1}
            # Flattened B index: a*p + k = a*8 + k
            for a in range(2):
                for k in range(2):
                    val = S_V[r, a*2 + k]
                    if val != 0:
                        V[a*8 + (k0 + k), col] = val
            
            # W: C indices (k, i) where k in {k0, k0+1}, i in {0,1}
            # Flattened C index: k*n + i = k*2 + i
            for k in range(2):
                for i in range(2):
                    val = S_W[r, k*2 + i]
                    if val != 0:
                        W[(k0 + k)*2 + i, col] = val
            
            col += 1
            
    # Part 2: A2 (2x1) * B2 (1x8)
    # A2 uses col 2 of A (index 2 in flattened A, a=2)
    # B2 uses row 2 of B (indices 16..23 in flattened B, a=2)
    # Output uses all cols of C (indices 0..15 in flattened C)
    
    for i in range(2):
        for k in range(8):
            # U: A2 index (i, 2) -> i*3 + 2
            U[i*3 + 2, col] = 1
            # V: B2 index (2, k) -> 2*8 + k
            V[2*8 + k, col] = 1
            # W: C index (k, i) -> k*2 + i
            W[k*2 + i, col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
