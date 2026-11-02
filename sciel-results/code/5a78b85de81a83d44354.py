import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=12
    # Using Block Strassen decomposition
    # 6 blocks of 2x2 multiplication, each rank 7
    rank = 42
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen coefficients for 2x2 matrix multiplication
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    
    # Products:
    # P0: (a00+a11)(b00+b11) -> c00+=P0, c11+=P0
    # P1: (a10+a11)b00       -> c10+=P1, c11-=P1
    # P2: a00(b01-b11)       -> c01+=P2, c11+=P2
    # P3: a11(b10-b00)       -> c00+=P3, c10+=P3
    # P4: (a00+a01)b11       -> c00-=P4, c01+=P4
    # P5: (a10-a00)(b00+b01) -> c11+=P5
    # P6: (a01-a11)(b10+b11) -> c00+=P6
    
    # Loop over 6 blocks of B (columns 0-1, 2-3, ..., 10-11)
    for l in range(6):
        base = l * 7
        
        # B indices for this block
        # B is 2x12. Block l uses cols 2l, 2l+1
        # Row 0: 2l, 2l+1
        # Row 1: 12+2l, 12+2l+1
        b00_idx = 2*l
        b01_idx = 2*l + 1
        b10_idx = 12 + 2*l
        b11_idx = 12 + 2*l + 1
        
        # C indices for this block
        # C is 12x2 (p*n). Block l uses rows 2l, 2l+1
        # Row 2l: c00 (col 0), c10 (col 1) -> indices 4l, 4l+1
        # Row 2l+1: c01 (col 0), c11 (col 1) -> indices 4l+2, 4l+3
        c00_idx = 4*l
        c10_idx = 4*l + 1
        c01_idx = 4*l + 2
        c11_idx = 4*l + 3
        
        # P0
        col = base + 0
        U[0, col] = 1; U[3, col] = 1
        V[b00_idx, col] = 1; V[b11_idx, col] = 1
        W[c00_idx, col] = 1; W[c11_idx, col] = 1
        
        # P1
        col = base + 1
        U[2, col] = 1; U[3, col] = 1
        V[b00_idx, col] = 1
        W[c10_idx, col] = 1; W[c11_idx, col] = -1
        
        # P2
        col = base + 2
        U[0, col] = 1
        V[b01_idx, col] = 1; V[b11_idx, col] = -1
        W[c01_idx, col] = 1; W[c11_idx, col] = 1
        
        # P3
        col = base + 3
        U[3, col] = 1
        V[b10_idx, col] = 1; V[b00_idx, col] = -1
        W[c00_idx, col] = 1; W[c10_idx, col] = 1
        
        # P4
        col = base + 4
        U[0, col] = 1; U[1, col] = 1
        V[b11_idx, col] = 1
        W[c00_idx, col] = -1; W[c01_idx, col] = 1
        
        # P5
        col = base + 5
        U[2, col] = 1; U[0, col] = -1
        V[b00_idx, col] = 1; V[b01_idx, col] = 1
        W[c11_idx, col] = 1
        
        # P6
        col = base + 6
        U[1, col] = 1; U[3, col] = -1
        V[b10_idx, col] = 1; V[b11_idx, col] = 1
        W[c00_idx, col] = 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
