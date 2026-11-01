"""Rewritten program for matrix multiplication 2x2 by 2x5 tensor decomposition."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will construct the decomposition using modular blocks.
    # Block 1: Strassen for 2x2x2 (Rank 7)
    # Block 2: Strassen for 2x2x2 (Rank 7)
    # Block 3: Schoolbook for 2x2x1 (Rank 4)
    
    total_rank = 7 + 7 + 4
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Helper to add a block
    def add_block(offset, block_type, col_start):
        if block_type == 'strassen':
            # Strassen coefficients for 2x2x2
            # U (A)
            # M1: A00+A11
            U[0][offset+0] += 1
            U[3][offset+0] += 1
            # M2: A10+A11
            U[2][offset+1] += 1
            U[3][offset+1] += 1
            # M3: A00
            U[0][offset+2] += 1
            # M4: A11
            U[3][offset+3] += 1
            # M5: A00+A01
            U[0][offset+4] += 1
            U[1][offset+4] += 1
            # M6: A10-A00
            U[2][offset+5] += 1
            U[0][offset+5] += -1
            # M7: A01-A11
            U[1][offset+6] += 1
            U[3][offset+6] += -1
            
            # V (B)
            k0 = col_start
            k1 = col_start + 1
            # M1: B00+B11
            V[0*p + k0][offset+0] += 1
            V[1*p + k1][offset+0] += 1
            # M2: B00
            V[0*p + k0][offset+1] += 1
            # M3: B01-B11
            V[0*p + k1][offset+2] += 1
            V[1*p + k1][offset+2] += -1
            # M4: B10-B00
            V[1*p + k0][offset+3] += 1
            V[0*p + k0][offset+3] += -1
            # M5: B11
            V[1*p + k1][offset+4] += 1
            # M6: B00+B01
            V[0*p + k0][offset+5] += 1
            V[0*p + k1][offset+5] += 1
            # M7: B10+B11
            V[1*p + k0][offset+6] += 1
            V[1*p + k1][offset+6] += 1
            
            # W (C)
            # C00 = M1 + M4 - M5 + M7
            W[k0*n + 0][offset+0] += 1
            W[k0*n + 0][offset+3] += 1
            W[k0*n + 0][offset+4] += -1
            W[k0*n + 0][offset+6] += 1
            # C01 = M3 + M5
            W[k1*n + 0][offset+2] += 1
            W[k1*n + 0][offset+4] += 1
            # C10 = M2 + M4
            W[k0*n + 1][offset+1] += 1
            W[k0*n + 1][offset+3] += 1
            # C11 = M1 - M2 + M3 + M6
            W[k1*n + 1][offset+0] += 1
            W[k1*n + 1][offset+1] += -1
            W[k1*n + 1][offset+2] += 1
            W[k1*n + 1][offset+5] += 1
            
        elif block_type == 'schoolbook':
            # Schoolbook for 2x2x1 (single column)
            k0 = col_start
            # C0k = A00*B0k + A01*B1k
            # Term 1: A00*B0k
            U[0][offset+0] += 1
            V[0*p + k0][offset+0] += 1
            W[k0*n + 0][offset+0] += 1
            # Term 2: A01*B1k
            U[1][offset+1] += 1
            V[1*p + k0][offset+1] += 1
            W[k0*n + 0][offset+1] += 1
            # Term 3: A10*B0k
            U[2][offset+2] += 1
            V[0*p + k0][offset+2] += 1
            W[k0*n + 1][offset+2] += 1
            # Term 4: A11*B1k
            U[3][offset+3] += 1
            V[1*p + k0][offset+3] += 1
            W[k0*n + 1][offset+3] += 1

    # Add blocks
    add_block(0, 'strassen', 0)
    add_block(7, 'strassen', 2)
    add_block(14, 'schoolbook', 4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
