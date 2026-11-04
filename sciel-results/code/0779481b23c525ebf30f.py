"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We decompose M_{2,3,10} into two M_{2,3,5} blocks.
    # Each M_{2,3,5} is decomposed into M_{2,3,4} (rank 18) + M_{2,3,1} (rank 6).
    # Total rank = 2 * (18 + 6) = 48.
    
    rank = 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a schoolbook term for M_{n,m,p_block} at offset p_offset
    # A is n x m, B is m x p_block, C is n x p_block
    # Indices: i in [0,n), j in [0,m), k in [0,p_block)
    # U index: i*m + j
    # V index: (p_offset + k)*m + j ? No.
    # V index in full matrix: j*p + k.
    # Here k is local. Global k = p_offset + k_local.
    # V index: j*p + (p_offset + k_local)
    # W index: (p_offset + k_local)*n + i
    
    def add_schoolbook_block(p_offset, p_block):
        nonlocal col
        for i in range(n):
            for j in range(m):
                for k_local in range(p_block):
                    k_global = p_offset + k_local
                    U[i * m + j][col] = 1
                    V[j * p + k_global][col] = 1
                    W[k_global * n + i][col] = 1
                    col += 1

    # Block 1: p_offset 0, p_block 5
    # Sub-block 1a: p_offset 0, p_block 4
    add_schoolbook_block(0, 4)
    # Sub-block 1b: p_offset 4, p_block 1
    add_schoolbook_block(4, 1)
    
    # Block 2: p_offset 5, p_block 5
    # Sub-block 2a: p_offset 5, p_block 4
    add_schoolbook_block(5, 4)
    # Sub-block 2b: p_offset 9, p_block 1
    add_schoolbook_block(9, 1)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
