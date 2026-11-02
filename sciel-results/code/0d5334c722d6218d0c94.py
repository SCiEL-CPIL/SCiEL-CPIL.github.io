"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    # We will decompose p into 7 blocks of size 2, and use Strassen (rank 7) for each.
    # Total rank = 7 * 7 = 49.
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 decomposition
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = A * B
    #
    # M1 = (a00 + a11) * (b00 + b11)
    # M2 = (a10 + a11) * b00
    # M3 = a00 * (b01 - b11)
    # M4 = a11 * (b10 - b00)
    # M5 = (a00 + a01) * b11
    # M6 = (a10 - a00) * (b00 + b01)
    # M7 = (a01 - a11) * (b10 + b11)
    #
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Indices for U: i * m + j = 2i + j
    # Indices for V: j * p + k
    # Indices for W: k * n + i = 14k + i
    
    # For each block b (0..6), k_local in {0, 1}
    # k_global = 2*b + k_local
    
    for b in range(7):
        k_base = 2 * b
        
        # M1: (a00 + a11) * (b00 + b11)
        # U: a00=1, a11=1
        # V: b00=1 (j=0, k=k_base), b11=1 (j=1, k=k_base+1)
        # W: C00 += 1 (k=k_base, i=0), C11 += 1 (k=k_base+1, i=1)
        
        col = b * 7 + 0
        U[0][col] += 1  # a00
        U[3][col] += 1  # a11
        V[0 * p + k_base][col] += 1      # b00
        V[1 * p + (k_base + 1)][col] += 1 # b11
        W[k_base * n + 0][col] += 1      # C00
        W[(k_base + 1) * n + 1][col] += 1 # C11
        
        # M2: (a10 + a11) * b00
        # U: a10=1, a11=1
        # V: b00=1
        # W: C10 += 1 (k=k_base, i=1)
        
        col = b * 7 + 1
        U[2][col] += 1  # a10
        U[3][col] += 1  # a11
        V[0 * p + k_base][col] += 1      # b00
        W[k_base * n + 1][col] += 1      # C10
        
        # M3: a00 * (b01 - b11)
        # U: a00=1
        # V: b01=1 (j=0, k=k_base+1), b11=-1 (j=1, k=k_base+1)
        # W: C01 += 1 (k=k_base+1, i=0)
        
        col = b * 7 + 2
        U[0][col] += 1  # a00
        V[0 * p + (k_base + 1)][col] += 1  # b01
        V[1 * p + (k_base + 1)][col] += -1 # b11
        W[(k_base + 1) * n + 0][col] += 1  # C01
        
        # M4: a11 * (b10 - b00)
        # U: a11=1
        # V: b10=1 (j=1, k=k_base), b00=-1 (j=0, k=k_base)
        # W: C00 -= 1 (k=k_base, i=0), C10 += 1 (k=k_base, i=1)
        
        col = b * 7 + 3
        U[3][col] += 1  # a11
        V[1 * p + k_base][col] += 1      # b10
        V[0 * p + k_base][col] += -1     # b00
        W[k_base * n + 0][col] += -1     # C00
        W[k_base * n + 1][col] += 1      # C10
        
        # M5: (a00 + a01) * b11
        # U: a00=1, a01=1
        # V: b11=1 (j=1, k=k_base+1)
        # W: C00 += 1 (k=k_base, i=0), C01 += 1 (k=k_base+1, i=0)
        
        col = b * 7 + 4
        U[0][col] += 1  # a00
        U[1][col] += 1  # a01
        V[1 * p + (k_base + 1)][col] += 1 # b11
        W[k_base * n + 0][col] += 1      # C00
        W[(k_base + 1) * n + 0][col] += 1 # C01
        
        # M6: (a10 - a00) * (b00 + b01)
        # U: a10=1, a00=-1
        # V: b00=1 (j=0, k=k_base), b01=1 (j=0, k=k_base+1)
        # W: C11 += 1 (k=k_base+1, i=1)
        
        col = b * 7 + 5
        U[2][col] += 1   # a10
        U[0][col] += -1  # a00
        V[0 * p + k_base][col] += 1      # b00
        V[0 * p + (k_base + 1)][col] += 1 # b01
        W[(k_base + 1) * n + 1][col] += 1 # C11
        
        # M7: (a01 - a11) * (b10 + b11)
        # U: a01=1, a11=-1
        # V: b10=1 (j=1, k=k_base), b11=1 (j=1, k=k_base+1)
        # W: C00 -= 1 (k=k_base, i=0)
        
        col = b * 7 + 6
        U[1][col] += 1   # a01
        U[3][col] += -1  # a11
        V[1 * p + k_base][col] += 1      # b10
        V[1 * p + (k_base + 1)][col] += 1 # b11
        W[k_base * n + 0][col] += -1     # C00

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
