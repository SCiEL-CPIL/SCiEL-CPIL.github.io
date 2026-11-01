import time

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 120s limit
    
    # Rank 7 Strassen decomposition
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    # U indices: i*2 + j
    # V indices: j*2 + k
    # W indices: k*2 + i
    
    # Product 1: (a00 + a11)(b00 + b11)
    # U: a00 (0), a11 (3)
    # V: b00 (0), b11 (3)
    # W: c00 (0), c11 (3)
    U[0][0] = 1
    U[3][0] = 1
    V[0][0] = 1
    V[3][0] = 1
    W[0][0] = 1
    W[3][0] = 1
    
    # Product 2: (a10 + a11)b00
    # U: a10 (2), a11 (3)
    # V: b00 (0)
    # W: c10 (1), c11 (3) with sign -1 for c11
    U[2][1] = 1
    U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1
    W[3][1] = -1
    
    # Product 3: a00(b01 - b11)
    # U: a00 (0)
    # V: b01 (1), b11 (3) with sign -1
    # W: c01 (2), c11 (3)
    U[0][2] = 1
    V[1][2] = 1
    V[3][2] = -1
    W[2][2] = 1
    W[3][2] = 1
    
    # Product 4: a11(b10 - b00)
    # U: a11 (3)
    # V: b10 (2), b00 (0) with sign -1
    # W: c00 (0), c10 (1)
    U[3][3] = 1
    V[2][3] = 1
    V[0][3] = -1
    W[0][3] = 1
    W[1][3] = 1
    
    # Product 5: (a00 + a01)b11
    # U: a00 (0), a01 (1)
    # V: b11 (3)
    # W: c00 (0) with sign -1, c01 (2)
    U[0][4] = 1
    U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1
    W[2][4] = 1
    
    # Product 6: (a10 - a00)(b00 + b01)
    # U: a10 (2), a00 (0) with sign -1
    # V: b00 (0), b01 (1)
    # W: c11 (3)
    U[2][5] = 1
    U[0][5] = -1
    V[0][5] = 1
    V[1][5] = 1
    W[3][5] = 1
    
    # Product 7: (a01 - a11)(b10 + b11)
    # U: a01 (1), a11 (3) with sign -1
    # V: b10 (2), b11 (3)
    # W: c00 (0)
    U[1][6] = 1
    U[3][6] = -1
    V[2][6] = 1
    V[3][6] = 1
    W[0][6] = 1
    
    # Verify decomposition validity if time permits (optional, usually fast)
    # For static Strassen, this is always valid.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
