"""Program for matrix multiplication tensor decomposition: 2x2 by 2x16."""
import time

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 2, 2, 16
    num_blocks = p // 2  # 8 blocks of width 2
    rank = 7 * num_blocks  # 56
    
    # Initialize matrices with zeros
    # U: (n*m, rank) = (4, 56)
    # V: (m*p, rank) = (32, 56)
    # W: (p*n, rank) = (32, 56)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 7 Strassen factors for 2x2 matrix multiplication.
    # Each factor uses (i, j, coeff) for U, (j, k, coeff) for V, (k, i, coeff) for W
    # where i in {0,1}, j in {0,1}, k in {0,1} are local indices within a 2x2 block.
    
    strassen_factors = [
        # s=0: M1 = (a00+a11)(b00+b11)
        {
            'u': [(0, 0, 1), (1, 1, 1)],  # a00, a11
            'v': [(0, 0, 1), (1, 1, 1)],  # b00, b11
            'w': [(0, 0, 1), (1, 1, 1)]   # c00, c11
        },
        # s=1: M2 = (a10+a11)b00
        {
            'u': [(1, 0, 1), (1, 1, 1)],  # a10, a11
            'v': [(0, 0, 1)],             # b00
            'w': [(0, 1, 1), (1, 1, -1)]  # c10, c11
        },
        # s=2: M3 = a00(b01-b11)
        {
            'u': [(0, 0, 1)],             # a00
            'v': [(0, 1, 1), (1, 1, -1)], # b01, b11
            'w': [(1, 0, 1), (1, 1, 1)]   # c01, c11
        },
        # s=3: M4 = a11(b10-b00)
        {
            'u': [(1, 1, 1)],             # a11
            'v': [(1, 0, 1), (0, 0, -1)], # b10, b00
            'w': [(0, 0, 1), (0, 1, 1)]   # c00, c10
        },
        # s=4: M5 = (a00+a01)b11
        {
            'u': [(0, 0, 1), (0, 1, 1)],  # a00, a01
            'v': [(1, 1, 1)],             # b11
            'w': [(0, 0, -1), (1, 0, 1)]  # c00, c01
        },
        # s=5: M6 = (a10-a00)(b00+b01)
        {
            'u': [(1, 0, 1), (0, 0, -1)], # a10, a00
            'v': [(0, 0, 1), (0, 1, 1)],  # b00, b01
            'w': [(1, 1, 1)]              # c11
        },
        # s=6: M7 = (a01-a11)(b10+b11)
        {
            'u': [(0, 1, 1), (1, 1, -1)], # a01, a11
            'v': [(1, 0, 1), (1, 1, 1)],  # b10, b11
            'w': [(0, 0, 1)]              # c00
        }
    ]
    
    # Fill the matrices
    for t in range(num_blocks):
        if time.time() > deadline:
            break
        block_start = 2 * t
        for s in range(7):
            col = t * 7 + s
            factor = strassen_factors[s]
            
            # Fill U: row = i*m + j
            for i, j, coeff in factor['u']:
                U[i * m + j][col] = coeff
                
            # Fill V: row = j*p + (block_start + k_local)
            for j, k_local, coeff in factor['v']:
                V[j * p + block_start + k_local][col] = coeff
                
            # Fill W: row = (block_start + k_local)*n + i
            for k_local, i, coeff in factor['w']:
                W[(block_start + k_local) * n + i][col] = coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
