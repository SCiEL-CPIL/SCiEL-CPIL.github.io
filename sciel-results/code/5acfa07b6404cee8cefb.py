import time

DIMENSIONS = [2, 2, 7]

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 110.0  # 10 second margin
    
    n, m, p = DIMENSIONS
    
    # Rank 25 decomposition:
    # 3 blocks of 2x2x2 (Strassen, rank 7)
    # 1 block of 2x2x1 (Schoolbook, rank 4)
    
    rank = 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a block
    def add_block(offset, block_U, block_V, block_W, u_start, v_start, w_start):
        for r in range(len(block_U[0])):
            col = offset + r
            for i in range(len(block_U)):
                U[u_start + i][col] += block_U[i][r]
            for i in range(len(block_V)):
                V[v_start + i][col] += block_V[i][r]
            for i in range(len(block_W)):
                W[w_start + i][col] += block_W[i][r]
        
    # Strassen 2x2x2 factors
    # U: 4x7, V: 4x7, W: 4x7
    # Indices:
    # A: 00, 01, 10, 11
    # B: 00, 01, 10, 11
    # C: 00, 01, 10, 11 (k*n+i: 0*2+0=0, 0*2+1=1, 1*2+0=2, 1*2+1=3)
    
    S_U = [
        [1, 0, 1, 0, 1, -1, 1], # a00
        [0, 0, 0, 0, 1, 0, -1], # a01
        [0, 1, 0, 0, 0, 1, 0],  # a10
        [1, 1, 0, 1, 0, 0, -1]  # a11
    ]
    S_V = [
        [1, 1, 0, 0, 0, 1, 1],  # b00
        [0, 0, 1, 0, 0, 1, 0],  # b01
        [1, 0, -1, 1, 0, 0, 1], # b10
        [1, 0, 0, -1, 1, 0, 1]  # b11
    ]
    S_W = [
        [1, 0, 0, 1, -1, 0, 1], # c00
        [0, 0, 1, 0, 1, 0, 0],  # c01
        [0, 1, 0, 1, 0, 0, 0],  # c10
        [1, -1, 1, 0, 0, 1, 0]  # c11
    ]
    
    # Schoolbook 2x2x1 factors
    # U: 4x4, V: 2x4, W: 2x4
    # A: 00, 01, 10, 11
    # B: 00, 10 (col 0, rows 0,1)
    # C: 00, 10 (col 0, rows 0,1) -> k*n+i: 0*2+0=0, 1*2+0=2
    # Wait, W shape is p*n. Here p=1, n=2. So W is 2x4.
    # Indices: k=0, i=0 -> 0. k=0, i=1 -> 1.
    # C is 2x1. c00, c10.
    # k is col of B (0). i is row of A (0,1).
    # So W[0] is c00, W[1] is c10.
    
    K_U = [
        [1, 0, 0, 0], # a00
        [0, 1, 0, 0], # a01
        [0, 0, 1, 0], # a10
        [0, 0, 0, 1]  # a11
    ]
    K_V = [
        [1, 0, 1, 0], # b00
        [0, 1, 0, 1]  # b10
    ]
    K_W = [
        [1, 1, 0, 0], # c00
        [0, 0, 1, 1]  # c10
    ]
    
    # Block 1: Cols 0-1 of B (2x2)
    # U start 0, V start 0, W start 0
    add_block(0, S_U, S_V, S_W, 0, 0, 0)
    
    # Block 2: Cols 2-3 of B (2x2)
    # U start 0, V start 2, W start 2
    # V indices: 0,1 for col 0-1. 2,3 for col 2-3.
    # W indices: 0,1,2,3 for cols 0-1. 4,5,6,7 for cols 2-3?
    # W index k*n+i.
    # Col 0: k=0. i=0,1 -> 0,1.
    # Col 1: k=1. i=0,1 -> 2,3.
    # Col 2: k=2. i=0,1 -> 4,5.
    # Col 3: k=3. i=0,1 -> 6,7.
    # So W start for col 2 is 4.
    add_block(7, S_U, S_V, S_W, 0, 2, 4)
    
    # Block 3: Cols 4-5 of B (2x2)
    # V start 4. W start 8.
    add_block(14, S_U, S_V, S_W, 0, 4, 8)
    
    # Block 4: Col 6 of B (2x1)
    # V start 6. W start 12.
    # V index for col 6: k=6. i=0,1 -> 6*2+0=12? No.
    # V is m*p = 2*7 = 14 rows.
    # V index j*p+k? No, V[j, r] where j is row of B?
    # Task: V shape (m*p, rank).
    # Index j*p+k?
    # B is m x p.
    # V index corresponds to B element.
    # Usually V[j, r] corresponds to B[j, k]?
    # Let's check seed: V[j*p+k, c] = 1.
    # So V index is j*p+k.
    # For col 6 (k=6), j=0,1.
    # Indices: 0*7+6=6, 1*7+6=13.
    # So V start is 6.
    
    # W index k*n+i.
    # Col 6 (k=6). i=0,1.
    # Indices: 6*2+0=12, 6*2+1=13.
    # So W start is 12.
    
    add_block(21, K_U, K_V, K_W, 0, 6, 12)
    
    return {"U": U, "V": V, "W": W}
