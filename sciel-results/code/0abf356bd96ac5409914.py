import time

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Total Rank 39
    # Part 1: M_{2,1,7} (Rank 14)
    # Part 2: M_{2,2,7} (Rank 25)
    #   Part 2a: M_{2,2,3} (Rank 11)
    #     Part 2a-i: M_{2,2,2} (Rank 7)
    #     Part 2a-ii: M_{2,2,1} (Rank 4)
    #   Part 2b: M_{2,2,4} (Rank 14)
    #     Part 2b-i: M_{2,2,2} (Rank 7)
    #     Part 2b-ii: M_{2,2,2} (Rank 7)
    
    total_rank = 39
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Helper to add a block
    # block_U, block_V, block_W are lists of lists
    # row_offsets_U, row_offsets_V, row_offsets_W map local row indices to global row indices
    def add_block(bU, bV, bW, u_offs, v_offs, w_offs):
        nonlocal col_offset
        rank = len(bU[0])
        for local_r in range(len(bU)):
            global_r = u_offs[local_r]
            for c in range(rank):
                U[global_r][col_offset + c] = bU[local_r][c]
        for local_r in range(len(bV)):
            global_r = v_offs[local_r]
            for c in range(rank):
                V[global_r][col_offset + c] = bV[local_r][c]
        for local_r in range(len(bW)):
            global_r = w_offs[local_r]
            for c in range(rank):
                W[global_r][col_offset + c] = bW[local_r][c]
        col_offset += rank

    # Block definitions
    
    # M_{2,1,1}
    def get_211():
        U = [[1, 0], [0, 1]]
        V = [[1, 1]]
        W = [[1, 0], [0, 1]]
        return U, V, W

    # M_{2,2,1}
    def get_221():
        U = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        V = [[1, 0, 1, 0], [0, 1, 0, 1]]
        W = [[1, 1, 0, 0], [0, 0, 1, 1]]
        return U, V, W

    # M_{2,2,2} Strassen
    def get_222():
        U = [
            [1, 0, 1, 0, 1, -1, 0], # a
            [0, 0, 0, 0, 0, -1, 1], # b
            [0, 1, 0, 0, 1, 0, 0], # c
            [1, 1, 0, 1, 0, 1, -1]  # d
        ]
        V = [
            [1, 1, 0, -1, 0, 1, 0], # e
            [0, 0, 1, 0, 0, 1, 0], # f
            [0, 0, 0, 1, 0, 0, 1], # g
            [1, 0, -1, 0, 1, 0, 1]  # h
        ]
        W = [
            [1, 0, 0, 1, -1, 0, 1], # C00
            [0, 0, 1, 0, 1, 0, 0], # C01
            [0, 1, 0, 1, 0, 0, 0], # C10
            [1, -1, 1, 0, 0, 1, 0]  # C11
        ]
        return U, V, W

    # Part 1: M_{2,1,7}
    # j=0. i in {0,1}, k in {0..6}
    # Local 211 block for each k
    for k in range(7):
        bU, bV, bW = get_211()
        u_offs = [0*3 + 0, 1*3 + 0] # i=0,j=0; i=1,j=0
        v_offs = [0*7 + k] # j=0, k=k
        w_offs = [k*2 + 0, k*2 + 1] # k=k, i=0; k=k, i=1
        add_block(bU, bV, bW, u_offs, v_offs, w_offs)

    # Part 2: M_{2,2,7}
    # j in {1,2}. i in {0,1}, k in {0..6}
    # Split k into {0,1,2} and {3,4,5,6}
    
    # Part 2a: M_{2,2,3} for k in {0,1,2}
    # Split into M_{2,2,2} (k=0,1) and M_{2,2,1} (k=2)
    
    # Part 2a-i: M_{2,2,2} for k in {0,1}
    bU, bV, bW = get_222()
    # U local rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Global U rows: i*3 + j. j=1,2.
    u_offs = [
        0*3 + 1, # i=0, j=1 (a)
        0*3 + 2, # i=0, j=2 (b)
        1*3 + 1, # i=1, j=1 (c)
        1*3 + 2  # i=1, j=2 (d)
    ]
    # V local rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Global V rows: j*7 + k. j=1,2. k=0,1.
    v_offs = [
        1*7 + 0, # j=1, k=0 (e)
        1*7 + 1, # j=1, k=1 (f)
        2*7 + 0, # j=2, k=0 (g)
        2*7 + 1  # j=2, k=1 (h)
    ]
    # W local rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Global W rows: k*2 + i. k=0,1. i=0,1.
    w_offs = [
        0*2 + 0, # k=0, i=0
        0*2 + 1, # k=0, i=1
        1*2 + 0, # k=1, i=0
        1*2 + 1  # k=1, i=1
    ]
    add_block(bU, bV, bW, u_offs, v_offs, w_offs)

    # Part 2a-ii: M_{2,2,1} for k=2
    bU, bV, bW = get_221()
    u_offs = [
        0*3 + 1, # a
        0*3 + 2, # b
        1*3 + 1, # c
        1*3 + 2  # d
    ]
    # V local rows: 0:(0,0), 1:(1,0)
    # Global V rows: j*7 + k. j=1,2. k=2.
    v_offs = [
        1*7 + 2, # e
        2*7 + 2  # g
    ]
    # W local rows: 0:(0,0), 1:(1,0)
    # Global W rows: k*2 + i. k=2. i=0,1.
    w_offs = [
        2*2 + 0, # C00
        2*2 + 1  # C10
    ]
    add_block(bU, bV, bW, u_offs, v_offs, w_offs)

    # Part 2b: M_{2,2,4} for k in {3,4,5,6}
    # Split into M_{2,2,2} (k=3,4) and M_{2,2,2} (k=5,6)
    
    # Part 2b-i: M_{2,2,2} for k in {3,4}
    bU, bV, bW = get_222()
    u_offs = [
        0*3 + 1, # a
        0*3 + 2, # b
        1*3 + 1, # c
        1*3 + 2  # d
    ]
    v_offs = [
        1*7 + 3, # e
        1*7 + 4, # f
        2*7 + 3, # g
        2*7 + 4  # h
    ]
    w_offs = [
        3*2 + 0, # k=3, i=0
        3*2 + 1, # k=3, i=1
        4*2 + 0, # k=4, i=0
        4*2 + 1  # k=4, i=1
    ]
    add_block(bU, bV, bW, u_offs, v_offs, w_offs)

    # Part 2b-ii: M_{2,2,2} for k in {5,6}
    bU, bV, bW = get_222()
    u_offs = [
        0*3 + 1, # a
        0*3 + 2, # b
        1*3 + 1, # c
        1*3 + 2  # d
    ]
    v_offs = [
        1*7 + 5, # e
        1*7 + 6, # f
        2*7 + 5, # g
        2*7 + 6  # h
    ]
    w_offs = [
        5*2 + 0, # k=5, i=0
        5*2 + 1, # k=5, i=1
        6*2 + 0, # k=6, i=0
        6*2 + 1  # k=6, i=1
    ]
    add_block(bU, bV, bW, u_offs, v_offs, w_offs)

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
