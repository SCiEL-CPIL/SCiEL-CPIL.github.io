import numpy as np
import time

def construct():
    start_time = time.time()
    n, m, p = 4, 4, 8
    rank = 98

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # 2x2 Strassen decomposition
    # A (2x2), B (2x2), C (2x2)
    # Indices for 2x2:
    # A: 00, 01, 10, 11 -> 0, 1, 2, 3 (row major i*2+j)
    # B: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C: 00, 01, 10, 11 -> 0, 1, 2, 3 (row major i*2+j)
    # Note: The 2x2 Strassen formulas are standard.
    
    S2_U = np.zeros((4, 7))
    S2_V = np.zeros((4, 7))
    S2_W = np.zeros((4, 7))

    # M0 = (A00+A11)(B00+B11)
    S2_U[0, 0] = 1; S2_U[3, 0] = 1
    S2_V[0, 0] = 1; S2_V[3, 0] = 1
    S2_W[0, 0] = 1; S2_W[3, 0] = 1

    # M1 = (A10+A11)B00
    S2_U[2, 1] = 1; S2_U[3, 1] = 1
    S2_V[0, 1] = 1
    S2_W[2, 1] = 1; S2_W[3, 1] = -1

    # M2 = A00(B10-B11)
    S2_U[0, 2] = 1
    S2_V[2, 2] = 1; S2_V[3, 2] = -1
    S2_W[1, 2] = 1; S2_W[3, 2] = 1

    # M3 = A11(B01-B10)
    S2_U[3, 3] = 1
    S2_V[1, 3] = 1; S2_V[0, 3] = -1
    S2_W[0, 3] = 1; S2_W[2, 3] = 1

    # M4 = (A00+A01)B11
    S2_U[0, 4] = 1; S2_U[1, 4] = 1
    S2_V[3, 4] = 1
    S2_W[0, 4] = -1; S2_W[1, 4] = 1

    # M5 = (A10-A00)(B00+B01)
    S2_U[2, 5] = 1; S2_U[0, 5] = -1
    S2_V[0, 5] = 1; S2_V[1, 5] = 1
    S2_W[3, 5] = 1

    # M6 = (A01-A11)(B10+B11)
    S2_U[1, 6] = 1; S2_U[3, 6] = -1
    S2_V[2, 6] = 1; S2_V[3, 6] = 1
    S2_W[0, 6] = 1

    # Define index mappings for 4x4 blocks to global flattened indices
    # A and B (4x4): row-major i*4+j
    # Block 0 (00): i in 0..1, j in 0..1 -> 0, 1, 4, 5
    # Block 1 (01): i in 0..1, j in 2..3 -> 2, 3, 6, 7
    # Block 2 (10): i in 2..3, j in 0..1 -> 8, 9, 12, 13
    # Block 3 (11): i in 2..3, j in 2..3 -> 10, 11, 14, 15
    
    # Local 2x2 indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Map local (r,c) to global index for A/B
    def map_AB_block(block_idx, r, c):
        if block_idx == 0: return r * 4 + c
        if block_idx == 1: return r * 4 + (c + 2)
        if block_idx == 2: return (r + 2) * 4 + c
        if block_idx == 3: return (r + 2) * 4 + (c + 2)

    # C (4x4): Contract W[k*n+i, r] -> index k*4 + i
    # Block 0 (00): i in 0..1, k in 0..1
    #   r=i, c=k. Index c*4 + r.
    #   (0,0)->0, (0,1)->4, (1,0)->1, (1,1)->5
    # Block 1 (01): i in 0..1, k in 2..3
    #   (0,2)->8, (0,3)->12, (1,2)->9, (1,3)->13
    # Block 2 (10): i in 2..3, k in 0..1
    #   (2,0)->2, (2,1)->6, (3,0)->3, (3,1)->7
    # Block 3 (11): i in 2..3, k in 2..3
    #   (2,2)->10, (2,3)->14, (3,2)->11, (3,3)->15
    
    def map_C_block(block_idx, r, c):
        # r is row i, c is col k
        if block_idx == 0: return c * 4 + r
        if block_idx == 1: return (c + 2) * 4 + r
        if block_idx == 2: return c * 4 + (r + 2)
        if block_idx == 3: return (c + 2) * 4 + (r + 2)

    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))

    # Block-level Strassen specs
    # U_blocks, V_blocks, W_blocks: list of (block_idx, coeff)
    block_specs = [
        # M0 = (A00+A11)(B00+B11)
        {
            'U_blocks': [(0, 1), (3, 1)],
            'V_blocks': [(0, 1), (3, 1)],
            'W_blocks': [(0, 1), (3, 1)]
        },
        # M1 = (A10+A11)B00
        {
            'U_blocks': [(2, 1), (3, 1)],
            'V_blocks': [(0, 1)],
            'W_blocks': [(2, 1), (3, -1)]
        },
        # M2 = A00(B10-B11)
        {
            'U_blocks': [(0, 1)],
            'V_blocks': [(2, 1), (3, -1)],
            'W_blocks': [(1, 1), (3, 1)]
        },
        # M3 = A11(B01-B10)
        {
            'U_blocks': [(3, 1)],
            'V_blocks': [(1, 1), (2, -1)],
            'W_blocks': [(0, 1), (2, 1)]
        },
        # M4 = (A00+A01)B11
        {
            'U_blocks': [(0, 1), (1, 1)],
            'V_blocks': [(3, 1)],
            'W_blocks': [(0, -1), (1, 1)]
        },
        # M5 = (A10-A00)(B00+B01)
        {
            'U_blocks': [(2, 1), (0, -1)],
            'V_blocks': [(0, 1), (1, 1)],
            'W_blocks': [(3, 1)]
        },
        # M6 = (A01-A11)(B10+B11)
        {
            'U_blocks': [(1, 1), (3, -1)],
            'V_blocks': [(2, 1), (3, 1)],
            'W_blocks': [(0, 1)]
        },
    ]

    for k in range(7):
        spec = block_specs[k]
        for s in range(7):
            col = k * 7 + s

            # U contributions
            for block_idx, coeff in spec['U_blocks']:
                for r in range(2):
                    for c in range(2):
                        global_idx = map_AB_block(block_idx, r, c)
                        local_idx = r * 2 + c
                        U4[global_idx, col] += coeff * S2_U[local_idx, s]

            # V contributions
            for block_idx, coeff in spec['V_blocks']:
                for r in range(2):
                    for c in range(2):
                        global_idx = map_AB_block(block_idx, r, c)
                        local_idx = r * 2 + c
                        V4[global_idx, col] += coeff * S2_V[local_idx, s]

            # W contributions
            for block_idx, coeff in spec['W_blocks']:
                for r in range(2):
                    for c in range(2):
                        global_idx = map_C_block(block_idx, r, c)
                        local_idx = r * 2 + c
                        W4[global_idx, col] += coeff * S2_W[local_idx, s]

    # Assemble 4x4x8
    # Part 1: B cols 0-3
    U[:, 0:49] = U4
    V[0:16, 0:49] = V4
    W[0:16, 0:49] = W4

    # Part 2: B cols 4-7
    # U is same
    U[:, 49:98] = U4
    # V rows 16-31 correspond to B cols 4-7
    # B index for col 4..7 is j*8 + k where k in 4..7?
    # No, B is 4x8. Flattened j*8+k.
    # Part 1: k in 0..3. Indices 0..15.
    # Part 2: k in 4..7. Indices 16..31.
    # The structure of V4 is for a 4x4 B.
    # So V[16:32, 49:98] should be V4.
    V[16:32, 49:98] = V4
    
    # W for Part 2:
    # C indices for k in 4..7.
    # W index is k*4 + i.
    # k=4..7 -> indices 16..31.
    # The mapping in map_C_block assumed k in 0..3.
    # We need to shift the W indices by 16?
    # Let's check:
    # map_C_block(0,0,0) = 0. For part 2, k=4, i=0 -> 4*4+0 = 16.
    # So yes, W indices for part 2 are W4 indices + 16.
    W[16:32, 49:98] = W4

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
