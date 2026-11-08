import numpy as np
import time

def construct():
    # Deadline guard: The evaluator kills after 150s.
    # We should return the best result found before 120s to be safe.
    t0 = time.time()
    limit = t0 + 115.0  # 5s margin

    n, m, p = 4, 4, 8

    # Rank-98 decomposition: two independent Strassen-based 4x4x4 blocks.
    # A is 4x4, B is 4x8 (split into two 4x4 halves), C is 8x4 (two 4x4 halves).
    # Each 4x4x4 uses a rank-49 decomposition built from 7 "super-Strassen"
    # products, each of which is a 2x2x2 Strassen product (rank 7).
    # Total rank = 2 * 49 = 98.

    rank = 98
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    # Strassen 2x2x2 factor matrices (4 entries each side, 7 products)
    # Index convention: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))

    # P0 = (A00+A11)(B00+B11)
    U2[0,0]=1; U2[3,0]=1; V2[0,0]=1; V2[3,0]=1; W2[0,0]=1; W2[3,0]=1
    # P1 = (A10+A11)B00
    U2[2,1]=1; U2[3,1]=1; V2[0,1]=1; W2[2,1]=1; W2[3,1]=-1
    # P2 = A00(B01-B11)
    U2[0,2]=1; V2[1,2]=1; V2[3,2]=-1; W2[1,2]=1; W2[3,2]=1
    # P3 = A11(B10-B00)
    U2[3,3]=1; V2[2,3]=1; V2[0,3]=-1; W2[0,3]=1; W2[2,3]=1
    # P4 = (A00+A01)B11
    U2[0,4]=1; U2[1,4]=1; V2[3,4]=1; W2[0,4]=-1; W2[1,4]=1
    # P5 = (A10-A00)(B00+B01)
    U2[2,5]=1; U2[0,5]=-1; V2[0,5]=1; V2[1,5]=1; W2[3,5]=1
    # P6 = (A01-A11)(B10+B11)
    U2[1,6]=1; U2[3,6]=-1; V2[2,6]=1; V2[3,6]=1; W2[0,6]=1

    # For each 4x4x4 block, define 7 "super-products" P_k
    # A_eff = sum of specified 2x2 blocks of A
    # B_eff = sum of specified 2x2 blocks of B
    # C contribution: P_k is added/subtracted from specified C blocks
    # Block indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)

    A_spec = [
        [(1,0),(1,3)],  # P0: A00+A11
        [(1,2),(1,3)],  # P1: A10+A11
        [(1,0)],        # P2: A00
        [(1,3)],        # P3: A11
        [(1,0),(1,1)],  # P4: A00+A01
        [(1,2),(-1,0)], # P5: A10-A00
        [(1,1),(-1,3)], # P6: A01-A11
    ]
    B_spec = [
        [(1,0),(1,3)],  # P0: B00+B11
        [(1,0)],        # P1: B00
        [(1,1),(-1,3)], # P2: B01-B11
        [(1,2),(-1,0)], # P3: B10-B00
        [(1,3)],        # P4: B11
        [(1,0),(1,1)],  # P5: B00+B01
        [(1,2),(1,3)],  # P6: B10+B11
    ]
    C_spec = [
        [(1,0),(1,3)],   # P0: C00+C11
        [(1,2),(-1,3)],  # P1: C10-C11
        [(1,1),(1,3)],   # P2: C01+C11
        [(1,0),(1,2)],   # P3: C00+C10
        [(-1,0),(1,1)],  # P4: -C00+C01
        [(1,3)],         # P5: C11
        [(1,0)],         # P6: C00
    ]

    # Precompute the 2x2 block-to-element mapping
    # For a 2x2 block with top-left at (br, bc) in a 4x4 matrix:
    #   element (r,c) in block -> global (br*2+r, bc*2+c)
    block_offsets = []
    for bi in range(4):
        br, bc = divmod(bi, 2)
        block_offsets.append((br, bc))

    for half in range(2):
        base_col = half * 49
        for kp in range(7):
            for ms in range(7):
                col = base_col + kp * 7 + ms
                if time.time() > limit:
                    break

                # --- U: Input A (4x4) ---
                for ca, bidx in A_spec[kp]:
                    br, bc = block_offsets[bidx]
                    for r in range(2):
                        for c in range(2):
                            U[br*2+r, bc*2+c, ]  # placeholder
                            # A index: row*4 + col  (n*m = 4*4 = 16, but we use 4x4)
                            U[(br*2+r)*4 + (bc*2+c), col] += ca * U2[r*2+c, ms]

                # --- V: Input B (4x8) ---
                # B half index: half*4 + col within half
                for cb, bidx in B_spec[kp]:
                    br, bc = block_offsets[bidx]
                    for r in range(2):
                        for c in range(2):
                            # B global col = bc*2+c + half*4
                            V[(br*2+r)*8 + (bc*2+c + half*4), col] += cb * V2[r*2+c, ms]

                # --- W: Output C (8x4) ---
                # C index convention: W[k*n+i, r] where k is col of C (0..7), i is row of C (0..3)
                # So C element at row i, col k -> index k*4 + i
                for cc, cidx in C_spec[kp]:
                    bi, bj = divmod(cidx, 2)
                    for r in range(2):
                        for c in range(2):
                            # C row = bi*2+r, C col = bj*2+c + half*4
                            C_row = bi*2 + r
                            C_col = bj*2 + c + half*4
                            W[C_col*4 + C_row, col] += cc * W2[r*2+c, ms]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
