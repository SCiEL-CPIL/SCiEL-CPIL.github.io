"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    deadline = time.time() + 140  # safety margin before 150s kill
    
    rank = 65
    U = [[0] * rank for _ in range(n * m)]  # 16 x 65
    V = [[0] * rank for _ in range(m * p)]  # 20 x 65
    W = [[0] * rank for _ in range(p * n)]  # 20 x 65
    
    col = 0
    
    # Precompute the 49 Strassen terms for 4x4 * 4x4
    # Level 1: 7 block-level Strassen products
    # Level 2: 7 element-level Strassen products for each 2x2 block product
    
    # Level 1 definitions: (p_blocks, q_blocks, c_contrib)
    # p_blocks: list of (block_idx, coeff) for A
    # q_blocks: list of (block_idx, coeff) for B  
    # c_contrib: dict mapping block_idx -> coeff for C
    level1 = [
        # M0 = (A00+A11)(B00+B11), contributes to C00 and C11
        ([(0,1),(3,1)], [(0,1),(3,1)], {0:1, 3:1}),
        # M1 = (A10+A11)B00, contributes to C10 and -C11
        ([(2,1),(3,1)], [(0,1)], {2:1, 3:-1}),
        # M2 = A00(B01-B11), contributes to C01 and C11
        ([(0,1)], [(1,1),(3,-1)], {1:1, 3:1}),
        # M3 = A11(B10-B00), contributes to C00 and C10
        ([(3,1)], [(2,1),(0,-1)], {0:1, 2:1}),
        # M4 = (A00+A01)B11, contributes to -C00 and C01
        ([(0,1),(1,1)], [(3,1)], {0:-1, 1:1}),
        # M5 = (A10-A00)(B00+B01), contributes to C11
        ([(2,1),(0,-1)], [(0,1),(1,1)], {3:1}),
        # M6 = (A01-A11)(B10+B11), contributes to C00
        ([(1,1),(3,-1)], [(2,1),(3,1)], {0:1}),
    ]
    
    # Level 2 definitions for 2x2 Strassen
    # Each entry: (p_elems, q_elems, c_contrib)
    # p_elems: list of (elem_idx, coeff) where elem_idx = r*2+c
    # q_elems: list of (elem_idx, coeff)
    # c_contrib: dict mapping elem_idx -> coeff
    level2 = [
        # S0 = (P00+P11)(Q00+Q11)
        ([(0,1),(3,1)], [(0,1),(3,1)], {0:1, 3:1}),
        # S1 = (P10+P11)Q00
        ([(2,1),(3,1)], [(0,1)], {2:1, 3:-1}),
        # S2 = P00(Q01-Q11)
        ([(0,1)], [(1,1),(3,-1)], {1:1, 3:1}),
        # S3 = P11(Q10-Q00)
        ([(3,1)], [(2,1),(0,-1)], {0:1, 2:1}),
        # S4 = (P00+P01)Q11
        ([(0,1),(1,1)], [(3,1)], {0:-1, 1:1}),
        # S5 = (P10-P00)(Q00+Q01)
        ([(2,1),(0,-1)], [(0,1),(1,1)], {3:1}),
        # S6 = (P01-P11)(Q10+Q11)
        ([(1,1),(3,-1)], [(2,1),(3,1)], {0:1}),
    ]
    
    # Helper to map block_idx and local (r,c) to global (i,j) for A
    def a_global(block_idx, r, c):
        if block_idx == 0: return (r, c)
        elif block_idx == 1: return (r, 2+c)
        elif block_idx == 2: return (2+r, c)
        else: return (2+r, 2+c)
    
    # Helper to map block_idx and local (r,c) to global (j,k) for B
    def b_global(block_idx, r, c):
        if block_idx == 0: return (r, c)
        elif block_idx == 1: return (r, 2+c)
        elif block_idx == 2: return (2+r, c)
        else: return (2+r, 2+c)
    
    # Helper to map block_idx and local (r,c) to global (i,j) for C
    def c_global(block_idx, r, c):
        if block_idx == 0: return (r, c)
        elif block_idx == 1: return (r, 2+c)
        elif block_idx == 2: return (2+r, c)
        else: return (2+r, 2+c)
    
    # Generate all 49 terms
    for l1_idx in range(7):
        p_blocks, q_blocks, c_contrib = level1[l1_idx]
        for l2_idx in range(7):
            p_elems, q_elems, c_contrib_l2 = level2[l2_idx]
            
            if time.time() > deadline:
                break
            
            # Build U contributions
            for p_bidx, p_bcoeff in p_blocks:
                for p_eidx, p_ecoeff in p_elems:
                    r = p_eidx // 2
                    c = p_eidx % 2
                    gi, gj = a_global(p_bidx, r, c)
                    coeff = p_bcoeff * p_ecoeff
                    if coeff != 0:
                        U[gi * m + gj][col] += coeff
            
            # Build V contributions
            for q_bidx, q_bcoeff in q_blocks:
                for q_eidx, q_ecoeff in q_elems:
                    r = q_eidx // 2
                    c = q_eidx % 2
                    gj, gk = b_global(q_bidx, r, c)
                    coeff = q_bcoeff * q_ecoeff
                    if coeff != 0:
                        V[gj * p + gk][col] += coeff
            
            # Build W contributions
            for l1_cblock, l1_ccoeff in c_contrib.items():
                for l2_cidx, l2_ccoeff in c_contrib_l2.items():
                    r = l2_cidx // 2
                    c = l2_cidx % 2
                    gi, gj = c_global(l1_cblock, r, c)
                    coeff = l1_ccoeff * l2_ccoeff
                    if coeff != 0:
                        W[gj * n + gi][col] += coeff
            
            col += 1
    
    # Part 2: 4x4 * 4x1 using schoolbook (rank 16)
    for i in range(n):
        for j in range(m):
            if time.time() > deadline:
                break
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
    
    assert col == rank, f"Expected {rank} terms, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
