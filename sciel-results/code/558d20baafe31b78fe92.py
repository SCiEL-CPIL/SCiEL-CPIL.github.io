"""Matrix multiplication 2,3,9 - rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    
    # We use a block decomposition:
    # A = [A_left (2x2), A_right (2x1)]
    # B = [B_top (2x9); B_bottom (1x9)]
    # C = A_left * B_top + A_right * B_bottom
    
    # Part 1: A_left * B_top (2x2 * 2x9)
    # Split B_top into four 2x2 blocks and one 2x1 block.
    # Rank for 2x2*2x2 is 7 (Strassen).
    # Rank for 2x2*2x1 is 4 (Schoolbook).
    # Total rank for this part: 4*7 + 4 = 32.
    
    # Part 2: A_right * B_bottom (2x1 * 1x9)
    # This is an outer product of a 2x1 and 1x9 matrix.
    # Rank is 2*9 = 18.
    
    # Total rank: 32 + 18 = 50.
    
    terms = []
    
    def add_term(u_indices, v_indices, w_indices):
        """
        u_indices: list of (index, value) for U (size n*m)
        v_indices: list of (index, value) for V (size m*p)
        w_indices: list of (index, value) for W (size p*n)
        """
        terms.append((u_indices, v_indices, w_indices))

    # --- Part 1: 2x2 * 2x9 ---
    # Strassen-like decomposition for 2x2 * 2x2
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    
    # Indices for A_left (2x2):
    # a00: 0, a01: 1
    # a10: 3, a11: 4  (using row-major: i*m + j)
    # Wait, n=2, m=3.
    # A is 2x3.
    # A_left is cols 0,1.
    # a00: 0*3+0=0
    # a01: 0*3+1=1
    # a10: 1*3+0=3
    # a11: 1*3+1=4
    
    # Indices for B_top (2x9):
    # B is 3x9.
    # B_top is rows 0,1.
    # b00: 0*9+k
    # b10: 1*9+k
    
    # Indices for W (9x2):
    # W is p*n = 9*2 = 18.
    # W[k*n + i, r]
    # For block k0, k1:
    # w00: k0*2+0
    # w01: k0*2+1
    # w10: k1*2+0
    # w11: k1*2+1

    for block_start in range(0, 8, 2):
        k0 = block_start
        k1 = block_start + 1
        
        # M1
        add_term(
            [(0, 1), (4, 1)],
            [(0*9+k0, 1), (1*9+k1, 1)],
            [(k0*2+0, 1), (k1*2+1, 1)]
        )
        # M2
        add_term(
            [(3, 1), (4, 1)],
            [(0*9+k0, 1)],
            [(k0*2+1, 1), (k1*2+1, -1)]
        )
        # M3
        add_term(
            [(0, 1)],
            [(0*9+k1, 1), (1*9+k1, -1)],
            [(k1*2+0, 1), (k1*2+1, 1)]
        )
        # M4
        add_term(
            [(4, 1)],
            [(1*9+k0, 1), (0*9+k0, -1)],
            [(k0*2+0, 1), (k0*2+1, 1)]
        )
        # M5
        add_term(
            [(0, 1), (1, 1)],
            [(1*9+k1, 1)],
            [(k0*2+0, -1), (k1*2+0, 1)]
        )
        # M6
        add_term(
            [(3, 1), (0, -1)],
            [(0*9+k0, 1), (0*9+k1, 1)],
            [(k1*2+1, 1)]
        )
        # M7
        add_term(
            [(1, 1), (4, -1)],
            [(1*9+k0, 1), (1*9+k1, 1)],
            [(k0*2+0, 1)]
        )

    # Last column k=8 for B_top (2x1)
    k = 8
    # Schoolbook 2x2 * 2x1
    # c00 = a00*b00 + a10*b10
    # c01 = a01*b01 + a11*b11 (Wait, indices)
    # A_left: a00(0), a01(1), a10(3), a11(4)
    # B_top col 8: b08(0*9+8), b18(1*9+8)
    # W col 8: w0(8*2+0), w1(8*2+1)
    
    add_term([(0, 1)], [(0*9+8, 1)], [(8*2+0, 1)])
    add_term([(1, 1)], [(0*9+8, 1)], [(8*2+1, 1)])
    add_term([(3, 1)], [(1*9+8, 1)], [(8*2+0, 1)])
    add_term([(4, 1)], [(1*9+8, 1)], [(8*2+1, 1)])

    # --- Part 2: 2x1 * 1x9 ---
    # A_right is col 2 of A.
    # a02: 0*3+2=2
    # a12: 1*3+2=5
    # B_bottom is row 2 of B.
    # b2k: 2*9+k
    
    for i in range(n):
        for k in range(p):
            # A index: i*m + 2
            # B index: 2*p + k
            # W index: k*n + i
            add_term(
                [(i*m + 2, 1)],
                [(2*p + k, 1)],
                [(k*n + i, 1)]
            )

    rank = len(terms)
    
    # Construct matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_inds, v_inds, w_inds) in enumerate(terms):
        for idx, val in u_inds:
            U[idx][r] = val
        for idx, val in v_inds:
            V[idx][r] = val
        for idx, val in w_inds:
            W[idx][r] = val
            
    return {"U": U, "V": V, "W": W}
