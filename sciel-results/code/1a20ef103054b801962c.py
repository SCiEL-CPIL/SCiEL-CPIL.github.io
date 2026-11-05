"""Matrix multiplication 3x4x8 - rank 88 via Strassen+Schoolbook decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix by a 4x8 matrix.
    Uses a decomposition based on splitting into 3x2x2 blocks.
    Rank 88 (improvement over schoolbook 96).
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    Score is the rank (lower is better).
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=3, m=4, p=8
    
    U_rows = n * m  # 12
    V_rows = m * p  # 32
    W_rows = p * n  # 24
    
    # Total rank: 2 (A splits) * 4 (B splits per A split) * 11 (rank of 3x2x2) = 88
    rank_total = 88
    
    U = [[0] * rank_total for _ in range(U_rows)]
    V = [[0] * rank_total for _ in range(V_rows)]
    W = [[0] * rank_total for _ in range(W_rows)]
    
    col = 0
    
    def add_term(a_entries, b_entries, c_entries):
        """
        a_entries: list of (row_in_A, col_in_A, coeff)
        b_entries: list of (row_in_B, col_in_B, coeff)
        c_entries: list of (row_in_C, col_in_C, coeff)
        """
        nonlocal col
        for (r, c, val) in a_entries:
            if val != 0:
                U[r * m + c][col] += val
        for (r, c, val) in b_entries:
            if val != 0:
                V[r * p + c][col] += val
        for (r, c, val) in c_entries:
            if val != 0:
                # C is n x p, W index is k*n + i where k is col in C, i is row in C
                W[c * n + r][col] += val
        col += 1
    
    # Define the 11-term decomposition for 3x2x2
    # A (3x2) * B (2x2) = C (3x2)
    # A = [[a00, a01], [a10, a11], [a20, a21]]
    # B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11], [c20, c21]]
    
    # Strassen terms for top 2x2 (rows 0,1 of A)
    # m1 = (a00+a11)(b00+b11)
    add_term(
        [(0,0,1), (1,1,1)],
        [(0,0,1), (1,1,1)],
        [(0,0,1), (1,1,1), (0,1,-1), (1,0,-1)]
    )
    
    # m2 = (a10+a11)b00
    add_term(
        [(1,0,1), (1,1,1)],
        [(0,0,1)],
        [(1,0,-1), (1,1,1)]
    )
    
    # m3 = a00(b10-b11)
    add_term(
        [(0,0,1)],
        [(1,0,1), (1,1,-1)],
        [(0,0,1), (1,0,-1)]
    )
    
    # m4 = a11(b01-b00)
    add_term(
        [(1,1,1)],
        [(0,1,1), (0,0,-1)],
        [(0,1,1), (1,1,-1)]
    )
    
    # m5 = (a00+a01)b11
    add_term(
        [(0,0,1), (0,1,1)],
        [(1,1,1)],
        [(0,1,1), (1,1,1)]
    )
    
    # m6 = (a10-a00)(b00+b01)
    add_term(
        [(1,0,1), (0,0,-1)],
        [(0,0,1), (0,1,1)],
        [(1,0,1), (1,1,-1)]
    )
    
    # m7 = (a01-a11)(b10+b11)
    add_term(
        [(0,1,1), (1,1,-1)],
        [(1,0,1), (1,1,1)],
        [(0,0,-1), (0,1,1)]
    )
    
    # Schoolbook terms for bottom 1x2 (row 2 of A)
    # c20 = a20*b00 + a21*b10
    # c21 = a20*b01 + a21*b11
    
    # m8 = a20*b00
    add_term(
        [(2,0,1)],
        [(0,0,1)],
        [(2,0,1)]
    )
    
    # m9 = a21*b10
    add_term(
        [(2,1,1)],
        [(1,0,1)],
        [(2,0,1)]
    )
    
    # m10 = a20*b01
    add_term(
        [(2,0,1)],
        [(0,1,1)],
        [(2,1,1)]
    )
    
    # m11 = a21*b11
    add_term(
        [(2,1,1)],
        [(1,1,1)],
        [(2,1,1)]
    )
    
    # This defines the local 3x2x2 decomposition.
    # Now we need to tile this into the global 3x4x8 problem.
    
    # Reset col to 0 to start building the full matrix?
    # No, the add_term above already incremented col 11 times.
    # We need to apply this pattern 8 times (2 A-splits * 4 B-splits).
    
    # Let's restart and build the full matrix correctly.
    U = [[0] * rank_total for _ in range(U_rows)]
    V = [[0] * rank_total for _ in range(V_rows)]
    W = [[0] * rank_total for _ in range(W_rows)]
    col = 0
    
    # Define the 11 local terms again as data for reuse
    # (a_entries, b_entries, c_entries)
    local_terms = [
        # m1
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1), (0,1,-1), (1,0,-1)]),
        # m2
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,-1), (1,1,1)]),
        # m3
        ([(0,0,1)], [(1,0,1), (1,1,-1)], [(0,0,1), (1,0,-1)]),
        # m4
        ([(1,1,1)], [(0,1,1), (0,0,-1)], [(0,1,1), (1,1,-1)]),
        # m5
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,1,1), (1,1,1)]),
        # m6
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,0,1), (1,1,-1)]),
        # m7
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,-1), (0,1,1)]),
        # m8
        ([(2,0,1)], [(0,0,1)], [(2,0,1)]),
        # m9
        ([(2,1,1)], [(1,0,1)], [(2,0,1)]),
        # m10
        ([(2,0,1)], [(0,1,1)], [(2,1,1)]),
        # m11
        ([(2,1,1)], [(1,1,1)], [(2,1,1)])
    ]
    
    # Tiling:
    # A (3x4) splits into A_L (cols 0,1) and A_R (cols 2,3).
    # B (4x8) splits into B_L (rows 0,1) and B_R (rows 2,3).
    # C = A_L B_L + A_R B_R.
    
    for j_A_offset in [0, 2]:
        for j_B_offset in [0, 2]:
            # B_L or B_R is 2x8. Split into 4 blocks of 2x2.
            for k_offset in [0, 2, 4, 6]:
                # For each 3x2x2 block, add the 11 terms
                for a_entries, b_entries, c_entries in local_terms:
                    # Map local indices to global indices
                    global_a = []
                    for (r, c, val) in a_entries:
                        # r is 0,1,2 (global row)
                        # c is 0,1 (local col) -> global col = j_A_offset + c
                        global_a.append((r, j_A_offset + c, val))
                    
                    global_b = []
                    for (r, c, val) in b_entries:
                        # r is 0,1 (local row) -> global row = j_B_offset + r
                        # c is 0,1 (local col) -> global col = k_offset + c
                        global_b.append((j_B_offset + r, k_offset + c, val))
                    
                    global_c = []
                    for (r, c, val) in c_entries:
                        # r is 0,1,2 (global row)
                        # c is 0,1 (local col) -> global col = k_offset + c
                        global_c.append((r, k_offset + c, val))
                    
                    # Add to global matrices
                    for (r, c, val) in global_a:
                        if val != 0:
                            U[r * m + c][col] += val
                    for (r, c, val) in global_b:
                        if val != 0:
                            V[r * p + c][col] += val
                    for (r, c, val) in global_c:
                        if val != 0:
                            W[c * n + r][col] += val
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
