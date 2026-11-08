"""Matrix multiplication 3x3x4 tensor decomposition, rank 34."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=4
    rank = 34
    U = [[0] * rank for _ in range(n * m)]  # 9 x 34
    V = [[0] * rank for _ in range(m * p)]  # 12 x 34
    W = [[0] * rank for _ in range(p * n)]  # 12 x 34

    col = 0

    def add_term(u_entries, v_entries, w_entries):
        nonlocal col
        for idx, val in u_entries:
            U[idx][col] += val
        for idx, val in v_entries:
            V[idx][col] += val
        for idx, val in w_entries:
            W[idx][col] += val
        col += 1

    # Partition:
    # A (3x3): rows 0-1 are top, row 2 is bottom
    #   A11 = A[0:2, 0:2] -> U indices: 0,1,3,4
    #   A12 = A[0:2, 2]   -> U indices: 2,5
    #   A21 = A[2, 0:2]   -> U indices: 6,7
    #   A22 = A[2, 2]     -> U index: 8
    #
    # B (3x4): cols 0-1 are left, cols 2-3 are right
    #   B1L = B[0:2, 0:2] -> V indices: 0,1,4,5
    #   B1R = B[0:2, 2:4] -> V indices: 2,3,6,7
    #   B2L = B[2, 0:2]   -> V indices: 8,9
    #   B2R = B[2, 2:4]   -> V indices: 10,11
    #
    # C (3x4): W index = k*n + i where k=col, i=row
    #   C1L = C[0:2, 0:2] -> W indices: 0,3,1,4
    #   C1R = C[0:2, 2:4] -> W indices: 6,9,7,10
    #   C2L = C[2, 0:2]   -> W indices: 2,5
    #   C2R = C[2, 2:4]   -> W indices: 8,11

    # --- Strassen for A11 * B1L (7 terms) ---
    # P1 = A11 * B1L
    add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (4, 1)])
    # P2 = (A[1,0]+A[1,1]) * B[0,0]
    add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (4, -1)])
    # P3 = A[0,0] * (B[1,0]-B[0,0])
    add_term([(0, 1)], [(1, 1), (5, -1)], [(3, 1), (4, 1)])
    # P4 = A[1,1] * (B[1,1]-B[0,1])
    add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)])
    # P5 = (A[0,0]+A[0,1]) * B[1,1]
    add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (3, 1)])
    # P6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(4, 1)])
    # P7 = (A[0,1]-A[1,1]) * (B[0,1]+B[1,1])
    add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)])

    # --- A12 * B2L (4 terms) ---
    # C[0,0] += A[0,2]*B[2,0]
    add_term([(2, 1)], [(8, 1)], [(0, 1)])
    # C[0,1] += A[0,2]*B[2,1]
    add_term([(2, 1)], [(9, 1)], [(3, 1)])
    # C[1,0] += A[1,2]*B[2,0]
    add_term([(5, 1)], [(8, 1)], [(1, 1)])
    # C[1,1] += A[1,2]*B[2,1]
    add_term([(5, 1)], [(9, 1)], [(4, 1)])

    # --- A21 * B1L (4 terms) ---
    # C[2,0] += A[2,0]*B[0,0]
    add_term([(6, 1)], [(0, 1)], [(2, 1)])
    # C[2,1] += A[2,0]*B[0,1]
    add_term([(6, 1)], [(1, 1)], [(5, 1)])
    # C[2,0] += A[2,1]*B[1,0]
    add_term([(7, 1)], [(4, 1)], [(2, 1)])
    # C[2,1] += A[2,1]*B[1,1]
    add_term([(7, 1)], [(5, 1)], [(5, 1)])

    # --- A22 * B2L (2 terms) ---
    # C[2,0] += A[2,2]*B[2,0]
    add_term([(8, 1)], [(8, 1)], [(2, 1)])
    # C[2,1] += A[2,2]*B[2,1]
    add_term([(8, 1)], [(9, 1)], [(5, 1)])

    # --- Strassen for A11 * B1R (7 terms) ---
    # P1 = A11 * B1R
    add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(6, 1), (10, 1)])
    # P2 = (A[1,0]+A[1,1]) * B[0,2]
    add_term([(3, 1), (4, 1)], [(2, 1)], [(7, 1), (10, -1)])
    # P3 = A[0,0] * (B[1,2]-B[0,2])
    add_term([(0, 1)], [(3, 1), (7, -1)], [(9, 1), (10, 1)])
    # P4 = A[1,1] * (B[1,3]-B[0,3])
    add_term([(4, 1)], [(6, 1), (2, -1)], [(6, 1), (7, 1)])
    # P5 = (A[0,0]+A[0,1]) * B[1,3]
    add_term([(0, 1), (1, 1)], [(7, 1)], [(6, -1), (9, 1)])
    # P6 = (A[1,0]-A[0,0]) * (B[0,2]+B[0,3])
    add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(10, 1)])
    # P7 = (A[0,1]-A[1,1]) * (B[0,3]+B[1,3])
    add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(6, 1)])

    # --- A12 * B2R (4 terms) ---
    # C[0,2] += A[0,2]*B[2,2]
    add_term([(2, 1)], [(10, 1)], [(6, 1)])
    # C[0,3] += A[0,2]*B[2,3]
    add_term([(2, 1)], [(11, 1)], [(9, 1)])
    # C[1,2] += A[1,2]*B[2,2]
    add_term([(5, 1)], [(10, 1)], [(7, 1)])
    # C[1,3] += A[1,2]*B[2,3]
    add_term([(5, 1)], [(11, 1)], [(10, 1)])

    # --- A21 * B1R (4 terms) ---
    # C[2,2] += A[2,0]*B[0,2]
    add_term([(6, 1)], [(2, 1)], [(8, 1)])
    # C[2,3] += A[2,0]*B[0,3]
    add_term([(6, 1)], [(3, 1)], [(11, 1)])
    # C[2,2] += A[2,1]*B[1,2]
    add_term([(7, 1)], [(6, 1)], [(8, 1)])
    # C[2,3] += A[2,1]*B[1,3]
    add_term([(7, 1)], [(7, 1)], [(11, 1)])

    # --- A22 * B2R (2 terms) ---
    # C[2,2] += A[2,2]*B[2,2]
    add_term([(8, 1)], [(10, 1)], [(8, 1)])
    # C[2,3] += A[2,2]*B[2,3]
    add_term([(8, 1)], [(11, 1)], [(11, 1)])

    assert col == rank, f"Expected {rank} terms, got {col}"
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
