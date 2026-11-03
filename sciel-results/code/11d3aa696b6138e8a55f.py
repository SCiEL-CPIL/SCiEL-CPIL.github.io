"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Block decomposition:
    # m=5 split into 2, 3
    # p=7 split into 2, 2, 2, 1
    # Sub-blocks:
    # m=2, p=2: Rank 4 (3x2x2)
    # m=2, p=1: Rank 2 (3x2x1)
    # m=3, p=2: Rank 5 (3x3x2)
    # m=3, p=1: Rank 3 (3x3x1)
    
    ranks = {
        (2, 2): 4,
        (2, 1): 2,
        (3, 2): 5,
        (3, 1): 3
    }
    
    total_rank = 0
    # Calculate total rank
    # m=2 block: 3*(2,2) + 1*(2,1) = 3*4 + 2 = 14
    # m=3 block: 3*(3,2) + 1*(3,1) = 3*5 + 3 = 18
    # Total 32
    total_rank = 14 + 18
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add 3x2x2 block
    def add_3x2x2(i0, j0, k0, c):
        # r=0,1 (k_l=0)
        for r in range(2):
            jl = r
            for il in range(n):
                U[(i0 + il) * m + (j0 + jl)][c + r] = 1
            V[(j0 + jl) * p + (k0 + 0)][c + r] = 1
            for il in range(n):
                W[(k0 + 0) * n + (i0 + il)][c + r] = 1
        # r=2,3 (k_l=1)
        for r in range(2):
            jl = r
            for il in range(n):
                U[(i0 + il) * m + (j0 + jl)][c + 2 + r] = 1
            V[(j0 + jl) * p + (k0 + 1)][c + 2 + r] = 1
            for il in range(n):
                W[(k0 + 1) * n + (i0 + il)][c + 2 + r] = 1

    # Helper to add 3x2x1 block
    def add_3x2x1(i0, j0, k0, c):
        # r=0,1 (j_l=r, k_l=0)
        for r in range(2):
            jl = r
            for il in range(n):
                U[(i0 + il) * m + (j0 + jl)][c + r] = 1
            V[(j0 + jl) * p + (k0 + 0)][c + r] = 1
            for il in range(n):
                W[(k0 + 0) * n + (i0 + il)][c + r] = 1

    # Helper to add 3x3x2 block
    def add_3x3x2(i0, j0, k0, c):
        # Part 1: 3x2x2 on j_l in {0,1}
        # r=0,1 (k_l=0)
        for r in range(2):
            jl = r
            for il in range(n):
                U[(i0 + il) * m + (j0 + jl)][c + r] = 1
            V[(j0 + jl) * p + (k0 + 0)][c + r] = 1
            for il in range(n):
                W[(k0 + 0) * n + (i0 + il)][c + r] = 1
        # r=2,3 (k_l=1)
        for r in range(2):
            jl = r
            for il in range(n):
                U[(i0 + il) * m + (j0 + jl)][c + 2 + r] = 1
            V[(j0 + jl) * p + (k0 + 1)][c + 2 + r] = 1
            for il in range(n):
                W[(k0 + 1) * n + (i0 + il)][c + 2 + r] = 1
        
        # Part 2: 3x1x2 on j_l=2
        # r=4
        jl = 2
        for il in range(n):
            U[(i0 + il) * m + (j0 + jl)][c + 4] = 1
        for kl in range(2):
            V[(j0 + jl) * p + (k0 + kl)][c + 4] = 1
            for il in range(n):
                W[(k0 + kl) * n + (i0 + il)][c + 4] = 1

    # Helper to add 3x3x1 block
    def add_3x3x1(i0, j0, k0, c):
        # r=0,1,2 (j_l=r, k_l=0)
        for r in range(3):
            jl = r
            for il in range(n):
                U[(i0 + il) * m + (j0 + jl)][c + r] = 1
            V[(j0 + jl) * p + (k0 + 0)][c + r] = 1
            for il in range(n):
                W[(k0 + 0) * n + (i0 + il)][c + r] = 1

    # m=2 block (j0=0)
    # p=2 blocks (k0=0, 2, 4)
    add_3x2x2(0, 0, 0, col); col += 4
    add_3x2x2(0, 0, 2, col); col += 4
    add_3x2x2(0, 0, 4, col); col += 4
    # p=1 block (k0=6)
    add_3x2x1(0, 0, 6, col); col += 2
    
    # m=3 block (j0=2)
    # p=2 blocks (k0=0, 2, 4)
    add_3x3x2(0, 2, 0, col); col += 5
    add_3x3x2(0, 2, 2, col); col += 5
    add_3x3x2(0, 2, 4, col); col += 5
    # p=1 block (k0=6)
    add_3x3x1(0, 2, 6, col); col += 3
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
