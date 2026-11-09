import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 84 decomposition using Strassen's algorithm for 2x2 blocks
    # Each of the 7 Strassen terms is a 2x2 by 2x3 multiplication
    # Each 2x2 by 2x3 multiplication is decomposed using schoolbook (rank 12)
    rank = 84
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define Strassen terms
    # Each term is a tuple (x_blocks, y_blocks, c_blocks)
    # x_blocks: list of (block_index, coeff) for A
    # y_blocks: list of (block_index, coeff) for B
    # c_blocks: list of (block_index, coeff) for C
    # Block indices:
    # A: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # C: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    terms = [
        # M1
        {
            'x': [(0, 1), (3, 1)],  # A11 + A22
            'y': [(0, 1), (3, 1)],  # B11 + B22
            'c': [(0, 1), (3, 1)]   # C11 + C22
        },
        # M2
        {
            'x': [(2, 1), (3, 1)],  # A21 + A22
            'y': [(0, 1)],          # B11
            'c': [(2, 1), (3, -1)]  # C21 - C22
        },
        # M3
        {
            'x': [(0, 1)],          # A11
            'y': [(1, 1), (3, -1)], # B12 - B22
            'c': [(1, 1), (3, 1)]   # C12 + C22
        },
        # M4
        {
            'x': [(3, 1)],          # A22
            'y': [(2, 1), (0, -1)], # B21 - B11
            'c': [(0, 1), (2, 1)]   # C11 + C21
        },
        # M5
        {
            'x': [(0, 1), (1, 1)],  # A11 + A12
            'y': [(3, 1)],          # B22
            'c': [(0, -1), (1, 1)]  # -C11 + C12
        },
        # M6
        {
            'x': [(2, -1), (0, -1)],# A21 - A11
            'y': [(0, 1), (1, 1)],  # B11 + B12
            'c': [(3, 1)]           # C22
        },
        # M7
        {
            'x': [(1, 1), (3, -1)], # A12 - A22
            'y': [(2, 1), (3, 1)],  # B21 + B22
            'c': [(0, 1), (3, 1)]   # C11 + C22
        }
    ]
    
    # Helper to get global index for A block
    def a_idx(block, i_s, j_s):
        # block: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        # i_s, j_s: 0 or 1
        bi, bj = divmod(block, 2)
        gi = bi * 2 + i_s
        gj = bj * 2 + j_s
        return gi * m + gj
        
    # Helper to get global index for B block
    def b_idx(block, j_s, k_s):
        # block: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        # j_s: 0 or 1
        # k_s: 0, 1, 2
        bi, bj = divmod(block, 2)
        gj = bi * 2 + j_s
        gk = bj * 3 + k_s
        return gj * p + gk
        
    # Helper to get global index for C block
    def c_idx(block, i_s, k_s):
        # block: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        # i_s: 0 or 1
        # k_s: 0, 1, 2
        bi, bj = divmod(block, 2)
        gi = bi * 2 + i_s
        gk = bj * 3 + k_s
        return gk * n + gi
        
    col = 0
    for t_idx, term in enumerate(terms):
        for i_s in range(2):
            for j_s in range(2):
                for k_s in range(3):
                    # U part
                    for block, coeff in term['x']:
                        if block == 0 or block == 3: # A11, A22
                            if (block == 0 and i_s < 2 and j_s < 2) or (block == 3 and i_s < 2 and j_s < 2):
                                # Wait, block 0 is A11 (top-left), block 3 is A22 (bottom-right)
                                # i_s, j_s are local indices within the 2x2 block
                                pass
                        # Actually, the block index determines which 2x2 submatrix of A is used.
                        # The local indices i_s, j_s determine which element within that 2x2 submatrix.
                        # So we just need to add coeff to U[a_idx(block, i_s, j_s), col]
                        U[a_idx(block, i_s, j_s)][col] += coeff
                        
                    # V part
                    for block, coeff in term['y']:
                        V[b_idx(block, j_s, k_s)][col] += coeff
                        
                    # W part
                    for block, coeff in term['c']:
                        W[c_idx(block, i_s, k_s)][col] += coeff
                        
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
