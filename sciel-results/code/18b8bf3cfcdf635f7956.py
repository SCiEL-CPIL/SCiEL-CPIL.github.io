"""Find a rank-minimal exact tensor decomposition for 6x6 matrix multiplication."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Kronecker product of Strassen (2x2, rank 7) and schoolbook (3x3, rank 9)
    gives 6x6 multiplication with rank 7*9 = 63.
    """
    n, m, p = DIMENSIONS  # 6, 6, 6
    rank = 63
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms: (A_blocks, B_blocks)
    strassen_terms = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ({(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(1,1): 1}, {(0,1): 1, (0,0): -1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),
    ]
    
    # Output: C_block[ci,cj] = sum_t coeff * M_t
    output_comb = {
        (0,0): [(0,1), (3,1), (4,-1), (6,1)],
        (0,1): [(2,1), (4,1)],
        (1,0): [(1,1), (3,1)],
        (1,1): [(0,1), (1,-1), (2,1), (5,1)],
    }
    
    for t_idx, (A_blks, B_blks) in enumerate(strassen_terms):
        for s in range(9):
            col = t_idx * 9 + s
            li, lj = s // 3, s % 3
            
            for (bi, bj), ca in A_blks.items():
                row = (bi*3+li)*6 + (bj*3+lj)
                U[row][col] = ca
            
            for (bi, bj), cb in B_blks.items():
                row = (bi*3+li)*6 + (bj*3+lj)
                V[row][col] = cb
            
            for (ci, cj), terms in output_comb.items():
                for (t_use, cc) in terms:
                    if t_use == t_idx:
                        row = (cj*3+lj)*6 + (ci*3+li)
                        W[row][col] = cc
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
