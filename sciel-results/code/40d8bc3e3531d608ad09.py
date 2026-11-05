"""
Rank-189 decomposition of 6x6 matrix multiplication using block-structured
Strassen: partition 6x6 into 3x3 blocks of 2x2 matrices, use schoolbook
for the 3x3 block structure (27 block products) and rank-7 Strassen for
each 2x2 block multiplication.

Total rank = 27 * 7 = 189.
"""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-189 decomposition of 6x6 matrix multiplication.
    
    Strategy: Partition each 6x6 matrix into a 3x3 grid of 2x2 blocks.
    Use schoolbook multiplication at the block level (27 block products),
    and Strassen's 7-term algorithm for each 2x2 block multiplication.
    
    Total rank = 27 * 7 = 189.
    
    Returns dict with U (36x189), V (36x189), W (36x189) integer matrices
    such that for the matrix multiplication tensor T[i,k,j] = delta:
      sum_r U[i,k,r] * V[k,j,r] * W[j,i,r] = T[i,k,j]
    
    Note: W follows the trace convention W[k*n+i, r] which for n=p=6
    means W is indexed by (output_row, input_row) pairs.
    """
    n, m, p = DIMENSIONS  # 6, 6, 6
    rank = 189
    
    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]  # (36, 189) - indices (i, k)
    V = [[0] * rank for _ in range(m * p)]  # (36, 189) - indices (k, j)
    W = [[0] * rank for _ in range(p * n)]  # (36, 189) - indices (j, i)
    
    # Strassen 7-term decomposition for 2x2 matrix multiplication:
    # For A = [[a,b],[c,d]] and B = [[e,f],[g,h]]:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each term: (u_indices, u_coeffs, v_indices, v_coeffs, w_indices, w_coeffs)
    # Indices are into the 2x2 block (0,1,2,3 for top-left, top-right, bottom-left, bottom-right)
    strassen_terms = [
        # M1 = (a+d)(e+h): U has a,d; V has e,h
        {
            'u': [(0, 1), (3, 1)],
            'v': [(0, 1), (3, 1)],
            'w': [(0, 1), (3, 1)]  # contributes to C11 and C22
        },
        # M2 = (c+d)e: U has c,d; V has e
        {
            'u': [(2, 1), (3, 1)],
            'v': [(0, 1)],
            'w': [(2, 1), (3, -1)]  # contributes to C21 (+) and C22 (-)
        },
        # M3 = a(f-h): U has a; V has f,h
        {
            'u': [(0, 1)],
            'v': [(1, 1), (3, -1)],
            'w': [(1, 1), (3, 1)]  # contributes to C12 (+) and C22 (+)
        },
        # M4 = d(g-e): U has d; V has g,e
        {
            'u': [(3, 1)],
            'v': [(2, 1), (0, -1)],
            'w': [(0, 1), (2, 1)]  # contributes to C11 (+) and C21 (+)
        },
        # M5 = (a+b)h: U has a,b; V has h
        {
            'u': [(0, 1), (1, 1)],
            'v': [(3, 1)],
            'w': [(0, -1), (1, 1)]  # contributes to C11 (-) and C12 (+)
        },
        # M6 = (c-a)(e+f): U has c,a; V has e,f
        {
            'u': [(2, 1), (0, -1)],
            'v': [(0, 1), (1, 1)],
            'w': [(3, 1)]  # contributes to C22 (+)
        },
        # M7 = (b-d)(g+h): U has b,d; V has g,h
        {
            'u': [(1, 1), (3, -1)],
            'v': [(2, 1), (3, 1)],
            'w': [(0, 1)]  # contributes to C11 (+)
        },
    ]
    
    # Iterate over all block positions and Strassen terms
    col = 0
    for bi in range(3):      # block row of A (and C)
        for bk in range(3):  # block col of A / block row of B
            for bj in range(3):  # block col of B (and C)
                for t in range(7):
                    term = strassen_terms[t]
                    
                    # Fill U: A is partitioned into blocks, A[bi][bk] is 2x2
                    # Global index for A[i,k]: i = 2*bi + r, k = 2*bk + c
                    # U index = i * 6 + k
                    for (block_idx, coeff) in term['u']:
                        r = block_idx // 2  # row within 2x2 block
                        c = block_idx % 2   # col within 2x2 block
                        global_i = 2 * bi + r
                        global_k = 2 * bk + c
                        U[global_i * 6 + global_k][col] += coeff
                    
                    # Fill V: B is partitioned into blocks, B[bk][bj] is 2x2
                    # Global index for B[k,j]: k = 2*bk + r, j = 2*bj + c
                    # V index = k * 6 + j
                    for (block_idx, coeff) in term['v']:
                        r = block_idx // 2
                        c = block_idx % 2
                        global_k = 2 * bk + r
                        global_j = 2 * bj + c
                        V[global_k * 6 + global_j][col] += coeff
                    
                    # Fill W: C output block C[bi][bj] is 2x2
                    # Global index for C[j,i] (trace convention): 
                    # C[bi][bj] has elements C[2*bi+r][2*bj+c]
                    # W index = j * 6 + i where j = 2*bj + c, i = 2*bi + r
                    for (block_idx, coeff) in term['w']:
                        r = block_idx // 2  # row within C block
                        c = block_idx % 2   # col within C block
                        global_i = 2 * bi + r
                        global_j = 2 * bj + c
                        W[global_j * 6 + global_i][col] += coeff
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
