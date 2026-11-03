"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition using Strassen for 2x2x2 blocks
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Iterate over 2 j-blocks and 4 k-blocks
    for j_block in range(2):
        j_off = j_block * 2
        for k_block in range(4):
            k_off = k_block * 2
            
            # Local indices:
            # i: 0, 1
            # j: 0, 1 (global j_off, j_off+1)
            # k: 0, 1 (global k_off, k_off+1)
            
            # Term 1: (a11 + a22) * (b11 + b22)
            # a11: i=0, j=j_off
            # a22: i=1, j=j_off+1
            # b11: j=j_off, k=k_off
            # b22: j=j_off+1, k=k_off+1
            # c11: i=0, k=k_off
            # c22: i=1, k=k_off+1
            U[0 * m + j_off][col] += 1
            U[1 * m + (j_off + 1)][col] += 1
            V[j_off * p + k_off][col] += 1
            V[(j_off + 1) * p + (k_off + 1)][col] += 1
            W[k_off * n + 0][col] += 1
            W[(k_off + 1) * n + 1][col] += 1
            col += 1
            
            # Term 2: (a21 + a22) * b11
            # a21: i=1, j=j_off
            # a22: i=1, j=j_off+1
            # b11: j=j_off, k=k_off
            # c21: i=1, k=k_off
            U[1 * m + j_off][col] += 1
            U[1 * m + (j_off + 1)][col] += 1
            V[j_off * p + k_off][col] += 1
            W[k_off * n + 1][col] += 1
            col += 1
            
            # Term 3: a11 * (b12 - b22)
            # a11: i=0, j=j_off
            # b12: j=j_off, k=k_off+1
            # b22: j=j_off+1, k=k_off+1
            # c12: i=0, k=k_off+1
            U[0 * m + j_off][col] += 1
            V[j_off * p + (k_off + 1)][col] += 1
            V[(j_off + 1) * p + (k_off + 1)][col] -= 1
            W[(k_off + 1) * n + 0][col] += 1
            col += 1
            
            # Term 4: a22 * (b21 - b11)
            # a22: i=1, j=j_off+1
            # b21: j=j_off+1, k=k_off
            # b11: j=j_off, k=k_off
            # c21: i=1, k=k_off
            U[1 * m + (j_off + 1)][col] += 1
            V[(j_off + 1) * p + k_off][col] += 1
            V[j_off * p + k_off][col] -= 1
            W[k_off * n + 1][col] += 1
            col += 1
            
            # Term 5: (a11 + a12) * b22
            # a11: i=0, j=j_off
            # a12: i=0, j=j_off+1
            # b22: j=j_off+1, k=k_off+1
            # c12: i=0, k=k_off+1
            U[0 * m + j_off][col] += 1
            U[0 * m + (j_off + 1)][col] += 1
            V[(j_off + 1) * p + (k_off + 1)][col] += 1
            W[(k_off + 1) * n + 0][col] += 1
            col += 1
            
            # Term 6: (a21 - a11) * (b11 + b12)
            # a21: i=1, j=j_off
            # a11: i=0, j=j_off
            # b11: j=j_off, k=k_off
            # b12: j=j_off, k=k_off+1
            # c22: i=1, k=k_off+1
            U[1 * m + j_off][col] += 1
            U[0 * m + j_off][col] -= 1
            V[j_off * p + k_off][col] += 1
            V[j_off * p + (k_off + 1)][col] += 1
            W[(k_off + 1) * n + 1][col] += 1
            col += 1
            
            # Term 7: (a12 - a22) * (b21 + b22)
            # a12: i=0, j=j_off+1
            # a22: i=1, j=j_off+1
            # b21: j=j_off+1, k=k_off
            # b22: j=j_off+1, k=k_off+1
            # c22: i=1, k=k_off+1
            U[0 * m + (j_off + 1)][col] += 1
            U[1 * m + (j_off + 1)][col] -= 1
            V[(j_off + 1) * p + k_off][col] += 1
            V[(j_off + 1) * p + (k_off + 1)][col] += 1
            W[(k_off + 1) * n + 1][col] += 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
