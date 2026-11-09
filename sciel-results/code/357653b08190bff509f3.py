"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    # For 2x4x4: U is (8, R), V is (16, R), W is (8, R)

    # We use a direct Strassen-based decomposition with rank 28.
    # The 2x4x4 multiplication can be viewed as 4 blocks of 2x2x2.
    # Each 2x2x2 block uses Strassen (rank 7), total rank = 4*7 = 28.

    R = 28
    U = [[0] * R for _ in range(n * m)]   # 8 x R
    V = [[0] * R for _ in range(m * p)]   # 16 x R
    W = [[0] * R for _ in range(p * n)]   # 8 x R

    # Strassen 2x2x2 decomposition.
    # For A (2x2), B (2x2), C = A*B:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)*e
    # M3 = a*(f-h)
    # M4 = d*(g-e)
    # M5 = (a+b)*h
    # M6 = (c-a)*(e+f)
    # M7 = (b-d)*(g+h)
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 - M4
    # C[1,1] = M1 - M2 + M3 + M6

    # In the tensor decomposition, for each of the 4 blocks, we assign 7 columns.
    # Block layout:
    #   Block b corresponds to j_offset = (b >> 1) * 2, k_offset = (b & 1) * 2
    #   Wait, let's think: A is 2x4, so rows i=0,1, cols j=0,1,2,3
    #   B is 4x4, so rows j=0,1,2,3, cols k=0,1,2,3
    #   C is 2x4, so rows i=0,1, cols k=0,1,2,3
    #
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # We can split j into {0,1} and {2,3}, and k into {0,1} and {2,3}.
    # So 4 blocks: (j in {0,1}, k in {0,1}), (j in {0,1}, k in {2,3}),
    #               (j in {2,3}, k in {0,1}), (j in {2,3}, k in {2,3})
    #
    # For each block, we have a 2x2 A submatrix (rows i=0,1, cols j=j_off..j_off+1)
    # and a 2x2 B submatrix (rows j=j_off..j_off+1, cols k=k_off..k_off+1)
    # and a 2x2 C submatrix (rows i=0,1, cols k=k_off..k_off+1)

    block_offsets = [
        (0, 0),  # j=0,1, k=0,1
        (0, 2),  # j=0,1, k=2,3
        (2, 0),  # j=2,3, k=0,1
        (2, 2),  # j=2,3, k=2,3
    ]

    # Strassen terms: each term has U entries (i, local_j, coeff), V entries (local_j, local_k, coeff), W entries (local_k, i, coeff)
    # U index: (i, local_j) -> global row i * m + (j_off + local_j)
    # V index: (local_j, local_k) -> global row (j_off + local_j) * p + (k_off + local_k)
    # W index: (local_k, i) -> global row (k_off + local_k) * n + i

    strassen = [
        # M1 = (a+d)(e+h): U: (0,0,1),(1,1,1); V: (0,0,1),(1,1,1); W: (0,0,1),(1,1,1)
        {'U': [(0, 0, 1), (1, 1, 1)], 'V': [(0, 0, 1), (1, 1, 1)], 'W': [(0, 0, 1), (1, 1, 1)]},
        # M2 = (c+d)*e: U: (1,0,1),(1,1,1); V: (0,0,1); W: (1,1,1),(1,0,-1)
        # C[1,1] += M2, C[1,0] -= M2  => W: (local_k=1, i=1, 1), (local_k=0, i=1, -1)
        {'U': [(1, 0, 1), (1, 1, 1)], 'V': [(0, 0, 1)], 'W': [(1, 1, 1), (0, 1, -1)]},
        # M3 = a*(f-h): U: (0,0,1); V: (0,1,1),(1,1,-1); W: (1,0,1),(1,1,1)
        # C[0,1] += M3, C[1,1] += M3  => W: (local_k=1, i=0, 1), (local_k=1, i=1, 1)
        {'U': [(0, 0, 1)], 'V': [(0, 1, 1), (1, 1, -1)], 'W': [(1, 0, 1), (1, 1, 1)]},
        # M4 = d*(g-e): U: (1,1,1); V: (1,0,1),(0,0,-1); W: (0,0,1),(0,1,1)
        # C[0,0] += M4, C[1,0] -= M4  => W: (local_k=0, i=0, 1), (local_k=0, i=1, -1)
        {'U': [(1, 1, 1)], 'V': [(1, 0, 1), (0, 0, -1)], 'W': [(0, 0, 1), (0, 1, -1)]},
        # M5 = (a+b)*h: U: (0,0,1),(0,1,1); V: (1,1,1); W: (0,0,-1),(1,0,1)
        # C[0,0] -= M5, C[0,1] += M5  => W: (local_k=0, i=0, -1), (local_k=1, i=0, 1)
        {'U': [(0, 0, 1), (0, 1, 1)], 'V': [(1, 1, 1)], 'W': [(0, 0, -1), (1, 0, 1)]},
        # M6 = (c-a)*(e+f): U: (1,0,1),(0,0,-1); V: (0,0,1),(0,1,1); W: (1,1,1)
        # C[1,1] += M6  => W: (local_k=1, i=1, 1)
        {'U': [(1, 0, 1), (0, 0, -1)], 'V': [(0, 0, 1), (0, 1, 1)], 'W': [(1, 1, 1)]},
        # M7 = (b-d)*(g+h): U: (0,1,1),(1,1,-1); V: (1,0,1),(1,1,1); W: (0,0,1)
        # C[0,0] += M7  => W: (local_k=0, i=0, 1)
        {'U': [(0, 1, 1), (1, 1, -1)], 'V': [(1, 0, 1), (1, 1, 1)], 'W': [(0, 0, 1)]},
    ]

    col = 0
    for j_off, k_off in block_offsets:
        for term in strassen:
            # U entries
            for (li, lj, coeff) in term['U']:
                row = li * m + (j_off + lj)
                U[row][col] += coeff
            # V entries
            for (lj, lk, coeff) in term['V']:
                row = (j_off + lj) * p + (k_off + lk)
                V[row][col] += coeff
            # W entries
            for (lk, li, coeff) in term['W']:
                row = (k_off + lk) * n + li
                W[row][col] += coeff
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
