"""Matrix multiplication 2x4x8 via Strassen block decomposition (rank 56)."""
DIMENSIONS = [2, 4, 8]

# Strassen rank-1 tensors for 2x2 matrix multiplication.
# Each term: (a_coeffs, b_coeffs, c_coeffs) where a_coeffs[i] is coeff of A_i,
# b_coeffs[j] is coeff of B_j, c_coeffs[k] is coeff of C_k.
# Local indices: A: [00,01,10,11] -> [0,1,2,3]
#                B: [00,01,10,11] -> [0,1,2,3]
#                C: [00,01,10,11] -> [0,1,2,3]

STRASSEN_TERMS = [
    ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # m1
    ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),   # m2
    ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),   # m3
    ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),   # m4
    ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),   # m5
    ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),   # m6
    ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),   # m7
]


def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    # A: 2x4, B: 4x8, C = A*B: 2x8
    # U: (n*m, rank) = (8, rank), row i*m+j
    # V: (m*p, rank) = (32, rank), row j*p+k
    # W: (p*n, rank) = (16, rank), row k*n+i
    
    # Partition C into 4 column blocks of width 2.
    # For block t: C[:, 2t:2t+2] = A[:, 0:2] * B[0:2, 2t:2t+2] + A[:, 2:4] * B[2:4, 2t:2t+2]
    # 8 total 2x2 matmuls, each rank 7 via Strassen. Total rank = 56.
    
    rank = 56
    U = [[0] * rank for _ in range(8)]
    V = [[0] * rank for _ in range(32)]
    W = [[0] * rank for _ in range(16)]
    
    col = 0
    for t in range(4):
        for s in range(2):
            # s=0: A[:,0:2] * B[0:2, 2t:2t+2]
            # s=1: A[:,2:4] * B[2:4, 2t:2t+2]
            j_base = s * 2  # row offset in B (and col offset in A)
            
            for (a_coeffs, b_coeffs, c_coeffs) in STRASSEN_TERMS:
                # Set U entries for this rank-1 tensor
                for li in range(2):
                    for lj in range(2):
                        local_a_idx = li * 2 + lj
                        u_row = li * m + (j_base + lj)
                        U[u_row][col] = a_coeffs[local_a_idx]
                
                # Set V entries
                for lj in range(2):
                    for lk in range(2):
                        local_b_idx = lj * 2 + lk
                        v_row = (j_base + lj) * p + (2 * t + lk)
                        V[v_row][col] = b_coeffs[local_b_idx]
                
                # Set W entries
                for li in range(2):
                    for lk in range(2):
                        local_c_idx = li * 2 + lk
                        w_row = (2 * t + lk) * n + li
                        W[w_row][col] = c_coeffs[local_c_idx]
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
