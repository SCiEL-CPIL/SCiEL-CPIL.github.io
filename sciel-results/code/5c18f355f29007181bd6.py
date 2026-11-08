import numpy as np

DIMENSIONS = [3, 4, 8]


def construct():
    n, m, p = 3, 4, 8
    rank = 88

    # U shape: (n*m, rank) = (12, 88)
    # V shape: (m*p, rank) = (32, 88)
    # W shape: (p*n, rank) = (24, 88)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    col = 0

    # Block 0: row 0 of A (i=0) times all columns j=0..3, all rows k=0..7 of B
    # This is a 1x4 by 4x8 multiplication: rank 32 (each output C[0,k] = sum_j A[0,j]*B[j,k])
    # But we can do better: since A[0,*] is a single row, each C[0,k] needs m=4 scalar products.
    # Actually the naive approach for row 0: for each j in 0..3, for each k in 0..7:
    #   U[0*m+j][col] = 1  (A[0,j])
    #   V[j*p+k][col] = 1  (B[j,k])
    #   W[k*n+0][col] = 1  (C[0,k])
    for j in range(m):
        for k in range(p):
            U[0 * m + j, col] = 1
            V[j * p + k, col] = 1
            W[k * n + 0, col] = 1
            col += 1
    # col is now 32

    # Strassen decomposition for 2x2 matrices (rank 7)
    # A2 = A[1:3, j_start:j_start+2], B2 = B[j_start:j_start+2, k_start:k_start+2], C2 = C[k_start:k_start+2, 1:3]
    # Flatten indices: A2 index l = i'*2 + j', B2 index l = j'*2 + k', C2 index l = i'*2 + k'
    # where i',j',k' in {0,1}
    strassen_U = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ], dtype=float)

    strassen_V = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ], dtype=float)

    strassen_W = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ], dtype=float)

    # For each pair of j-columns (2 blocks) and each pair of k-rows (4 blocks):
    # Use Strassen on A[1:3, j_block*2:(j_block+1)*2] (2x2), B[j_block*2:(j_block+1)*2, k_block*2:(k_block+1)*2] (2x2)
    # Output goes to C[k_block*2:(k_block+1)*2, 1:3] (2x2)
    for j_block in range(2):  # j in {0,1}
        j_start = j_block * 2
        for k_block in range(4):  # k in {0,1,2,3}
            k_start = k_block * 2
            for t in range(7):
                # U: A indices, flattened as i'*2 + j'
                for l in range(4):
                    val = strassen_U[t, l]
                    if val != 0:
                        local_i_prime = l // 2
                        local_j_prime = l % 2
                        global_i = 1 + local_i_prime  # i in {1,2}
                        global_j = j_start + local_j_prime
                        U[global_i * m + global_j, col] += val
                # V: B indices, flattened as j'*2 + k'
                for l in range(4):
                    val = strassen_V[t, l]
                    if val != 0:
                        local_j_prime = l // 2
                        local_k_prime = l % 2
                        global_j = j_start + local_j_prime
                        global_k = k_start + local_k_prime
                        V[global_j * p + global_k, col] += val
                # W: C indices, flattened as i'*2 + k'
                for l in range(4):
                    val = strassen_W[t, l]
                    if val != 0:
                        local_i_prime = l // 2
                        local_k_prime = l % 2
                        global_i = 1 + local_i_prime  # i in {1,2}
                        global_k = k_start + local_k_prime
                        W[global_k * n + global_i, col] += val
                col += 1
    # col should be 32 + 2*4*7 = 32 + 56 = 88

    return {"U": U, "V": V, "W": W}
