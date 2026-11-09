import time
import numpy as np

def construct():
    n, m, p = 4, 4, 8
    start_time = time.time()
    deadline = start_time + 140

    # Block decomposition: 2x2 blocks
    # Left matrix: 4x4 -> 2x2 blocks of 2x2 (i2 in {0,1}, k2 in {0,1})
    # Right matrix: 4x8 -> 2x4 blocks of 2x2 (k2 in {0,1}, j2 in {0,1,2,3})
    # Output: 4x8 -> 2x4 blocks of 2x2 (i2 in {0,1}, j2 in {0,1,2,3})
    
    # Strassen for 2x2 * 2x2 -> 7 multiplications
    # For each block triple (i2, k2, j2), we add 7 rank-1 terms
    # Total rank = 2 * 2 * 4 * 7 = 112... wait, that's worse than 128?
    # No: 2*2*4 = 16 block triples, each rank 7 -> 112. That's better than 128!
    
    # Actually let me recount:
    # i2 in {0,1} (2 values), k2 in {0,1} (2 values), j2 in {0,1,2,3} (4 values)
    # Number of block multiplications = 2 * 2 * 4 = 16
    # Each uses Strassen rank 7
    # Total rank = 16 * 7 = 112
    
    # But wait - can we do better? What about using the fact that the right matrix
    # has 4 blocks of 2 columns? We could potentially use a different decomposition.
    
    # Let's first implement the Strassen block approach for rank 112.
    
    # Strassen coefficients for 2x2 matrix multiplication:
    # m1 = (A00+A11)(B00+B11)
    # m2 = (A10+A11)B00
    # m3 = A00(B10-B11)
    # m4 = A11(B01-B10)
    # m5 = (A00+A01)B11
    # m6 = (A10-A00)(B00+B10)
    # m7 = (A01-A11)(B01+B11)
    
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # For each multiplication mc, we need to specify:
    # - Which (ii, kk) entries of A are involved and with what coefficient
    # - Which (kk, jj) entries of B are involved and with what coefficient
    # - Which (ii, jj) entries of C are involved and with what coefficient
    
    # Let's define the 7 Strassen multiplications:
    # Each is a product of a linear form in A and a linear form in B
    # Then C is a linear combination of these products
    
    # For a given multiplication mc:
    # A_part: dict mapping (ii, kk) -> coeff
    # B_part: dict mapping (kk, jj) -> coeff
    # C_part: dict mapping (ii, jj) -> coeff
    
    # m1: A: {(0,0):1, (1,1):1}, B: {(0,0):1, (0,1):1, (1,0):1, (1,1):1}
    #     C: {(0,0):1, (1,1):1}
    
    # m2: A: {(1,0):1, (1,1):1}, B: {(0,0):1}
    #     C: {(1,0):1, (1,1):-1}
    
    # m3: A: {(0,0):1}, B: {(1,0):1, (1,1):-1}
    #     C: {(0,1):1, (1,1):1}
    
    # m4: A: {(1,1):1}, B: {(0,1):1, (1,0):-1}
    #     C: {(0,0):1, (1,0):1}
    
    # m5: A: {(0,0):1, (0,1):1}, B: {(1,1):1}
    #     C: {(0,0):-1, (0,1):1}
    
    # m6: A: {(1,0):1, (0,0):-1}, B: {(0,0):1, (1,0):1}
    #     C: {(1,1):1}
    
    # m7: A: {(0,1):1, (1,1):-1}, B: {(0,1):1, (1,1):1}
    #     C: {(0,0):1}
    
    strassen_mults = [
        # (A_part, B_part, C_part)
        ({(0,0):1, (1,1):1}, {(0,0):1, (0,1):1, (1,0):1, (1,1):1}, {(0,0):1, (1,1):1}),  # m1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),  # m2
        ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}),  # m3
        ({(1,1):1}, {(0,1):1, (1,0):-1}, {(0,0):1, (1,0):1}),  # m4
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),  # m5
        ({(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}, {(1,1):1}),  # m6
        ({(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}, {(0,0):1}),  # m7
    ]
    
    rank = 2 * 2 * 4 * 7  # 112
    U = [[0] * rank for _ in range(n * m)]  # 16 x rank
    V = [[0] * rank for _ in range(m * p)]  # 32 x rank
    W = [[0] * rank for _ in range(p * n)]  # 32 x rank
    
    col = 0
    for i2 in range(2):
        for k2 in range(2):
            for j2 in range(4):
                for A_part, B_part, C_part in strassen_mults:
                    # A_part: (ii, kk) -> coeff, where ii in {0,1}, kk in {0,1}
                    for (ii, kk), coeff_a in A_part.items():
                        i = i2 * 2 + ii
                        k = k2 * 2 + kk
                        U[i * m + k][col] = coeff_a
                    
                    # B_part: (kk, jj) -> coeff, where kk in {0,1}, jj in {0,1}
                    for (kk, jj), coeff_b in B_part.items():
                        k = k2 * 2 + kk
                        j = j2 * 2 + jj
                        V[k * p + j][col] = coeff_b
                    
                    # C_part: (ii, jj) -> coeff, where ii in {0,1}, jj in {0,1}
                    for (ii, jj), coeff_c in C_part.items():
                        i = i2 * 2 + ii
                        j = j2 * 2 + jj
                        W[j * n + i][col] = coeff_c
                    
                    col += 1
    
    # Verify the solution
    def verify(U_list, V_list, W_list, tol=1e-9):
        U = np.array(U_list, dtype=float)
        V = np.array(V_list, dtype=float)
        W = np.array(W_list, dtype=float)
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    if not verify(U, V, W):
        # Fallback to schoolbook
        rank_school = n * m * p
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
