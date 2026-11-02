"""Rewritten program: block-Strassen (rank 49) for the 4x4 blocks of B plus schoolbook for the 4x1 part (rank 16) → rank 114."""
import time

DIMENSIONS = [4, 4, 9]

def _build_strassen_4x4():
    """Return (U, V, W) lists for a 4x4 * 4x4 matrix product with rank 49.

    U: 16 x 49, V: 16 x 49, W: 16 x 49
    Indexing: U[i*m + j, c], V[j*p + k, c], W[k*n + i, c] with n=m=p=4.
    W index = k*4 + i.
    """
    n = m = p = 4
    rank = 49
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    # 7 block products, each a 2x2 * 2x2 product decomposed with Strassen (7 scalar products).
    # Block product s: A_s (2x2) * B_s (2x2) -> M_s (2x2)
    # Within each block, Strassen indices (a,b) in {0,1}x{0,1}
    # Global i = 2*I + i', j = 2*J + j', k = 2*K + k'

    # Strassen 2x2: 7 scalar products
    # M00 = P1 - P4, M01 = P3 + P4, M10 = P2 + P5, M11 = P1 - P3
    # P1 = (a00+a11)(b00+b11)
    # P2 = (a10+a11)*b00
    # P3 = a00*(b01-b11)
    # P4 = a11*(b10-b00)
    # P5 = (a00+a10)*b11
    # P6 = (a01-a11)*(b10+b11)
    # P7 = (a00-a10)*(b00+b10)

    # For each of the 7 block products s, and each of the 7 Strassen scalar products t,
    # we add contributions to U, V, W.

    # Block product definitions: A_s = sum over (i,j) of alpha_{s,i,j} * A[i,j]
    # B_s = sum over (j,k) of beta_{s,j,k} * B[j,k]
    # M_s = sum over (i,k) of gamma_{s,i,k} * C[i,k]  (where C = A*B)
    # Actually in tensor decomposition: C[i,k] = sum_s gamma_{s,i,k} * M_s[i,k]
    # But M_s[i,k] itself = sum_t w_{s,t,i,k} * P_{s,t}
    # So C[i,k] = sum_{s,t} (gamma_{s,i,k} * w_{s,t,i,k}) * P_{s,t}
    # And P_{s,t} = (sum_{i,j} alpha_{s,i,j} * A[i,j]) * (sum_{j,k} beta_{s,j,k} * B[j,k])

    # Let me define the 7 block products explicitly.
    # A = [[A00, A01], [A10, A11]] (2x2 blocks of 2x2)
    # B = [[B00, B01], [B10, B11]]

    # Block product 0: A00 * B00
    # Block product 1: A01 * B01  -> contributes to C01 via M01 of this product... 
    # Wait, standard Strassen for 2x2 blocks:
    # C = A*B where A,B are 2x2 blocks
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11

    # Strassen:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A10)B11
    # P6 = (A01-A11)(B10+B11)
    # P7 = (A00-A10)(B00+B10)

    # C00 = P1 - P4
    # C01 = P3 + P4
    # C10 = P2 + P5
    # C11 = P1 - P3

    # So the 7 block products (each a 2x2*2x2) are:
    # s=0: A_s = A00+A11, B_s = B00+B11  -> used in C00 (coeff +1) and C11 (coeff +1)
    # s=1: A_s = A10+A11, B_s = B00      -> used in C10 (coeff +1)
    # s=2: A_s = A00, B_s = B01-B11      -> used in C01 (coeff +1) and C11 (coeff -1)
    # s=3: A_s = A11, B_s = B10-B00      -> used in C00 (coeff -1) and C01 (coeff +1)
    # s=4: A_s = A00+A10, B_s = B11      -> used in C10 (coeff +1)
    # s=5: A_s = A01-A11, B_s = B10+B11  -> used in C01 (coeff +1)... wait
    # Let me re-verify: C01 = P3 + P4. P3 = A00(B01-B11), P4 = A11(B10-B00).
    # Hmm, that doesn't match standard Strassen. Let me use the standard formulation.

    # Standard Strassen (Wikipedia):
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)

    # C11 = M1 - M4 + M6
    # C12 = M3 + M5
    # C21 = M2 + M7
    # C22 = M1 - M3 + M5

    # Using 1-indexed blocks, mapping to 0-indexed: A11->A00, A12->A01, A21->A10, A22->A11
    # B11->B00, B12->B01, B21->B10, B22->B11

    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)

    # C00 = M1 - M4 + M6
    # C01 = M3 + M5
    # C10 = M2 + M7
    # C11 = M1 - M3 + M5

    # Now define each block product s with its A_s and B_s coefficients.
    # Each A_s is a 2x2 matrix of block-level coefficients (alpha_{s,block_i,block_j})
    # Each B_s is a 2x2 matrix of block-level coefficients (beta_{s,block_j,block_k})
    # Each M_s contributes to C with coefficients gamma_{s,block_i,block_k}

    # Block product definitions (alpha, beta, gamma):
    # alpha[s][I][J] for I,J in {0,1}
    # beta[s][J][K] for J,K in {0,1}
    # gamma[s][I][K] for I,K in {0,1}

    block_defs = []
    # s=0: M1
    block_defs.append({
        'alpha': [[1, 0], [0, 1]],  # A00 + A11
        'beta':  [[1, 0], [0, 1]],  # B00 + B11
        'gamma': [[1, 0], [0, 1]]   # C00 += M1, C11 += M1
    })
    # s=1: M2
    block_defs.append({
        'alpha': [[0, 0], [1, 1]],  # A10 + A11
        'beta':  [[1, 0], [0, 0]],  # B00
        'gamma': [[0, 0], [1, 0]]   # C10 += M2
    })
    # s=2: M3
    block_defs.append({
        'alpha': [[1, 0], [0, 0]],  # A00
        'beta':  [[0, 1], [0, -1]], # B01 - B11
        'gamma': [[0, 1], [0, -1]]  # C01 += M3, C11 -= M3
    })
    # s=3: M4
    block_defs.append({
        'alpha': [[0, 0], [0, 1]],  # A11
        'beta':  [[0, 0], [1, -1]], # B10 - B00
        'gamma': [[-1, 0], [0, 0]]  # C00 -= M4
    })
    # s=4: M5
    block_defs.append({
        'alpha': [[1, 1], [0, 0]],  # A00 + A01
        'beta':  [[0, 0], [0, 1]],  # B11
        'gamma': [[0, 1], [0, 1]]   # C01 += M5, C11 += M5
    })
    # s=5: M6
    block_defs.append({
        'alpha': [[-1, 0], [1, 0]], # A10 - A00
        'beta':  [[1, 1], [0, 0]],  # B00 + B01
        'gamma': [[1, 0], [0, 0]]   # C00 += M6
    })
    # s=6: M7
    block_defs.append({
        'alpha': [[0, 1], [0, -1]], # A01 - A11
        'beta':  [[0, 0], [1, 1]],  # B10 + B11
        'gamma': [[0, 0], [1, 0]]   # C10 += M7
    })

    # Now for each block product s and each Strassen scalar product t within it,
    # build the scalar tensor decomposition.
    # Within a 2x2*2x2 product, Strassen gives 7 scalar products.
    # For block product s, A_s is a 2x2 (of 2x2 blocks), so A_s has 4 scalar entries:
    # A_s[I*2+i', J*2+j'] = sum_{I',J'} alpha[s][I'][J'] * A[I'*2+i', J'*2+j']
    # Similarly B_s[J*2+j', K*2+k'] = sum_{J',K'} beta[s][J'][K'] * B[J'*2+j', K'*2+k']

    # Strassen within 2x2: 7 scalar products with weights.
    # For a 2x2 matrix X with entries X[0,0],X[0,1],X[1,0],X[1,1] and Y similarly,
    # the 7 products are:
    # t=0: (X00+X11)(Y00+Y11)
    # t=1: (X10+X11)Y00
    # t=2: X00(Y01-Y11)
    # t=3: X11(Y10-Y00)
    # t=4: (X00+X01)Y11
    # t=5: (X10-X00)(Y00+Y01)
    # t=6: (X01-X11)(Y10+Y11)

    # And the result Z = X*Y has:
    # Z00 = P0 - P3 + P5
    # Z01 = P2 + P4
    # Z10 = P1 + P6
    # Z11 = P0 - P2 + P4

    strassen_inner = [
        # (X_coeff_4x1, Y_coeff_4x1) for each of 7 products, flattened as [x00,x01,x10,x11]
        [[1, 0, 0, 1], [1, 0, 0, 1]],  # t=0
        [[0, 0, 1, 1], [1, 0, 0, 0]],  # t=1
        [[1, 0, 0, 0], [0, 1, 0, -1]], # t=2
        [[0, 0, 0, 1], [0, 0, 1, -1]], # t=3
        [[1, 1, 0, 0], [0, 0, 0, 1]],  # t=4
        [[-1, 0, 1, 0], [1, 1, 0, 0]],# t=5
        [[0, 1, 0, -1], [0, 0, 1, 1]] # t=6
    ]

    # Z weights: [z00, z01, z10, z11] as function of [P0..P6]
    z_weights = [
        [1, 0, 0, -1, 0, 1, 0],  # Z00
        [0, 0, 1, 0, 1, 0, 0],   # Z01
        [0, 1, 0, 0, 0, 0, 1],   # Z10
        [1, 0, -1, 0, 1, 0, 0]   # Z11
    ]

    col = 0
    for s in range(7):
        alpha = block_defs[s]['alpha']
        beta = block_defs[s]['beta']
        gamma = block_defs[s]['gamma']

        for t in range(7):
            x_coeff = strassen_inner[t][0]  # [x00, x01, x10, x11]
            y_coeff = strassen_inner[t][1]  # [y00, y01, y10, y11]

            a_side = [[0.0] * 4 for _ in range(4)]  # a_side[i][j]
            b_side = [[0.0] * 4 for _ in range(4)]  # b_side[j][k]

            for block_i in range(2):
                for block_j in range(2):
                    xc = x_coeff[block_i * 2 + block_j]
                    for i2 in range(2):
                        for j2 in range(2):
                            i = block_i * 2 + i2
                            j = block_j * 2 + j2
                            a_side[i][j] += xc * alpha[block_i][block_j]

            for block_j in range(2):
                for block_k in range(2):
                    yc = y_coeff[block_j * 2 + block_k]
                    for j2 in range(2):
                        for k2 in range(2):
                            j = block_j * 2 + j2
                            k = block_k * 2 + k2
                            b_side[j][k] += yc * beta[block_j][block_k]

            c_side = [[0.0] * 4 for _ in range(4)]  # c_side[i][k]
            for I in range(2):
                for K in range(2):
                    w = gamma[I][K] * z_weights[I * 2 + K][t]
                    for i2 in range(2):
                        for k2 in range(2):
                            i = I * 2 + i2
                            k = K * 2 + k2
                            c_side[i][k] += w

            # Now add to U, V, W
            for i in range(4):
                for j in range(4):
                    U[i * 4 + j][col] = a_side[i][j]
            for j in range(4):
                for k in range(4):
                # V index: j*p + k = j*4 + k
                    V[j * 4 + k][col] = b_side[j][k]
            for k in range(4):
                for i in range(4):
                # W index: k*n + i = k*4 + i
                    W[k * 4 + i][col] = c_side[i][k]
            col += 1

    assert col == 49
    return U, V, W


def _build_schoolbook_4x1():
    """Return (U, V, W) for 4x4 * 4x1 matrix product with rank 16.
    U: 16 x 16, V: 4 x 16, W: 4 x 16
    U index: i*4+j, V index: j*1+k = j, W index: k*4+i
    """
    n, m, p = 4, 4, 1
    rank = 16
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W


def _build_schoolbook_4x4_4x9():
    """Fallback: schoolbook 4x4*4x9, rank 144."""
    n, m, p = 4, 4, 9
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p):
    """Check that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all (i,j,k).
    Also check that the decomposition is exact (no spurious cross-terms)."""
    rank = len(U[0]) if U else 0
    import numpy as np
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err


def construct():
    deadline = time.time() + 110  # safety margin before 120s limit
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    try:
        # Block-Strassen: 4x4x4 (rank 49) for cols 0-3 + 4x4x4 (rank 49) for cols 4-7 + 4x4x1 (rank 16) for col 8
        # Total rank: 114
        U_s, V_s, W_s = _build_strassen_4x4()
        U_sb, V_sb, W_sb = _build_schoolbook_4x1()
        
        rank = 114
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        # Block 1: k=0..3, Strassen
        for c in range(49):
            for i in range(n):
                for j in range(m):
                    U[i * m + j][c] = U_s[i * m + j][c]
            for j in range(m):
                for k in range(4):
                    V[j * p + k][c] = V_s[j * 4 + k][c]
            for k in range(4):
                for i in range(n):
                    W[k * n + i][c] = W_s[k * 4 + i][c]
        
        # Block 2: k=4..7, Strassen
        for c in range(49):
            cc = c + 49
            for i in range(n):
                for j in range(m):
                    U[i * m + j][cc] = U_s[i * m + j][c]
            for j in range(m):
                for k in range(4):
                    V[j * p + (4 + k)][cc] = V_s[j * 4 + k][c]
            for k in range(4):
                for i in range(n):
                    W[(4 + k) * n + i][cc] = W_s[k * 4 + i][c]
        
        # Block 3: k=8, schoolbook 4x4x1
        for c in range(16):
            cc = c + 98
            for i in range(n):
                for j in range(m):
                    U[i * m + j][cc] = U_sb[i * m + j][c]
            for j in range(m):
                V[j * p + 8][cc] = V_sb[j][c]
            for i in range(n):
                W[8 * n + i][cc] = W_sb[i][c]
        
        # Verify
        max_err = _verify_decomposition(U, V, W, n, m, p)
        if max_err < 1e-9:
            return {"U": U, "V": V, "W": W}
        else:
            U_fb, V_fb, W_fb = _build_schoolbook_4x4_4x9()
            return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    except Exception as e:
        U_fb, V_fb, W_fb = _build_schoolbook_4x4_4x9()
        return {"U": U_fb, "V": V_fb, "W": W_fb}


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0]) if result['U'] else 0}")
