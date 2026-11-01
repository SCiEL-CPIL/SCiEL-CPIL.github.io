"""Matrix multiplication 2,2,14 — rank-minimal exact tensor decomposition."""
import time
import random

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 135  # seconds; leave margin before 150s kill

    n, m, p = 2, 2, 14
    rank = n * m * p  # 56 (schoolbook)

    # Default: schoolbook decomposition
    U = [[0] * rank for _ in range(n * m)]  # 4 x 56
    V = [[0] * rank for _ in range(m * p)]  # 28 x 56
    W = [[0] * rank for _ in range(p * n)]  # 28 x 56
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    # Try to find a better decomposition using algebraic structure
    # For 2x2 x 2x14, we can use Strassen on 2x2 blocks
    # 14 = 7 * 2, so 7 blocks of 2x2
    # Each 2x2 x 2x2 has rank 7 (Strassen)
    # Total rank: 7 * 7 = 49
    
    # Build Strassen decomposition for 2x2 x 2x14
    # A: 2x2, B: 2x14, C: 2x14
    # Split B into 7 blocks of 2x2: B[:, 2b:2b+2]
    # For each block, use Strassen (7 products)
    # Total: 7 blocks * 7 products = 49 rank-1 terms

    # Strassen for 2x2:
    # A = [[a00, a01], [a10, a11]]
    # B_block = [[b00, b01], [b10, b11]]
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11)b00
    # M3 = a00(b01 - b11)
    # M4 = a11(b10 - b00)
    # M5 = (a00 + a01)b11
    # M6 = (a10 - a00)(b00 + b01)
    # M7 = (a01 - a11)(b10 + b11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # U indices: i*m + j = i*2 + j for (i,j) in {(0,0),(0,1),(1,0),(1,1)} -> 0,1,2,3
    # V indices: j*p + k for (j,k) where j in {0,1}, k in {0,...,13}
    # W indices: k*n + i for (k,i) where k in {0,...,13}, i in {0,1}

    # For each block b (0 to 6), local columns are 2b, 2b+1
    # Strassen products with local B columns (0, 1) mapping to global (2b, 2b+1)

    strassen_products = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # M1 = (a00 + a11)(b00 + b11)
        ({(0, 0): 1, (1, 1): 1},
         {(0, 0): 1, (1, 1): 1},
         {(0, 0): 1, (1, 1): 1}),
        # M2 = (a10 + a11)b00
        ({(1, 0): 1, (1, 1): 1},
         {(0, 0): 1},
         {(0, 1): 1, (1, 1): -1}),
        # M3 = a00(b01 - b11)
        ({(0, 0): 1},
         {(0, 1): 1, (1, 1): -1},
         {(0, 0): 1, (1, 0): 1}),
        # M4 = a11(b10 - b00)
        ({(1, 1): 1},
         {(1, 0): 1, (0, 0): -1},
         {(0, 0): 1, (0, 1): 1}),
        # M5 = (a00 + a01)b11
        ({(0, 0): 1, (0, 1): 1},
         {(1, 1): 1},
         {(0, 0): -1, (1, 0): 1}),
        # M6 = (a10 - a00)(b00 + b01)
        ({(1, 0): 1, (0, 0): -1},
         {(0, 0): 1, (0, 1): 1},
         {(1, 1): 1}),
        # M7 = (a01 - a11)(b10 + b11)
        ({(0, 1): 1, (1, 1): -1},
         {(1, 0): 1, (1, 1): 1},
         {(0, 0): 1, (1, 1): 1}),
    ]

    rank_strassen = 7 * 7  # 49
    U_s = [[0] * rank_strassen for _ in range(n * m)]  # 4 x 49
    V_s = [[0] * rank_strassen for _ in range(m * p)]  # 28 x 49
    W_s = [[0] * rank_strassen for _ in range(p * n)]  # 28 x 49

    for block in range(7):
        for s in range(7):
            col = block * 7 + s
            u_coeffs, v_coeffs, w_coeffs = strassen_products[s]

            # U: A indices (i,j) -> U[i*m+j, col]
            for (i, j), coeff in u_coeffs.items():
                U_s[i * m + j][col] = coeff

            # V: B indices (j, k_local) where k_local in {0,1}
            # Global k = block*2 + k_local
            # V row index = j * p + (block*2 + k_local)
            for (j, k_local), coeff in v_coeffs.items():
                k_global = block * 2 + k_local
                V_s[j * p + k_global][col] = coeff

            # W: C indices (k_local, i)
            # Global k = block*2 + k_local
            # W row index = (block*2 + k_local) * n + i
            for (k_local, i), coeff in w_coeffs.items():
                k_global = block * 2 + k_local
                W_s[k_global * n + i][col] = coeff

    # Validate the Strassen decomposition
    if time.time() - start_time < DEADLINE:
        random.seed(42)
        ok = True
        for _ in range(3):
            A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
            B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
            for i in range(n):
                for k in range(p):
                    expected = sum(A[i][j] * B[j][k] for j in range(m))
                    actual = 0
                    for r in range(rank_strassen):
                        u_val = sum(U_s[i * m + j][r] * A[i][j] for j in range(m))
                        v_val = sum(V_s[j * p + k][r] * B[j][k] for j in range(m))
                        actual += W_s[k * n + i][r] * u_val * v_val
                    if abs(actual - expected) > 1e-9:
                        ok = False
                        break
                if not ok:
                    break

        if ok:
            return {"U": U_s, "V": V_s, "W": W_s}

    # Fall back to schoolbook
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
