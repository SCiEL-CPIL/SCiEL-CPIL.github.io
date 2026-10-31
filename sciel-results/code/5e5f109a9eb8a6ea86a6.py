"""Matrix multiplication 4x4x4 using the 49-rank Strassen-based (Bini et al.) decomposition."""
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # 2x2x2 Strassen: 7 scalar multiplications
    # A, B are 2x2; C = A*B
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # alpha[s][u][v]: coefficient of A[u][v] in factor s of Strassen (2x2)
    # s: 0..6, u,v: 0..1
    alpha = [
        [[1, 0], [0, 1]],   # M1: (A00+A11)(B00+B11) -> A part: A00+A11
        [[0, 0], [1, 1]],   # M2: (A10+A11)B00 -> A part: A10+A11
        [[1, 0], [0, 0]],   # M3: A00(B10-B11) -> A part: A00
        [[0, 0], [0, 1]],   # M4: A11(B01-B00) -> A part: A11
        [[1, 1], [0, 0]],   # M5: (A00+A01)B11 -> A part: A00+A01
        [[-1, 0], [1, 0]],  # M6: (A10-A00)(B00+B01) -> A part: A10-A00
        [[0, 1], [0, -1]],  # M7: (A01-A11)(B10+B11) -> A part: A01-A11
    ]

    # beta[t][u][v]: coefficient of B[u][v] in factor t of Strassen (2x2)
    beta = [
        [[1, 0], [0, 1]],   # M1: B00+B11
        [[1, 0], [0, 0]],   # M2: B00
        [[0, 1], [0, -1]],  # M3: B10-B11
        [[-1, 1], [0, 0]],  # M4: B01-B00
        [[0, 0], [0, 1]],   # M5: B11
        [[1, 1], [0, 0]],   # M6: B00+B01
        [[0, 1], [0, 1]],   # M7: B10+B11
    ]

    # omega[t][u][v]: coefficient for C[u][v] from product t
    omega = [
        [[1, 0], [0, 1]],   # M1: C00 += 1, C11 += 1
        [[0, 0], [-1, 0]],  # M2: C10 += -1? Let's recheck.
        # Actually from above:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        # So:
        # omega[0] (M1): C00+=1, C11+=1
        # omega[1] (M2): C10+=1, C11+=-1
        # omega[2] (M3): C01+=1, C11+=1
        # omega[3] (M4): C00+=1, C10+=1
        # omega[4] (M5): C00+=-1, C01+=1
        # omega[5] (M6): C11+=1
        # omega[6] (M7): C00+=1
        [[1, 0], [0, 1]],   # M1
        [[0, 0], [1, -1]],  # M2
        [[0, 1], [0, 1]],   # M3
        [[1, 0], [1, 0]],   # M4
        [[-1, 1], [0, 0]],  # M5
        [[0, 0], [0, 1]],   # M6
        [[1, 0], [0, 0]],   # M7
    ]

    # The 49-rank decomposition:
    # We partition the 4x4 matrices into 2x2 blocks.
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bij is 2x2
    # C = A*B, so C00 = A00*B00 + A01*B10, C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10, C11 = A10*B01 + A11*B11
    #
    # For each 2x2 block product (e.g., A00*B00), we apply the 7-term Strassen.
    # But that would be 4*7 = 28 terms. Where do we get 49?
    #
    # The 49-rank algorithm is a 7x7 tensor product of the 2x2x2 Strassen.
    # Actually, the standard way to get 49 is to use the 7-rank 2x2x2 and
    # apply it to each of the 7 "super" blocks formed by combining the 4x4
    # into 2x2 of 2x2 blocks, then using a 7x7 grid of interactions.
    #
    # More precisely: the 49-rank algorithm treats the 4x4 multiplication as
    # a 2x2 block multiplication where each block is 2x2, and uses the 7-term
    # Strassen at the block level (giving 7 terms for the 4x4 structure),
    # and then for each of those 7 block-level products, uses the 7-term
    # Strassen at the 2x2 level. Total: 7*7 = 49.
    #
    # So we have two levels of Strassen:
    # Level 1 (block level): 7 terms, each term is a product of two 2x2 matrices (formed from 4x4 blocks)
    # Level 2 (element level): 7 terms per level-1 term
    #
    # Let's define the level-1 (block) Strassen factors:
    # Block indices: 00, 01, 10, 11
    # M1_block = (A00+A11)(B00+B11)
    # M2_block = (A10+A11)B00
    # M3_block = A00(B10-B11)
    # M4_block = A11(B01-B00)
    # M5_block = (A00+A01)B11
    # M6_block = (A10-A00)(B00+B01)
    # M7_block = (A01-A11)(B10+B11)
    #
    # C00_block = M1_block + M4_block - M5_block + M7_block
    # C01_block = M3_block + M5_block
    # C10_block = M2_block + M4_block
    # C11_block = M1_block - M2_block + M3_block + M6_block
    #
    # Each M_s_block is a 2x2 matrix product. We decompose each using the 7-term 2x2 Strassen.
    # So for block term s and element term t, we have a combined term (s,t).
    # Total 49 terms.
    #
    # For term (s, t):
    # U: For each 2x2 block position (I,J) in the 4x4 matrix A, and for each 2x2 position (u,v) within that block:
    #   The coefficient is: gamma_s[I][J] * alpha_t[u][v]
    #   where gamma_s[I][J] is the block-level coefficient for A[I][J] in Strassen term s
    #
    # Similarly for V with delta_s[J][K] * beta_t[u][v]
    #
    # And for W: eta_s[I][K] * omega_t[u][v]
    #
    # Let's define gamma, delta, eta for the block-level Strassen:
    # gamma_s[I][J]: coefficient of block A[I][J] in block Strassen term s
    gamma = [
        [[1, 0], [0, 1]],   # M1: A00 + A11
        [[0, 0], [1, 1]],   # M2: A10 + A11
        [[1, 0], [0, 0]],   # M3: A00
        [[0, 0], [0, 1]],   # M4: A11
        [[1, 1], [0, 0]],   # M5: A00 + A01
        [[-1, 0], [1, 0]],  # M6: A10 - A00
        [[0, 1], [0, -1]],  # M7: A01 - A11
    ]

    delta = [
        [[1, 0], [0, 1]],   # M1: B00 + B11
        [[1, 0], [0, 0]],   # M2: B00
        [[0, 1], [0, -1]],  # M3: B10 - B11
        [[-1, 1], [0, 0]],  # M4: B01 - B00
        [[0, 0], [0, 1]],   # M5: B11
        [[1, 1], [0, 0]],   # M6: B00 + B01
        [[0, 1], [0, 1]],   # M7: B10 + B11
    ]

    eta = [
        [[1, 0], [0, 1]],   # M1: C00 + C11
        [[0, 0], [1, -1]],  # M2: C10 - C11? Wait, let me recheck.
        # C00_block = M1 + M4 - M5 + M7
        # C01_block = M3 + M5
        # C10_block = M2 + M4
        # C11_block = M1 - M2 + M3 + M6
        # So:
        # eta[0] (M1): C00 += 1, C11 += 1
        # eta[1] (M2): C10 += 1, C11 += -1
        # eta[2] (M3): C01 += 1, C11 += 1
        # eta[3] (M4): C00 += 1, C10 += 1
        # eta[4] (M5): C00 += -1, C01 += 1
        # eta[5] (M6): C11 += 1
        # eta[6] (M7): C00 += 1
        [[1, 0], [0, 1]],   # M1
        [[0, 0], [1, -1]],  # M2
        [[0, 1], [0, 1]],   # M3
        [[1, 0], [1, 0]],   # M4
        [[-1, 1], [0, 0]],  # M5
        [[0, 0], [0, 1]],   # M6
        [[1, 0], [0, 0]],   # M7
    ]

    for s in range(7):
        for t in range(7):
            r = s * 7 + t
            # Fill U
            for I in range(2):
                for J in range(2):
                    g = gamma[s][I][J]
                    if g == 0:
                        continue
                    for u in range(2):
                        for v in range(2):
                            a = alpha[t][u][v]
                            if a == 0:
                                continue
                            i = 2 * I + u
                            j = 2 * J + v
                            U[i * m + j, r] = g * a
            # Fill V
            for J in range(2):
                for K in range(2):
                    d = delta[s][J][K]
                    if d == 0:
                        continue
                    for u in range(2):
                        for v in range(2):
                            b = beta[t][u][v]
                            if b == 0:
                                continue
                            j = 2 * J + u
                            k = 2 * K + v
                            V[j * p + k, r] = d * b
            # Fill W
            for I in range(2):
                for K in range(2):
                    e = eta[s][I][K]
                    if e == 0:
                        continue
                    for u in range(2):
                        for v in range(2):
                            w = omega[t][u][v]
                            if w == 0:
                                continue
                            i = 2 * I + u
                            k = 2 * K + v
                            W[k * n + i, r] += e * w

    # Convert to lists for the required return format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    # Ensure integer entries (they should be, but let's be safe)
    U_int = [[int(round(x)) for x in row] for row in U_list]
    V_int = [[int(round(x)) for x in row] for row in V_list]
    W_int = [[int(round(x)) for x in row] for row in W_list]

    return {"U": U_int, "V": V_int, "W": W_int}
# EVOLVE-BLOCK-END
