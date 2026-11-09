"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    _start = time.time()
    _TIME_LIMIT = 140.0  # safety margin before 150s kill

    # Rank 44 decomposition
    # Part 1: A1 (2x2) * B1 (2x8) using 4 Strassen blocks (28 mults)
    # Part 2: A2 (2x1) * B2 (1x8) using outer product (16 mults)
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank for _ in range(p * n)]  # 16 x 44

    col = 0

    # Part 1: Strassen for A1 * B1
    for b in range(4):
        if time.time() - _start > _TIME_LIMIT - 5.0:
            break

        # Block b corresponds to B columns 2b, 2b+1
        b00_idx = 0 * 8 + 2 * b
        b01_idx = 0 * 8 + 2 * b + 1
        b10_idx = 1 * 8 + 2 * b
        b11_idx = 1 * 8 + 2 * b + 1

        # C indices for this block:
        c00_idx = 2 * b * 2 + 0
        c01_idx = (2 * b + 1) * 2 + 0
        c10_idx = 2 * b * 2 + 1
        c11_idx = (2 * b + 1) * 2 + 1

        # A indices for A1:
        a00_idx = 0
        a01_idx = 1
        a10_idx = 3
        a11_idx = 4

        # Strassen M1: (A00+A11)(B00+B11)
        U[a00_idx][col] += 1
        U[a11_idx][col] += 1
        V[b00_idx][col] += 1
        V[b11_idx][col] += 1
        W[c00_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1

        # Strassen M2: (A10+A11)B00
        U[a10_idx][col] += 1
        U[a11_idx][col] += 1
        V[b00_idx][col] += 1
        W[c10_idx][col] += 1
        W[c11_idx][col] -= 1
        col += 1

        # Strassen M3: A00(B01-B11)
        U[a00_idx][col] += 1
        V[b01_idx][col] += 1
        V[b11_idx][col] -= 1
        W[c01_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1

        # Strassen M4: A11(B10-B00)
        U[a11_idx][col] += 1
        V[b10_idx][col] += 1
        V[b00_idx][col] -= 1
        W[c00_idx][col] += 1
        W[c10_idx][col] += 1
        col += 1

        # Strassen M5: (A00+A01)B11
        U[a00_idx][col] += 1
        U[a01_idx][col] += 1
        V[b11_idx][col] += 1
        W[c00_idx][col] -= 1
        W[c01_idx][col] += 1
        col += 1

        # Strassen M6: (A10-A00)(B00+B01)
        U[a10_idx][col] += 1
        U[a00_idx][col] -= 1
        V[b00_idx][col] += 1
        V[b01_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1

        # Strassen M7: (A01-A11)(B10+B11)
        U[a01_idx][col] += 1
        U[a11_idx][col] -= 1
        V[b10_idx][col] += 1
        V[b11_idx][col] += 1
        W[c00_idx][col] += 1
        col += 1

    # Part 2: Outer product for A2 * B2
    for i in range(2):
        if time.time() - _start > _TIME_LIMIT - 5.0:
            break
        a_idx = i * 3 + 2
        for k in range(8):
            if time.time() - _start > _TIME_LIMIT - 5.0:
                break
            b_idx = 2 * 8 + k
            c_idx = k * 2 + i

            U[a_idx][col] += 1
            V[b_idx][col] += 1
            W[c_idx][col] += 1
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
