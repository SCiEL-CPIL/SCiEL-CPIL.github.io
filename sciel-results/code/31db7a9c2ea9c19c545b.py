"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose the 2x10 matrix B into 5 blocks of 2x2.
    # Each 2x2 multiplication uses Strassen's algorithm with rank 7.
    # Total rank = 5 * 7 = 35.
    num_blocks = p // 2
    rank = num_blocks * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen coefficients for one 2x2 block
    # M1 = a00 * (b10 + b11)
    # M2 = (a00 + a01) * b10
    # M3 = (a10 + a11) * b00
    # M4 = a11 * (b00 + b01)
    # M5 = (a00 + a11) * (b10 + b11)
    # M6 = (a01 - a11) * (b10 + b11)
    # M7 = (a10 + a00) * (b00 - b10)
    
    # Output reconstruction:
    # c00 = M1 + M4 - M2 + M5
    # c01 = -M1 + M4 + M5 + M6
    # c10 = -M3 + M5 - M6 + M7
    # c11 = M3 - M5 + M7

    for b in range(num_blocks):
        base_col = b * 7
        # Block b corresponds to columns 2*b and 2*b+1 of B
        # and rows 2*b and 2*b+1 of C
        
        # Indices in A: a00=0, a01=1, a10=2, a11=3
        # Indices in B block: b00=2*b, b01=2*b+1, b10=2*b+2, b11=2*b+3 (in V, row index is j*p+k)
        # Wait, V index is j*p + k.
        # j=0, k=2*b -> 2*b
        # j=0, k=2*b+1 -> 2*b+1
        # j=1, k=2*b -> p + 2*b
        # j=1, k=2*b+1 -> p + 2*b + 1
        
        # Indices in C block: c00, c01, c10, c11
        # W index is k*n + i.
        # k=2*b, i=0 -> 2*b
        # k=2*b+1, i=0 -> 2*b+1
        # k=2*b, i=1 -> 2*b + n
        # k=2*b+1, i=1 -> 2*b + 1 + n

        # M1
        col = base_col
        U[0][col] += 1 # a00
        V[2*b][col] += 1 # b00? No, b10. j=1, k=2*b -> p + 2*b
        V[p + 2*b][col] += 1 # b10
        V[p + 2*b + 1][col] += 1 # b11
        W[2*b][col] += 1 # c10? No, M1 contributes to c00 and c10?
        # c00 = ... + M1
        # c10 = ... + M1? No.
        # Let's map W indices carefully.
        # c00 is at W[2*b, 0] ? No, W[k*n+i]. k=2*b, i=0 -> 2*b.
        # c01 is at W[2*b+1, 0] -> 2*b+1.
        # c10 is at W[2*b, 1] -> 2*b + n.
        # c11 is at W[2*b+1, 1] -> 2*b + 1 + n.

        # M1 = a00(b10+b11)
        # c00 += M1, c10 += M1? No.
        # c00 = a00b00 + a01b10. M1 has a00b10.
        # c10 = a10b00 + a11b10. M1 has a00b10.
        # So M1 contributes to c00 and c10?
        # c00 = M1 + M4 - M2 + M5
        # c10 = -M3 + M5 - M6 + M7
        # So M1 contributes to c00 only?
        # c00 = ... + M1
        # c10 does not have M1.
        
        # Let's re-verify the Strassen reconstruction.
        # c00 = M1 + M4 - M2 + M5
        # c01 = -M1 + M4 + M5 + M6
        # c10 = -M3 + M5 - M6 + M7
        # c11 = M3 - M5 + M7
        
        # M1 contributes to c00 (+1) and c01 (-1).
        # M2 contributes to c00 (-1).
        # M3 contributes to c10 (-1) and c11 (+1).
        # M4 contributes to c00 (+1) and c01 (+1).
        # M5 contributes to c00 (+1), c01 (+1), c10 (+1), c11 (-1).
        # M6 contributes to c01 (+1), c10 (-1).
        # M7 contributes to c10 (+1), c11 (+1).

        # M1
        U[0][col] = 1
        V[p + 2*b][col] = 1 # b10
        V[p + 2*b + 1][col] = 1 # b11
        W[2*b][col] = 1 # c00
        W[2*b + 1][col] = -1 # c01

        # M2
        col = base_col + 1
        U[0][col] = 1 # a00
        U[1][col] = 1 # a01
        V[p + 2*b][col] = 1 # b10
        W[2*b][col] = -1 # c00

        # M3
        col = base_col + 2
        U[2][col] = 1 # a10
        U[3][col] = 1 # a11
        V[2*b][col] = 1 # b00
        W[2*b + n][col] = -1 # c10
        W[2*b + 1 + n][col] = 1 # c11

        # M4
        col = base_col + 3
        U[3][col] = 1 # a11
        V[2*b][col] = 1 # b00
        V[2*b + 1][col] = 1 # b01
        W[2*b][col] = 1 # c00
        W[2*b + 1][col] = 1 # c01

        # M5
        col = base_col + 4
        U[0][col] = 1 # a00
        U[3][col] = 1 # a11
        V[p + 2*b][col] = 1 # b10
        V[p + 2*b + 1][col] = 1 # b11
        W[2*b][col] = 1 # c00
        W[2*b + 1][col] = 1 # c01
        W[2*b + n][col] = 1 # c10
        W[2*b + 1 + n][col] = -1 # c11

        # M6
        col = base_col + 5
        U[1][col] = 1 # a01
        U[3][col] = -1 # -a11
        V[p + 2*b][col] = 1 # b10
        V[p + 2*b + 1][col] = 1 # b11
        W[2*b + 1][col] = 1 # c01
        W[2*b + n][col] = -1 # c10

        # M7
        col = base_col + 6
        U[2][col] = 1 # a10
        U[0][col] = 1 # a00
        V[2*b][col] = 1 # b00
        V[p + 2*b][col] = -1 # -b10
        W[2*b + n][col] = 1 # c10
        W[2*b + 1 + n][col] = 1 # c11

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
