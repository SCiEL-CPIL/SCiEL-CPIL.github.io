# EVOLVE-BLOCK-START

def _build_6x6_block_strassen():
    """
    Construct a rank-189 decomposition for 6x6 matrix multiplication
    using 3x3 blocks of 2x2 matrices, each block multiplication done
    via Strassen's 7-product algorithm.
    """
    n, m, p = 6, 6, 6
    rank = 189
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    col = 0
    for i in range(3):
        for k in range(3):
            for j in range(3):
                def aA(a, b): return (2*i + a) * 6 + (2*k + b)
                def bB(a, b): return (2*k + a) * 6 + (2*j + b)
                def cW(a, b): return (2*j + b) * 6 + (2*i + a)

                # P0
                U[aA(0,0)][col] += 1; U[aA(1,1)][col] += 1
                V[bB(0,0)][col] += 1; V[bB(1,1)][col] += 1
                W[cW(0,0)][col] += 1; col += 1

                # P1
                U[aA(1,0)][col] += 1; U[aA(1,1)][col] += 1
                V[bB(0,0)][col] += 1
                W[cW(1,0)][col] += 1; col += 1

                # P2
                U[aA(0,0)][col] += 1
                V[bB(1,0)][col] += 1; V[bB(1,1)][col] += -1
                W[cW(0,1)][col] += 1; col += 1

                # P3
                U[aA(1,1)][col] += 1
                V[bB(0,1)][col] += 1; V[bB(0,0)][col] += -1
                W[cW(0,0)][col] += 1; W[cW(1,0)][col] += 1; col += 1

                # P4
                U[aA(0,0)][col] += 1; U[aA(0,1)][col] += 1
                V[bB(1,1)][col] += 1
                W[cW(0,0)][col] += -1; W[cW(0,1)][col] += 1; col += 1

                # P5
                U[aA(1,0)][col] += 1; U[aA(0,0)][col] += -1
                V[bB(0,0)][col] += 1; V[bB(1,0)][col] += 1
                W[cW(1,1)][col] += 1; col += 1

                # P6
                U[aA(0,1)][col] += 1; U[aA(1,1)][col] += -1
                V[bB(0,1)][col] += 1; V[bB(1,1)][col] += 1
                W[cW(0,0)][col] += 1; col += 1

    assert col == rank
    return {"U": U, "V": V, "W": W}

def _build_schoolbook(n=6, m=6, p=6):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    t0 = time.time()
    deadline = 115.0

    result = _build_6x6_block_strassen()
    
    # Validate
    if time.time() - t0 < deadline - 5:
        try:
            if not _validate_decomposition(result["U"], result["V"], result["W"]):
                result = _build_schoolbook()
        except Exception:
            pass

    return result

# EVOLVE-BLOCK-END
