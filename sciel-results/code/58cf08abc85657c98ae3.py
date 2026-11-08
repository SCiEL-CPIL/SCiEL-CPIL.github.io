import time

def construct():
    start = time.time()
    deadline = start + 110  # safety margin before 120s

    n, m, p = 6, 6, 6

    # 2x2 blocking with 3x3 blocks
    # Trivial 3x3 multiplication (rank 9)
    # For each block (I, K), sum over J in 0..1
    # Each term is a 3x3x3 tensor (rank 9)
    # Total rank = 4 * 2 * 9 = 72

    total_rank = 72
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]

    col = 0
    for I in range(2):
        for K in range(2):
            for J in range(2):
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            for l in range(3):
                                r_u = (I * 3 + i) * 6 + (J * 3 + j)
                                r_v = (J * 3 + k) * 6 + (K * 3 + l)
                                r_w = (K * 3 + l) * 6 + (I * 3 + i)
                                U[r_u][col] += 1.0
                                V[r_v][col] += 1.0
                                W[r_w][col] += 1.0
                                col += 1
                                if time.time() > deadline:
                                    break
                            else:
                                continue
                            break
                        else:
                            continue
                        break
                    else:
                        continue
                    break
                else:
                    continue
                break
            else:
                continue
            break
        else:
            continue
        break
    else:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
