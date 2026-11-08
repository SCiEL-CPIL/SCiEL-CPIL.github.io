import time

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    n, m, p = DIMENSIONS
    rank = 88

    u_rows = n * m  # 12
    v_rows = m * p  # 32
    w_rows = p * n  # 24

    U_flat = [0] * (u_rows * rank)
    V_flat = [0] * (v_rows * rank)
    W_flat = [0] * (w_rows * rank)

    col = 0

    # Part 1: i=0, schoolbook 1x4x8 = 32 terms
    for j in range(m):
        for k in range(p):
            if time.time() > deadline:
                break
            U_flat[(0 * m + j) * rank + col] = 1
            V_flat[(j * p + k) * rank + col] = 1
            W_flat[(k * n + 0) * rank + col] = 1
            col += 1

    if time.time() > deadline:
        pass
    else:
        # Part 2: i=1,2 using Strassen 2x2x2 blocks
        strassen_U = [
            (1, 0, 0, 1),
            (0, 0, 1, 1),
            (1, 0, 0, 0),
            (0, 0, 0, 1),
            (1, 1, 0, 0),
            (-1, 0, 1, 0),
            (0, 1, 0, -1),
        ]
        strassen_V = [
            (1, 0, 0, 1),
            (1, 0, 0, 0),
            (0, 1, 0, -1),
            (-1, 0, 1, 0),
            (0, 0, 0, 1),
            (1, 1, 0, 0),
            (0, 0, 1, 1),
        ]
        strassen_W = [
            (1, 0, 0, 1),
            (0, 0, 1, -1),
            (0, 1, 0, 1),
            (1, 0, 1, 0),
            (-1, 1, 0, 0),
            (0, 0, 0, 1),
            (1, 0, 0, 0),
        ]

        for j_block in range(m // 2):
            for k_block in range(p // 2):
                for t in range(7):
                    if time.time() > deadline:
                        break

                    for li in range(2):
                        for lj in range(2):
                            loc = li * 2 + lj
                            val = strassen_U[t][loc]
                            if val:
                                gi = 1 + li
                                gj = j_block * 2 + lj
                                U_flat[(gi * m + gj) * rank + col] += val

                    for lj in range(2):
                        for lk in range(2):
                            loc = lj * 2 + lk
                            val = strassen_V[t][loc]
                            if val:
                                gj = j_block * 2 + lj
                                gk = k_block * 2 + lk
                                V_flat[(gj * p + gk) * rank + col] += val

                    for li in range(2):
                        for lk in range(2):
                            loc = li * 2 + lk
                            val = strassen_W[t][loc]
                            if val:
                                gi = 1 + li
                                gk = k_block * 2 + lk
                                W_flat[(gk * n + gi) * rank + col] += val

                    col += 1

    U = [U_flat[r * rank:(r + 1) * rank] for r in range(u_rows)]
    V = [V_flat[r * rank:(r + 1) * rank] for r in range(v_rows)]
    W = [W_flat[r * rank:(r + 1) * rank] for r in range(w_rows)]

    return {"U": U, "V": V, "W": W}
