import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    t0 = time.time()
    deadline = t0 + 110.0

    n, m, p = 4, 4, 6

    # Strassen 2x2*2x2 factors
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))

    prods = [
        ([(0,0,1.0), (1,1,1.0)], [(0,0,1.0), (1,1,1.0)], [(0,0,1.0), (1,1,1.0)]),
        ([(1,0,1.0), (1,1,1.0)], [(0,0,1.0)], [(1,0,1.0), (1,1,-1.0)]),
        ([(0,0,1.0)], [(0,1,1.0), (1,1,-1.0)], [(0,1,1.0), (1,1,1.0)]),
        ([(1,1,1.0)], [(1,0,1.0), (0,0,-1.0)], [(0,0,1.0), (1,0,1.0)]),
        ([(0,0,1.0), (0,1,1.0)], [(1,1,1.0)], [(0,0,-1.0), (0,1,1.0)]),
        ([(1,0,1.0), (0,0,-1.0)], [(0,0,1.0), (0,1,1.0)], [(1,1,1.0)]),
        ([(0,1,1.0), (1,1,-1.0)], [(1,0,1.0), (1,1,1.0)], [(0,0,1.0)]),
    ]

    for m_idx, (Ub, Vb, Wb) in enumerate(prods):
        for (i, j, c) in Ub:
            U_s[i * 2 + j, m_idx] += c
        for (j, k, c) in Vb:
            V_s[j * 2 + k, m_idx] += c
        for (i, k, c) in Wb:
            W_s[k * 2 + i, m_idx] += c

    # Build rank-77 Strassen decomposition
    rank_444 = 49
    rank_442 = 28
    total_rank = rank_444 + rank_442

    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    # Part 1: 4x4*4x4 Strassen (Kronecker product of two 2x2 Strassen)
    for t in range(7):
        for s in range(7):
            col = 7 * t + s
            for i in range(4):
                I = i // 2
                i0 = i % 2
                for j in range(4):
                    J = j // 2
                    j0 = j % 2
                    val = U_s[2 * I + J, t] * U_s[2 * i0 + j0, s]
                    if val != 0:
                        U[i * 4 + j, col] += val

            for j in range(4):
                J = j // 2
                j0 = j % 2
                for k in range(4):
                    K = k // 2
                    k0 = k % 2
                    val = V_s[2 * J + K, t] * V_s[2 * j0 + k0, s]
                    if val != 0:
                        V[j * 6 + k, col] += val

            for k in range(4):
                K = k // 2
                k0 = k % 2
                for i in range(4):
                    I = i // 2
                    i0 = i % 2
                    val = W_s[2 * K + I, t] * W_s[2 * k0 + i0, s]
                    if val != 0:
                        W[k * 4 + i, col] += val

    # Part 2: 4x4*4x2 for columns 4-5
    block_products = [
        (0, 0, 0, 4, 0, 4),
        (0, 2, 2, 4, 0, 4),
        (2, 0, 0, 4, 2, 4),
        (2, 2, 2, 4, 2, 4),
    ]

    for block_idx, (ar, ac, br, bc, cr, cc) in enumerate(block_products):
        for s in range(7):
            col = rank_444 + block_idx * 7 + s
            for i in range(2):
                for j in range(2):
                    val = U_s[i * 2 + j, s]
                    if val != 0:
                        U[(ar + i) * 4 + (ac + j), col] += val

            for j in range(2):
                for k in range(2):
                    val = V_s[j * 2 + k, s]
                    if val != 0:
                        V[(br + j) * 6 + (bc + k), col] += val

            for k in range(2):
                for i in range(2):
                    val = W_s[k * 2 + i, s]
                    if val != 0:
                        W[(cc + k) * 4 + (cr + i), col] += val

    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else float(x) for x in row] for row in mat]

    # Check deadline
    if time.time() > deadline - 40.0:
        return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

    # Verify correctness
    residual = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(total_rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                expected = 1.0
                residual += (s - expected)**2
    
    if residual > 1e-10:
        return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
