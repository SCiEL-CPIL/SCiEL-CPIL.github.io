"""Matrix multiplication 2x2 by 2x9 - rank-32 Strassen-block decomposition."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = 2, 2, 9
    total_rank = 32

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    def _strassen_pair(ka, kb, base):
        ua, va, wa = 0, 0, 0
        ub, vb, wb = 1, 1, 1
        for r in range(7):
            ci = base + r
            if r == 0:
                U[ua][ci] += 1
                U[ub][ci] += 1
                V[va][ci] += 1
                V[vb][ci] += 1
                W[wa][ci] += 1
                W[wb][ci] += 1
            elif r == 1:
                U[ub][ci] += 1
                V[va][ci] += 1
                W[wb][ci] -= 1
                W[wa + 1][ci] += 1
            elif r == 2:
                U[ua][ci] += 1
                V[vb][ci] += 1
                V[va + p][ci] -= 1
                W[wb + 1][ci] += 1
            elif r == 3:
                U[ub][ci] += 1
                V[va + p][ci] += 1
                V[va][ci] -= 1
                W[wa][ci] += 1
                W[wa + 1][ci] += 1
            elif r == 4:
                U[ua][ci] += 1
                V[vb][ci] += 1
                W[wa][ci] -= 1
                W[wb + 1][ci] += 1
            elif r == 5:
                U[ua][ci] += 1
                V[va][ci] += 1
                V[vb][ci] += 1
                W[wb + 1][ci] += 1
            elif r == 6:
                U[ub][ci] += 1
                V[va + p][ci] += 1
                V[vb][ci] += 1
                W[wa][ci] += 1
            else:
                pass

    def _single_col(ka, base):
        for r in range(4):
            ci = base + r
            u_idx = r // 2
            v_idx = (r % 2) * p + ka
            w_idx = ka * n + r // 2
            U[u_idx][ci] += 1
            V[v_idx][ci] += 1
            W[w_idx][ci] += 1

    idx = 0
    for a in range(4):
        ka = 2 * a
        kb = 2 * a + 1
        _strassen_pair(ka, kb, idx)
        idx += 7
    _single_col(8, idx)

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
