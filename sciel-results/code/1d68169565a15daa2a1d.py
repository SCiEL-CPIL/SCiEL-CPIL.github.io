"""4x4 matrix multiplication tensor decomposition using Strassen on 2x2 blocks."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for a in range(2):
        for b in range(2):
            for k in range(2):
                for s in range(7):
                    if s == 0:
                        u_entries = [(1, a*2, k*2), (1, a*2+1, k*2+1)]
                        v_entries = [(1, k*2, b*2), (1, k*2+1, b*2+1)]
                        w_entries = [(1, a*2, b*2), (1, a*2+1, b*2+1)]
                    elif s == 1:
                        u_entries = [(1, a*2+1, k*2), (1, a*2+1, k*2+1)]
                        v_entries = [(1, k*2, b*2)]
                        w_entries = [(1, a*2+1, b*2), (-1, a*2+1, b*2+1)]
                    elif s == 2:
                        u_entries = [(1, a*2, k*2)]
                        v_entries = [(1, k*2, b*2+1), (-1, k*2+1, b*2+1)]
                        w_entries = [(1, a*2, b*2+1), (1, a*2+1, b*2+1)]
                    elif s == 3:
                        u_entries = [(1, a*2+1, k*2+1)]
                        v_entries = [(1, k*2+1, b*2), (-1, k*2, b*2)]
                        w_entries = [(1, a*2, b*2), (1, a*2+1, b*2)]
                    elif s == 4:
                        u_entries = [(1, a*2, k*2), (1, a*2, k*2+1)]
                        v_entries = [(1, k*2+1, b*2+1)]
                        w_entries = [(-1, a*2, b*2), (1, a*2, b*2+1)]
                    elif s == 5:
                        u_entries = [(1, a*2+1, k*2), (-1, a*2, k*2)]
                        v_entries = [(1, k*2, b*2), (1, k*2, b*2+1)]
                        w_entries = [(1, a*2+1, b*2+1)]
                    else:
                        u_entries = [(1, a*2, k*2+1), (-1, a*2+1, k*2+1)]
                        v_entries = [(1, k*2+1, b*2), (1, k*2+1, b*2+1)]
                        w_entries = [(1, a*2, b*2)]
                    for (c, i, j) in u_entries:
                        U[i*m+j][col] += c
                    for (c, i, j) in v_entries:
                        V[i*p+j][col] += c
                    for (c, i, j) in w_entries:
                        W[j*n+i][col] += c
                    col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
