"""Rewritten candidate: uses a different construction pipeline (list-of-tuples then zip)
to produce the schoolbook rank-48 decomposition for 2x4x6 matrix multiplication."""

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p

    # Build the tensor as a list of (i, j, k) tuples, one per rank component.
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]

    # From the triples, build U, V, W as lists of lists.
    # U has shape (n*m, rank): row i*m+j is 1 in column (i,j,k) index.
    # V has shape (m*p, rank): row j*p+k is 1 in column (i,j,k) index.
    # W has shape (p*n, rank): row k*n+i is 1 in column (i,j,k) index.
    U_rows = [[] for _ in range(n * m)]
    V_rows = [[] for _ in range(m * p)]
    W_rows = [[] for _ in range(p * n)]

    for (i, j, k) in triples:
        u_row = i * m + j
        v_row = j * p + k
        w_row = k * n + i
        U_rows[u_row].append(1)
        V_rows[v_row].append(1)
        W_rows[w_row].append(1)

    # Each row must have exactly rank entries; pad with zeros if needed (defensive).
    U = [row + [0] * (rank - len(row)) for row in U_rows]
    V = [row + [0] * (rank - len(row)) for row in V_rows]
    W = [row + [0] * (rank - len(row)) for row in W_rows]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
