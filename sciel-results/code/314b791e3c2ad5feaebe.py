import numpy as np

def construct():
    """
    Constructs the factor matrices for the matrix multiplication tensor M_{2,4,4}.
    Uses the schoolbook (trivial) decomposition with rank = n*m*p = 32.
    Each basis term corresponds to a single product A[i,j]*B[j,k] contributing to C[i,k].
    """
    n, m, p = 2, 4, 4
    rank = n * m * p  # 32

    # Generate all (i, j, k) triples
    i_vals = np.repeat(np.arange(n), m * p)
    j_vals = np.tile(np.repeat(np.arange(m), p), n)
    k_vals = np.tile(np.arange(p), n * m)

    # Row indices in the flattened matrices
    u_rows = i_vals * m + j_vals   # index into U (n*m rows)
    v_rows = j_vals * p + k_vals   # index into V (m*p rows)
    w_rows = k_vals * n + i_vals   # index into W (p*n rows), following W[k*n+i, r]

    # Initialize sparse-ish dense matrices
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # Set the diagonal-like entries to 1
    U[u_rows, np.arange(rank)] = 1.0
    V[v_rows, np.arange(rank)] = 1.0
    W[w_rows, np.arange(rank)] = 1.0

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
