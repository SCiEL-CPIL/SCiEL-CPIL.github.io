import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs an exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    
    Uses block-wise partitioning of the middle dimension m=5 into m1=3 and m2=2.
    The total rank is rank(3x3x7) + rank(3x2x7).
    
    For each sub-block, we use the schoolbook decomposition.
    Total rank = 3*3*7 + 3*2*7 = 63 + 42 = 105 (same as full schoolbook,
    but structurally different and verified via block construction).
    
    This approach can be extended: if better decompositions exist for sub-blocks,
    they can be substituted here to reduce total rank.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS
    # Split middle dimension m = m1 + m2
    m1 = 3
    m2 = m - m1  # 2

    # Schoolbook ranks for sub-blocks
    rank1 = n * m1 * p  # 3*3*7 = 63
    rank2 = n * m2 * p  # 3*2*7 = 42
    total_rank = rank1 + rank2  # 105

    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)

    # Block 1: rows 0..m1-1 of A, cols 0..m1-1 of B
    # U rows: i*m + j for i in [0,n), j in [0,m1)
    # V rows: j*p + k for j in [0,m1), k in [0,p)
    # W rows: k*n + i for k in [0,p), i in [0,n)
    i1, j1, k1 = np.indices((n, m1, p)).reshape(3, -1)
    i1f, j1f, k1f = i1.ravel(), j1.ravel(), k1.ravel()
    cols1 = np.arange(rank1)
    U[i1f * m + j1f, cols1] = 1
    V[j1f * p + k1f, cols1] = 1
    W[k1f * n + i1f, cols1] = 1

    # Block 2: rows m1..m-1 of A, cols m1..m-1 of B
    # U rows: i*m + (m1+j) for i in [0,n), j in [0,m2)
    # V rows: (m1+j)*p + k for j in [0,m2), k in [0,p)
    # W rows: k*n + i for k in [0,p), i in [0,n)
    i2, j2, k2 = np.indices((n, m2, p)).reshape(3, -1)
    i2f, j2f, k2f = i2.ravel(), j2.ravel(), k2.ravel()
    cols2 = np.arange(rank1, rank1 + rank2)
    U[i2f * m + (m1 + j2f), cols2] = 1
    V[(m1 + j2f) * p + k2f, cols2] = 1
    W[k2f * n + i2f, cols2] = 1

    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
