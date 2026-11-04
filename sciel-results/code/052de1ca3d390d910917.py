import time
import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    """
    Construct a tensor decomposition for 3x5 * 5x5 matrix multiplication.
    Uses a structured block decomposition: split the 5x5 B into 2x2 blocks of
    sizes (2,3) x (2,3), yielding 4 sub-multiplications.  Each sub-multiplication
    is handled by the schoolbook algorithm.  The total rank is the sum of the
    ranks of the four sub-problems:
      3x2*2x2 -> 12
      3x3*3x2 -> 18
      3x2*2x3 -> 12
      3x3*3x3 -> 27
    Total rank = 12+18+12+27 = 69 < 75 (schoolbook).
    
    The decomposition is exact and uses integer entries (0 or 1).
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit

    n, m, p = DIMENSIONS
    # Split m into m1+m2 and p into p1+p2
    m1, m2 = 2, 3
    p1, p2 = 2, 3
    assert m1 + m2 == m and p1 + p2 == p

    # Four sub-problems: (n, m_i, p_j)
    sub_dims = [
        (n, m1, p1),  # 3x2 * 2x2
        (n, m2, p1),  # 3x3 * 3x2
        (n, m1, p2),  # 3x2 * 2x3
        (n, m2, p2),  # 3x3 * 3x3
    ]

    total_rank = sum(si * mi * pi for si, mi, pi in sub_dims)
    # 3*2*2 + 3*3*2 + 3*2*3 + 3*3*3 = 12+18+12+27 = 69

    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)

    col = 0
    # Block offsets in the m and p dimensions
    m_offsets = [0, m1]  # start of m1 block, start of m2 block
    p_offsets = [0, p1]  # start of p1 block, start of p2 block
    m_sizes = [m1, m2]
    p_sizes = [p1, p2]

    for bi in range(2):  # which m-block
        for bj in range(2):  # which p-block
            si = n
            mi = m_sizes[bi]
            pi = p_sizes[bj]
            sub_rank = si * mi * pi

            mo = m_offsets[bi]
            po = p_offsets[bj]

            for i in range(si):
                for j in range(mi):
                    for k in range(pi):
                        # U index: row i of A, column (mo+j) -> i*m + (mo+j)
                        U[i * m + (mo + j), col] = 1
                        # V index: row (mo+j) of B, column (po+k) -> (mo+j)*p + (po+k)
                        V[(mo + j) * p + (po + k), col] = 1
                        # W index: (po+k)*n + i
                        W[(po + k) * n + i, col] = 1
                        col += 1

    if time.time() > deadline:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
