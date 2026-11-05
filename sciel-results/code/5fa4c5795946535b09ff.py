import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 3, 4, 8
    nm, mp, pn = n * m, m * p, p * n  # 12, 32, 24

    # Strassen terms for 2x2x2
    # U, V, W as (i,j), (j,k), (k,i)
    strassen_terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1}),
        ({(1,1):1}, {(0,0):-1, (1,0):1}, {(0,0):-1, (1,0):1, (1,1):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):1, (0,1):1}),
        ({(0,0):-1, (1,0):1}, {(0,0):1, (0,1):1}, {(1,0):1, (1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):-1, (0,1):1}),
    ]

    # Schoolbook terms for 1x2x2 (rank 4)
    # C_0 = A_0 B_00 + A_1 B_10
    # C_1 = A_0 B_01 + A_1 B_11
    # U: (0,0), (1,0) -> A_0, A_1
    # V: (0,0), (1,0) -> B_00, B_10
    # W: (0,0) -> C_0
    # etc
    # We can just use 4 rank-1 terms.
    schoolbook_1x2x2 = [
        ({(0,0):1}, {(0,0):1}, {(0,0):1}), # A0 B00 C0
        ({(0,0):1}, {(0,1):1}, {(0,1):1}), # A0 B01 C1
        ({(1,0):1}, {(1,0):1}, {(0,0):1}), # A1 B10 C0
        ({(1,0):1}, {(1,1):1}, {(0,1):1}), # A1 B11 C1
    ]

    # Block definitions
    # n_blocks: [0], [1,2]
    # m_blocks: [0,1], [2,3]
    # p_blocks: [0,1], [2,3], [4,5], [6,7]
    
    n_blocks = [
        (0, 1), # indices [0]
        (1, 2)  # indices [1,2]
    ]
    m_blocks = [
        (0, 2), # indices [0,1]
        (2, 4)  # indices [2,3]
    ]
    p_blocks = [
        (0, 2), # indices [0,1]
        (2, 4), # indices [2,3]
        (4, 6), # indices [4,5]
        (6, 8)  # indices [6,7]
    ]

    total_rank = 0
    # Calculate total rank to allocate matrices
    for n_start, n_end in n_blocks:
        n_size = n_end - n_start
        for m_start, m_end in m_blocks:
            m_size = m_end - m_start
            for p_start, p_end in p_blocks:
                p_size = p_end - p_start
                if n_size == 1 and m_size == 2 and p_size == 2:
                    total_rank += 4
                elif n_size == 2 and m_size == 2 and p_size == 2:
                    total_rank += 7
                else:
                    # Fallback
                    total_rank += n_size * m_size * p_size

    U = np.zeros((nm, total_rank), dtype=np.int64)
    V = np.zeros((mp, total_rank), dtype=np.int64)
    W = np.zeros((pn, total_rank), dtype=np.int64)
    col = 0

    for n_start, n_end in n_blocks:
        n_size = n_end - n_start
        for m_start, m_end in m_blocks:
            m_size = m_end - m_start
            for p_start, p_end in p_blocks:
                p_size = p_end - p_start
                
                if n_size == 1 and m_size == 2 and p_size == 2:
                    terms = schoolbook_1x2x2
                elif n_size == 2 and m_size == 2 and p_size == 2:
                    terms = strassen_terms
                else:
                    # Generate schoolbook terms
                    terms = []
                    for i in range(n_size):
                        for j in range(m_size):
                            for k in range(p_size):
                                terms.append(
                                    ({(i,j):1}, {(j,k):1}, {(k,i):1})
                                )

                for u_d, v_d, w_d in terms:
                    for (il, jl), u_val in u_d.items():
                        i_g = n_start + il
                        j_g = m_start + jl
                        U[i_g * m + j_g, col] += u_val
                    for (jl, kl), v_val in v_d.items():
                        j_g = m_start + jl
                        k_g = p_start + kl
                        V[j_g * p + k_g, col] += v_val
                    for (kl, il), w_val in w_d.items():
                        k_g = p_start + kl
                        i_g = n_start + il
                        W[k_g * n + i_g, col] += w_val
                    col += 1

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
