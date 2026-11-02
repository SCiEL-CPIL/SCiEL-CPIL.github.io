import time
import numpy as np

DIMENSIONS = [2, 2, 12]

def construct():
    start_time = time.time()
    deadline = start_time + 135

    n, m, p = DIMENSIONS
    num_blocks = p // 2  # 6 blocks
    rank = num_blocks * 7  # 42

    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    # Strassen 7-product decomposition for 2x2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[C00,C01],[C10,C11]]
    # Local indices:
    # A: 0=a, 1=b, 2=c, 3=d
    # B: 0=e, 1=f, 2=g, 3=h
    # C: 0=C00, 1=C01, 2=C10, 3=C11

    strassen_terms = [
        # M1 = (a+d)(e+h)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (c+d)e
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3 = a(f-h)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4 = d(g-e)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5 = (a+b)h
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6 = (c-a)(e+f)
        ({0: -1, 2: 1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (b-d)(g+h)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

    # Precompute global index mappings
    v_map = [[2*t, 2*t+1, m*p//2+2*t, m*p//2+2*t+1] for t in range(num_blocks)]
    w_map = [[(2*t)*n, (2*t+1)*n, (2*t)*n+1, (2*t+1)*n+1] for t in range(num_blocks)]
    
    for t in range(num_blocks):
        for s in range(7):
            col = t * 7 + s
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[s]

            for a_idx, val in u_coeffs.items():
                U[a_idx, col] = val
            for local_b, val in v_coeffs.items():
                V[v_map[t][local_b], col] = val
            for local_c, val in w_coeffs.items():
                W[w_map[t][local_c], col] = val

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
