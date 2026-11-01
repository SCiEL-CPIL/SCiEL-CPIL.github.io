"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
import time

def construct():
    # Start time to ensure we return within deadline if any search was attempted.
    # For this problem, the optimal solution (Strassen, rank 7) is known and returned directly.
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    n, m, p = DIMENSIONS
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen's rank-7 decomposition
    # U: A coefficients, V: B coefficients, W: C coefficients
    # Indices:
    # U: 0=A00, 1=A01, 2=A10, 3=A11
    # V: 0=B00, 1=B01, 2=B10, 3=B11
    # W: 0=C00, 1=C10, 2=C01, 3=C11 (based on W[k*n+i])
    
    strassen = [
        # M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2
        ({2: 1, 3: 1}, {0: 1}, {1: 1, 3: -1}),
        # M3
        ({0: 1}, {1: 1, 3: -1}, {2: 1, 3: 1}),
        # M4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 1: 1}),
        # M5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 2: 1}),
        # M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

    for r, (u_c, v_c, w_c) in enumerate(strassen):
        for idx, val in u_c.items():
            U[idx][r] = val
        for idx, val in v_c.items():
            V[idx][r] = val
        for idx, val in w_c.items():
            W[idx][r] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
