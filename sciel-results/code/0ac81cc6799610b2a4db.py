import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    def check_decomposition(U, V, W, R):
        # U: n*m x R, V: m*p x R, W: p*n x R
        # Check: sum_r U[i*m+j, r] * V[l*p+k, r] * W[k*n+i, r] == delta(j,l)
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for l in range(m):
                        s = 0
                        for r in range(R):
                            s += U[i * m + j][r] * V[l * p + k][r] * W[k * n + i][r]
                        target = 1 if j == l else 0
                        if abs(s - target) > 1e-9:
                            return False
        return True

    def try_find_decomposition(R, time_limit=10):
        # Random search with local tweaks
        start_time = time.time()
        # Use small integers: -1, 0, 1
        for attempt in range(50000):
            if time.time() - start_time > time_limit:
                return None

            # Initialize with sparse random matrices
            U = [[0] * R for _ in range(n * m)]
            V = [[0] * R for _ in range(m * p)]
            W = [[0] * R for _ in range(p * n)]

            # Set some random non-zero entries
            for _ in range(R):
                u_idx = random.randint(0, n * m - 1)
                v_idx = random.randint(0, m * p - 1)
                w_idx = random.randint(0, p * n - 1)
                u_val = random.choice([-1, 1])
                v_val = random.choice([-1, 1])
                w_val = random.choice([-1, 1])
                U[u_idx][_] = u_val
                V[v_idx][_] = v_val
                W[w_idx][_] = w_val

            # Additional random non-zero entries
            for _ in range(R // 2):
                u_idx = random.randint(0, n * m - 1)
                v_idx = random.randint(0, m * p - 1)
                w_idx = random.randint(0, p * n - 1)
                u_val = random.choice([-1, 0, 1])
                v_val = random.choice([-1, 0, 1])
                w_val = random.choice([-1, 0, 1])
                U[u_idx][_] = u_val
                V[v_idx][_] = v_val
                W[w_idx][_] = w_val

            if check_decomposition(U, V, W, R):
                return {"U": U, "V": V, "W": W}

        return None

    # Try ranks from 20 to 24
    for R in [20, 21, 22, 23, 24]:
        result = try_find_decomposition(R, time_limit=8)
        if result:
            return result

    # Fallback to schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
