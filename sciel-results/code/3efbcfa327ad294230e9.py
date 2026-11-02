"""Deterministic algebraic construction for 4x4 by 4x7 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    # Schoolbook construction: rank = n*m*p = 112
    # Each term corresponds to one (i,j,k) triple:
    #   C[i,k] += A[i,j] * B[j,k]
    # In tensor form: T[u,v,w] where u=i*m+j, v=j*p+k, w=k*n+i
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                U[u][col] = 1
                V[v][col] = 1
                W[w][col] = 1
                col += 1

    result = {"U": U, "V": V, "W": W}

    # Verify correctness if time permits
    if time.time() < deadline - 10:
        target = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target.add((i * m + j, j * p + k, k * n + i))

        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)

        # Check all target entries equal 1
        for (u, v, w) in target:
            val = float(np.dot(U_np[u] * V_np[v], W_np[w]))
            if abs(val - 1.0) > 1e-9:
                raise ValueError(f"Target entry ({u},{v},{w}) = {val}, expected 1")

        # Check all non-target entries equal 0 (sampled for speed)
        np.random.seed(42)
        for _ in range(5000):
            u = np.random.randint(0, n * m)
            v = np.random.randint(0, m * p)
            w = np.random.randint(0, p * n)
            if (u, v, w) in target:
                continue
            val = float(np.dot(U_np[u] * V_np[v], W_np[w]))
            if abs(val) > 1e-9:
                raise ValueError(f"Non-target entry ({u},{v},{w}) = {val}, expected 0")

    return result
# EVOLVE-BLOCK-END
