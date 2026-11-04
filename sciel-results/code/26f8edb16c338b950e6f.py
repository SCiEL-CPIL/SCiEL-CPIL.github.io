import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit

    def verify_decomposition(U, V, W, rank):
        """Verify that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = 0.0
                    for r in range(rank):
                        actual += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(actual - 1.0) > 1e-9:
                        return False
        return True

    def build_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W, rank

    # Build schoolbook decomposition
    U, V, W, rank = build_schoolbook()

    # Verify correctness
    if not verify_decomposition(U, V, W, rank):
        # This should never happen for schoolbook, but guard against it
        raise RuntimeError("Schoolbook decomposition verification failed")

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
