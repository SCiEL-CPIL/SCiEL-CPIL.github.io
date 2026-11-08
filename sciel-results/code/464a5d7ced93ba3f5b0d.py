"""Matrix multiplication 3x5x5 rank-minimal decomposition."""
import time

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard: must return before 120s wall-clock (evaluator kills at 150s)
    start = time.time()
    deadline = start + 110.0  # 10s safety margin

    n, m, p = 3, 5, 5
    rank_schoolbook = n * m * p  # 75

    # Build schoolbook decomposition
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
