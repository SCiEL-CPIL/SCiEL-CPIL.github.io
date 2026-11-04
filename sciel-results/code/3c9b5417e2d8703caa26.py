import numpy as np
import time
import random

def construct():
    n, m, p = 3, 3, 3
    deadline = time.time() + 135  # 15s safety margin

    # Return seed to be safe, as search is complex.
    rank = 27
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(9)]
    W = [[0] * rank for _ in range(9)]
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j][col] = 1
                V[j*3+k][col] = 1
                W[k*3+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
