import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = 4, 4, 7

    terms = []

    def add_term(u, v, w):
        terms.append((u, v, w))

    # Schoolbook for 4x4 * 4x4 (cols 0-3)
    for i in range(n):
        for j in range(m):
            u = [0] * (n * m)
            u[i * m + j] = 1
            v = [0] * (m * p)
            v[j * p + j] = 1  # Wait, B is 4x7. Cols 0-3 correspond to B[j, 0..3].
            # In schoolbook, C[i, k] = sum_j A[i, j] B[j, k].
            # Here k is the column of C.
            # For the head, k is 0..3.
            # So for a fixed i, j, we contribute to C[i, k] for k=0..3?
            # No, schoolbook term is A[i, j] * B[j, k].
            # This contributes to C[i, k].
            # So for each (i, j, k), we have a term.
            # The current loop is i, j. We need to loop k as well?
            # Or is one term A[i, j] * B[j, :]? No, that's not a rank-1 tensor.
            # A rank-1 tensor is u * v * w.
            # u has one non-zero at (i, j).
            # v has one non-zero at (j, k).
            # w has one non-zero at (k, i).
            # So we need a term for each (i, j, k).
            
            pass

    # Correct Schoolbook Generation
    # Head: C[0..3, 0..3] = A * B[:, 0..3]
    for i in range(n):
        for j in range(m):
            for k in range(4): # k is 0..3
                u = [0] * (n * m)
                u[i * m + j] = 1
                v = [0] * (m * p)
                v[j * p + k] = 1
                w = [0] * (p * n)
                w[k * n + i] = 1
                add_term(u, v, w)

    # Tail: C[0..3, 4..6] = A * B[:, 4..6]
    for i in range(n):
        for j in range(m):
            for k in range(4, 7): # k is 4..6
                u = [0] * (n * m)
                u[i * m + j] = 1
                v = [0] * (m * p)
                v[j * p + k] = 1
                w = [0] * (p * n)
                w[k * n + i] = 1
                add_term(u, v, w)

    if time.time() > deadline:
        return {"U": [], "V": [], "W": []}
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u, v, w) in enumerate(terms):
        for r in range(n * m):
            if u[r] != 0:
                U[r][c] = u[r]
        for r in range(m * p):
            if v[r] != 0:
                V[r][c] = v[r]
        for r in range(p * n):
            if w[r] != 0:
                W[r][c] = w[r]
    
    return {"U": U, "V": V, "W": W}
