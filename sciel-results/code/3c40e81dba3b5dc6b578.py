import numpy as np

def _build_conflict_matrix():
    n = 256
    A = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        a = i + 1
        for j in range(i + 1, n):
            b = j + 1
            if (a * b) % (a + b) == 0:
                A[i, j] = 1
                A[j, i] = 1
    return A

def construct():
    n = 256
    A = _build_conflict_matrix()
    w, v = np.linalg.eigh(A)
    # Use eigenvector of smallest eigenvalue for low-conflict core identification
    idx = int(np.argmin(w))
    vec = np.abs(v[:, idx])
    # Highest eigenvector values = low-conflict core
    order = np.argsort(vec)[::-1]
    selected = []
    in_set = np.zeros(n, dtype=bool)
    for i in order:
        if in_set[i]:
            continue
        if not np.any(A[i] & in_set):
            selected.append(i + 1)
            in_set[i] = True
    if not selected:
        selected = [1]
    return selected
