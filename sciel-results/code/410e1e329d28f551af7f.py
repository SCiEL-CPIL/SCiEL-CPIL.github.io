def synthesize(matrix, depth_limit):
    import numpy as np
    import time

    start_time = time.time()
    deadline = start_time + 85

    matrix = np.array(matrix, dtype=np.uint8)
    m, n = matrix.shape

    if m == 0:
        return {'gates': [], 'outputs': []}

    # Compute RREF and track row operations: T @ M = RREF
    A = matrix.copy()
    T = np.eye(m, dtype=np.uint8)
    pivot_cols = []

    for col in range(n):
        if time.time() > deadline:
            break
        pivot_row = None
        for row in range(len(pivot_cols), m):
            if A[row, col] == 1:
                pivot_row = row
                break
        if pivot_row is None:
            continue
        A[[len(pivot_cols), pivot_row]] = A[[pivot_row, len(pivot_cols)]]
        T[[len(pivot_cols), pivot_row]] = T[[pivot_row, len(pivot_cols)]]
        for row in range(m):
            if row != len(pivot_cols) and A[row, col] == 1:
                A[row] ^= A[len(pivot_cols)]
                T[row] ^= T[len(pivot_cols)]
        pivot_cols.append(col)

    rank = len(pivot_cols)
    rref = A[:rank].copy()

    # Compute T_inv so that M = T_inv @ RREF
    T_inv = np.eye(m, dtype=np.uint8)
    T_work = T.copy()
    for col in range(m):
        if time.time() > deadline:
            break
        pivot_row = None
        for row in range(col, m):
            if T_work[row, col] == 1:
                pivot_row = row
                break
        if pivot_row is None:
            continue
        T_work[[col, pivot_row]] = T_work[[pivot_row, col]]
        T_inv[[col, pivot_row]] = T_inv[[pivot_row, col]]
        for row in range(m):
            if row != col and T_work[row, col] == 1:
                T_work[row] ^= T_work[col]
                T_inv[row] ^= T_inv[col]

    gates = []
    next_wire = n + 1

    # Compute RREF row wires
    rref_wires = []
    for i in range(rank):
        cols = [c for c in range(n) if rref[i, c] == 1]
        if len(cols) == 0:
            rref_wires.append(n)
        elif len(cols) == 1:
            rref_wires.append(cols[0])
        else:
            cur = cols[0]
            for c in cols[1:]:
                gates.append([cur, c])
                cur = next_wire
                next_wire += 1
            rref_wires.append(cur)

    # Compute original outputs as linear combinations of RREF row wires
    outputs = []
    for j in range(m):
        deps = [i for i in range(rank) if T_inv[j, i] == 1]
        if len(deps) == 0:
            outputs.append(n)
        elif len(deps) == 1:
            outputs.append(rref_wires[deps[0]])
        else:
            cur = rref_wires[deps[0]]
            for i in deps[1:]:
                gates.append([cur, rref_wires[i]])
                cur = next_wire
                next_wire += 1
            outputs.append(cur)

    return {'gates': gates, 'outputs': outputs}
