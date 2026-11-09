import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2 decomposition matrices
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)

    rank = 49
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for I in range(2):
        for J in range(2):
            block_idx = I * 2 + J
            for i in range(2):
                for j in range(2):
                    inner_idx = i * 2 + j
                    u_idx = (I * 2 + i) * 4 + (J * 2 + j)
                    v_idx = (I * 2 + i) * 4 + (J * 2 + j)
                    
                    for r in range(7):
                        ub = U2[block_idx, r]
                        vb = V2[block_idx, r]
                        if ub == 0 and vb == 0:
                            continue
                        for s in range(7):
                            c = r * 7 + s
                            ui = U2[inner_idx, s]
                            vi = V2[inner_idx, s]
                            if ui == 0 and vi == 0:
                                continue
                            U[u_idx, c] = ub * ui
                            V[v_idx, c] = vb * vi

    for I in range(2):
        for J in range(2):
            block_idx = J * 2 + I
            for i in range(2):
                for j in range(2):
                    inner_idx = j * 2 + i
                    w_idx = (J * 2 + j) * 4 + (I * 2 + i)
                    
                    for r in range(7):
                        wb = W2[block_idx, r]
                        if wb == 0:
                            continue
                        for s in range(7):
                            c = r * 7 + s
                            wi = W2[inner_idx, s]
                            if wi == 0:
                                continue
                            W[w_idx, c] = wb * wi

    # Prune any zero columns
    keep = [c for c in range(rank) if not (np.all(U[:, c] == 0) or np.all(V[:, c] == 0) or np.all(W[:, c] == 0))]
    if len(keep) < rank:
        U = U[:, keep]
        V = V[:, keep]
        W = W[:, keep]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
