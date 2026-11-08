"""Rank-88 tensor decomposition for 3x4x8 matrix multiplication."""
import time
import numpy as np

N, M, P = 3, 4, 8
RANK = 88

def construct():
    t0 = time.time()
    U = np.zeros((N * M, RANK), dtype=int)
    V = np.zeros((M * P, RANK), dtype=int)
    W = np.zeros((P * N, RANK), dtype=int)
    
    col = 0
    deadline = t0 + 140.0
    
    # Part 1: i=0, schoolbook 1x4x8 (32 terms)
    for j in range(M):
        for k in range(P):
            U[j, col] = 1          # A[0,j] -> index 0*M+j = j
            V[j * P + k, col] = 1  # B[j,k]
            W[k * N + 0, col] = 1  # C[0,k] -> index k*N+0
            col += 1
            if time.time() > deadline:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Part 2: i=1,2 via 8 Strassen 2x2x2 blocks
    # j_blocks: [0,1] and [2,3]; k_blocks: [0,1],[2,3],[4,5],[6,7]
    # Strassen 2x2x2 factors
    su = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ], dtype=int)
    sv = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ], dtype=int)
    sw = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ], dtype=int)
    
    for jb in range(2):
        js = jb * 2
        for kb in range(4):
            ks = kb * 2
            for t in range(7):
                if time.time() > deadline:
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                # U: local (li, lj) -> global i=1+li, j=js+lj
                for li in range(2):
                    for lj in range(2):
                        val = su[t, li * 2 + lj]
                        if val != 0:
                            U[(1 + li) * M + (js + lj), col] += val
                # V: local (lj, lk) -> global j=js+lj, k=ks+lk
                for lj in range(2):
                    for lk in range(2):
                        val = sv[t, lj * 2 + lk]
                        if val != 0:
                            V[(js + lj) * P + (ks + lk), col] += val
                # W: local (li, lk) -> global k=ks+lk, i=1+li
                for li in range(2):
                    for lk in range(2):
                        val = sw[t, li * 2 + lk]
                        if val != 0:
                            W[(ks + lk) * N + (1 + li), col] += val
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
