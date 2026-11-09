import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Helper: Schoolbook decomposition for n x m x p
    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W

    # Helper: Strassen decomposition for 2 x 2 x 2 (Rank 7)
    def strassen_2x2():
        # U, V, W for 2x2x2
        # Indices: i in 0..1, j in 0..1, k in 0..1
        # U[i*2+j], V[j*2+k], W[k*2+i]
        U = [
            [1, 0, 0, 0, 0, 0, 0], # (0,0)
            [0, 1, 0, 0, 1, 0, 0], # (0,1)
            [0, 0, 1, 1, 0, 1, 0], # (1,0)
            [0, 0, 0, 0, 0, 0, 1]  # (1,1)
        ]
        V = [
            [1, 0, 0, 0, 0, 0, 0], # (0,0)
            [0, 0, 0, 1, 1, 0, 0], # (0,1)
            [0, 1, 1, 0, 0, 1, 0], # (1,0)
            [0, 0, 0, 0, 0, 0, 1]  # (1,1)
        ]
        W = [
            [0, 0, 0, 0, 1, 0, 0], # (0,0)
            [0, 0, 1, 1, 0, 0, 0], # (0,1)
            [0, 1, 1, 0, 0, 0, 0], # (1,0)
            [1, 0, 0, 0, 0, 1, 0]  # (1,1)
        ]
        return U, V, W

    # Combine decompositions
    def combine(decomp_list):
        # decomp_list: list of (U, V, W, i_offset, j_offset, k_offset, n, m, p)
        # i_offset: offset for A rows
        # j_offset: offset for A cols / B rows
        # k_offset: offset for B cols / C cols
        total_rank = sum(len(d[0]) for d in decomp_list)
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col_offset = 0
        for (u, v, w, io, jo, ko, sn, sm, sp) in decomp_list:
            for i in range(sn):
                for j in range(sm):
                    for c in range(len(u[0])):
                        U[(i + io) * m + (j + jo)][col_offset + c] = u[i * sm + j][c]
            for j in range(sm):
                for k in range(sp):
                    for c in range(len(v[0])):
                        V[(j + jo) * p + (k + ko)][col_offset + c] = v[j * sp + k][c]
            for k in range(sp):
                for i in range(sn):
                    for c in range(len(w[0])):
                        W[(k + ko) * n + (i + io)][col_offset + c] = w[k * sn + i][c]
            col_offset += len(u[0])
            
        return U, V, W

    # Decomposition 1: A[0:1, :] * B[2:5, :] (1 x 5 x 5)
    # A rows 0, B rows 2,3,4. C cols 0..4.
    # i_offset=0, j_offset=2, k_offset=0. n=1, m=5, p=5.
    U1, V1, W1 = schoolbook(1, 5, 5)
    decomp1 = (U1, V1, W1, 0, 2, 0, 1, 5, 5)

    # Decomposition 2: A[1:3, 0:2] * B[0:2, 0:2] (2 x 2 x 2)
    # i_offset=1, j_offset=0, k_offset=0. n=2, m=2, p=2.
    U2, V2, W2 = strassen_2x2()
    decomp2 = (U2, V2, W2, 1, 0, 0, 2, 2, 2)

    # Decomposition 3: A[1:3, 0:2] * B[0:2, 2:5] (2 x 2 x 3)
    # i_offset=1, j_offset=0, k_offset=2. n=2, m=2, p=3.
    U3, V3, W3 = schoolbook(2, 2, 3)
    decomp3 = (U3, V3, W3, 1, 0, 2, 2, 2, 3)

    # Decomposition 4: A[1:3, 2:5] * B[2:5, 0:2] (2 x 3 x 2)
    # i_offset=1, j_offset=2, k_offset=0. n=2, m=3, p=2.
    U4, V4, W4 = schoolbook(2, 3, 2)
    decomp4 = (U4, V4, W4, 1, 2, 0, 2, 3, 2)

    # Decomposition 5: A[1:3, 2:5] * B[2:5, 2:5] (2 x 3 x 3)
    # i_offset=1, j_offset=2, k_offset=2. n=2, m=3, p=3.
    U5, V5, W5 = schoolbook(2, 3, 3)
    decomp5 = (U5, V5, W5, 1, 2, 2, 2, 3, 3)

    # Combine all
    U, V, W = combine([decomp1, decomp2, decomp3, decomp4, decomp5])
    
    return {"U": U, "V": V, "W": W}

