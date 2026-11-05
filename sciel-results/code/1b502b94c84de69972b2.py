import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140

    def build_rank_22():
        # ... (keep the existing rank 22 construction) ...
        rank = 22
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_term(u_indices, v_indices, w_indices, col):
            for idx, val in u_indices:
                U[idx][col] += val
            for idx, val in v_indices:
                V[idx][col] += val
            for idx, val in w_indices:
                W[idx][col] += val

        # Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
        add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
        add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
        add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
        add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
        add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
        add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
        add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)

        # Block 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4]
        add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
        add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
        add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
        add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
        add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
        add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
        add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)

        # Outer Product
        add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
        add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
        add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
        add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
        add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
        add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
        add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
        add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)

        return U, V, W, rank

    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for t in range(rank):
                        s += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def to_lists(U, V, W):
        U_list = [[round(float(x), 10) for x in row] for row in U]
        V_list = [[round(float(x), 10) for x in row] for row in V]
        W_list = [[round(float(x), 10) for x in row] for row in W]
        return U_list, V_list, W_list

    def snap_to_half_integer(arr, tol=1e-6):
        snapped = np.zeros_like(arr)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = arr[i, j]
                nearest_int = round(val)
                if abs(val - nearest_int) < tol:
                    snapped[i, j] = nearest_int
                else:
                    nearest_half = round(val * 2) / 2.0
                    if abs(val - nearest_half) < tol:
                        snapped[i, j] = nearest_half
                    else:
                        snapped[i, j] = val
        return snapped

    # Constructive block-partitioning: 2x2x2 (Strassen rank 7) + 2x1x4 (rank 4) + 2x1x4 (rank 4) = rank 15
    def build_rank_15():
        rank = 15
        U = [[0] * rank for _ in range(n * m)]  # 6 rows
        V = [[0] * rank for _ in range(m * p)]  # 12 rows
        W = [[0] * rank for _ in range(p * n)]  # 8 rows

        def add_term(u_indices, v_indices, w_indices, col):
            for idx, val in u_indices:
                U[idx][col] += val
            for idx, val in v_indices:
                V[idx][col] += val
            for idx, val in w_indices:
                W[idx][col] += val

        # Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
        # U indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        # V indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        # W indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        add_term([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)], 0)
        add_term([(2, 1), (3, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
        add_term([(0, 1)], [(1, 1), (3, -1)], [(2, 1), (3, 1)], 2)
        add_term([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (1, 1)], 3)
        add_term([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (2, 1)], 4)
        add_term([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
        add_term([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)], 6)

        # Block 2: A[0:2, 2:3] * B[2:3, 0:4] -> C[0:2, 0:4] (2x1x4, rank 4)
        # U index: 4=(0,2), 5=(1,2)
        # V indices: 8=(2,0), 9=(2,1), 10=(2,2), 11=(2,3)
        # W indices: 0=(0,0), 1=(0,1), 2=(0,2), 3=(0,3), 4=(1,0), 5=(1,1), 6=(1,2), 7=(1,3)
        add_term([(4, 1)], [(8, 1)], [(0, 1)], 7)
        add_term([(4, 1)], [(9, 1)], [(1, 1)], 8)
        add_term([(4, 1)], [(10, 1)], [(2, 1)], 9)
        add_term([(4, 1)], [(11, 1)], [(3, 1)], 10)
        add_term([(5, 1)], [(8, 1)], [(4, 1)], 11)
        add_term([(5, 1)], [(9, 1)], [(5, 1)], 12)
        add_term([(5, 1)], [(10, 1)], [(6, 1)], 13)
        add_term([(5, 1)], [(11, 1)], [(7, 1)], 14)

        return U, V, W, rank

    # Try the constructive rank-15 first
    U15, V15, W15, rank15 = build_rank_15()
    if verify(U15, V15, W15, rank15):
        return {"U": U15, "V": V15, "W": W15}

    # Fallback to rank 22
    U, V, W, rank = build_rank_22()
    return {"U": U, "V": V, "W": W}
