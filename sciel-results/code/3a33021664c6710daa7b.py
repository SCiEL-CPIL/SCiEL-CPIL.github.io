import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = 4, 5, 5
    
    terms = []
    
    def add_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        u_vec = [0] * (n * m)
        v_vec = [0] * (m * p)
        w_vec = [0] * (p * n)
        
        for idx, val in zip(u_indices, u_vals):
            u_vec[idx] += val
        for idx, val in zip(v_indices, v_vals):
            v_vec[idx] += val
        for idx, val in zip(w_indices, w_vals):
            w_vec[idx] += val
            
        terms.append((u_vec, v_vec, w_vec))

    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i

    def add_strassen_2x2(i0, j0, k0):
        add_term(
            [u_idx(i0, j0), u_idx(i0+1, j0+1)], [1, 1],
            [v_idx(j0, k0), v_idx(j0+1, k0+1)], [1, 1],
            [w_idx(k0, i0), w_idx(k0+1, i0+1)], [1, 1]
        )
        add_term(
            [u_idx(i0+1, j0), u_idx(i0+1, j0+1)], [1, 1],
            [v_idx(j0, k0)], [1],
            [w_idx(k0, i0+1)], [1]
        )
        add_term(
            [u_idx(i0, j0)], [1],
            [v_idx(j0, k0+1), v_idx(j0+1, k0+1)], [1, -1],
            [w_idx(k0+1, i0)], [1]
        )
        add_term(
            [u_idx(i0+1, j0+1)], [1],
            [v_idx(j0+1, k0), v_idx(j0, k0)], [1, -1],
            [w_idx(k0, i0), w_idx(k0, i0+1)], [1, 1]
        )
        add_term(
            [u_idx(i0, j0), u_idx(i0, j0+1)], [1, 1],
            [v_idx(j0+1, k0+1)], [1],
            [w_idx(k0+1, i0), w_idx(k0, i0)], [1, -1]
        )
        add_term(
            [u_idx(i0+1, j0), u_idx(i0, j0)], [1, -1],
            [v_idx(j0, k0), v_idx(j0, k0+1)], [1, 1],
            [w_idx(k0+1, i0+1)], [1]
        )
        add_term(
            [u_idx(i0, j0+1), u_idx(i0+1, j0+1)], [1, -1],
            [v_idx(j0+1, k0), v_idx(j0+1, k0+1)], [1, 1],
            [w_idx(k0, i0), w_idx(k0+1, i0+1)], [1, 1]
        )

    def add_schoolbook_2x2x1(i0, j0, k0):
        add_term(
            [u_idx(i0, j0)], [1],
            [v_idx(j0, k0)], [1],
            [w_idx(k0, i0)], [1]
        )
        add_term(
            [u_idx(i0, j0+1)], [1],
            [v_idx(j0+1, k0)], [1],
            [w_idx(k0, i0)], [1]
        )
        add_term(
            [u_idx(i0+1, j0)], [1],
            [v_idx(j0, k0)], [1],
            [w_idx(k0, i0+1)], [1]
        )
        add_term(
            [u_idx(i0+1, j0+1)], [1],
            [v_idx(j0+1, k0)], [1],
            [w_idx(k0, i0+1)], [1]
        )

    def add_schoolbook_2x1x5(i0, j0, k0):
        for dk in range(5):
            add_term(
                [u_idx(i0, j0)], [1],
                [v_idx(j0, k0+dk)], [1],
                [w_idx(k0+dk, i0)], [1]
            )
        for dk in range(5):
            add_term(
                [u_idx(i0+1, j0)], [1],
                [v_idx(j0, k0+dk)], [1],
                [w_idx(k0+dk, i0+1)], [1]
            )

    def add_2x2x5(i0, j0, k0):
        add_strassen_2x2(i0, j0, k0)
        add_strassen_2x2(i0, j0, k0+2)
        add_schoolbook_2x2x1(i0, j0, k0+4)

    def add_2x3x5(i0, j0, k0):
        add_schoolbook_2x1x5(i0, j0, k0)
        add_2x2x5(i0, j0+1, k0)

    add_2x2x5(0, 0, 0)
    add_2x2x5(2, 0, 0)
    add_2x3x5(0, 2, 0)
    add_2x3x5(2, 2, 0)
    
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (u_vec, v_vec, w_vec) in enumerate(terms):
        for i in range(n * m):
            if u_vec[i] != 0:
                U[i][col] = u_vec[i]
        for j in range(m * p):
            if v_vec[j] != 0:
                V[j][col] = v_vec[j]
        for k in range(p * n):
            if w_vec[k] != 0:
                W[k][col] = w_vec[k]
                
    return {"U": U, "V": V, "W": W}
