import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = 36
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to set U, V, W for a term
    def add_term(u_indices, v_indices, w_indices):
        nonlocal col
        for idx, val in u_indices:
            U[idx, col] += val
        for idx, val in v_indices:
            V[idx, col] += val
        for idx, val in w_indices:
            W[idx, col] += val
        col += 1

    # Indices helper
    def u_idx(i, k): return i * m + k
    def v_idx(k, j): return k * p + j
    def w_idx(j, i): return j * n + i

    for b in range(3):
        j0 = 2 * b
        j1 = 2 * b + 1
        
        # Part 1: Strassen for 2x2 x 2x2 (A[0:2, 0:2] * B[0:2, j0:j1])
        # a=A00, b=A01, c=A10, d=A11
        # e=B0j0, f=B0j1, g=B1j0, h=B1j1
        
        # P1 = (a+d)(e+h)
        add_term(
            [(u_idx(0,0), 1), (u_idx(1,1), 1)],
            [(v_idx(0,j0), 1), (v_idx(1,j1), 1)],
            [(w_idx(j0,0), 1), (w_idx(j1,0), -1), (w_idx(j0,1), -1), (w_idx(j1,1), 1)]
        )
        
        # P2 = (c+d)e
        add_term(
            [(u_idx(1,0), 1), (u_idx(1,1), 1)],
            [(v_idx(0,j0), 1)],
            [(w_idx(j0,1), 1), (w_idx(j1,1), -1)]
        )
        
        # P3 = a(f-h)
        add_term(
            [(u_idx(0,0), 1)],
            [(v_idx(0,j1), 1), (v_idx(1,j1), -1)],
            [(w_idx(j1,0), 1), (w_idx(j1,1), 1)]
        )
        
        # P4 = d(g-e)
        add_term(
            [(u_idx(1,1), 1)],
            [(v_idx(1,j0), 1), (v_idx(0,j0), -1)],
            [(w_idx(j0,0), 1), (w_idx(j0,1), 1)]
        )
        
        # P5 = (a+b)h
        add_term(
            [(u_idx(0,0), 1), (u_idx(0,1), 1)],
            [(v_idx(1,j1), 1)],
            [(w_idx(j0,0), -1), (w_idx(j1,0), 1)]
        )
        
        # P6 = (c-a)(e+f)
        add_term(
            [(u_idx(1,0), 1), (u_idx(0,0), -1)],
            [(v_idx(0,j0), 1), (v_idx(0,j1), 1)],
            [(w_idx(j1,1), 1)]
        )
        
        # P7 = (b-d)(g+h)
        add_term(
            [(u_idx(0,1), 1), (u_idx(1,1), -1)],
            [(v_idx(1,j0), 1), (v_idx(1,j1), 1)],
            [(w_idx(j0,0), 1), (w_idx(j1,1), 1)]
        )
        
        # Part 2: Outer product for A[:, 2] * B[2, j0:j1]
        # u = [A02, A12], v = [B2j0, B2j1]
        
        # T8: A02 * B2j0 -> C[j0, 0]
        add_term(
            [(u_idx(0,2), 1)],
            [(v_idx(2,j0), 1)],
            [(w_idx(j0,0), 1)]
        )
        
        # T9: A02 * B2j1 -> C[j1, 0]
        add_term(
            [(u_idx(0,2), 1)],
            [(v_idx(2,j1), 1)],
            [(w_idx(j1,0), 1)]
        )
        
        # T10: A12 * B2j0 -> C[j0, 1]
        add_term(
            [(u_idx(1,2), 1)],
            [(v_idx(2,j0), 1)],
            [(w_idx(j0,1), 1)]
        )
        
        # T11: A12 * B2j1 -> C[j1, 1]
        add_term(
            [(u_idx(1,2), 1)],
            [(v_idx(2,j1), 1)],
            [(w_idx(j1,1), 1)]
        )

    # Part 3: Last column (j=6)
    # C[:, 6] = A * B[:, 6]
    
    # T12: A[:, 0] * B[0, 6]
    add_term(
        [(u_idx(0,0), 1), (u_idx(1,0), 1)],
        [(v_idx(0,6), 1)],
        [(w_idx(6,0), 1), (w_idx(6,1), 1)]
    )
    
    # T13: A[:, 1] * B[1, 6]
    add_term(
        [(u_idx(0,1), 1), (u_idx(1,1), 1)],
        [(v_idx(1,6), 1)],
        [(w_idx(6,0), 1), (w_idx(6,1), 1)]
    )
    
    # T14: A[:, 2] * B[2, 6]
    add_term(
        [(u_idx(0,2), 1), (u_idx(1,2), 1)],
        [(v_idx(2,6), 1)],
        [(w_idx(6,0), 1), (w_idx(6,1), 1)]
    )

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
