import time
import random

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 36
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_rank1(u_dict, v_dict, w_dict):
        nonlocal col
        if col >= rank:
            return
        for idx, val in u_dict.items():
            U[idx][col] += val
        for idx, val in v_dict.items():
            V[idx][col] += val
        for idx, val in w_dict.items():
            W[idx][col] += val
        col += 1

    def add_strassen_2x2_2(a_i0, a_j0, b_i0, b_j0, c_i0, c_k0):
        def a_idx(di, dj):
            return (a_i0 + di) * m + (a_j0 + dj)
        def b_idx(di, dj):
            return (b_i0 + di) * p + (b_j0 + dj)
        def c_idx(di, dk):
            return (c_k0 + dk) * n + (c_i0 + di)

        # M1
        add_rank1({a_idx(0,0): 1, a_idx(1,1): 1}, 
                  {b_idx(0,0): 1, b_idx(1,1): 1},
                  {c_idx(0,0): 1, c_idx(1,1): 1})
        
        # M2
        add_rank1({a_idx(1,0): 1, a_idx(1,1): 1},
                  {b_idx(0,0): 1},
                  {c_idx(1,0): 1, c_idx(1,1): -1})
        
        # M3
        add_rank1({a_idx(0,0): 1},
                  {b_idx(0,1): 1, b_idx(1,1): -1},
                  {c_idx(0,1): 1, c_idx(1,1): 1})
        
        # M4
        add_rank1({a_idx(1,1): 1},
                  {b_idx(1,0): 1, b_idx(0,0): -1},
                  {c_idx(0,0): 1, c_idx(1,0): 1})
        
        # M5
        add_rank1({a_idx(0,0): 1, a_idx(0,1): 1},
                  {b_idx(1,1): 1},
                  {c_idx(0,0): -1, c_idx(0,1): 1})
        
        # M6
        add_rank1({a_idx(1,0): 1, a_idx(0,0): -1},
                  {b_idx(0,0): 1, b_idx(0,1): 1},
                  {c_idx(1,1): 1})
        
        # M7
        add_rank1({a_idx(0,1): 1, a_idx(1,1): -1},
                  {b_idx(1,0): 1, b_idx(1,1): 1},
                  {c_idx(0,0): 1})

    def add_schoolbook_2x2_1(a_i0, a_j0, b_i0, b_j0, c_i0, c_k0):
        # 4 multiplications
        # C(0,k) = A(0,j0)*B(0,k) + A(0,j0+1)*B(1,k)
        # C(1,k) = A(1,j0)*B(0,k) + A(1,j0+1)*B(1,k)
        
        # Term 1: A(0,j0) * B(0,k) -> C(0,k)
        add_rank1({(a_i0 + 0)*m + (a_j0 + 0): 1},
                  {(b_i0 + 0)*p + (b_j0): 1},
                  {(c_k0)*n + (c_i0 + 0): 1})
        
        # Term 2: A(0,j0+1) * B(1,k) -> C(0,k)
        add_rank1({(a_i0 + 0)*m + (a_j0 + 1): 1},
                  {(b_i0 + 1)*p + (b_j0): 1},
                  {(c_k0)*n + (c_i0 + 0): 1})
        
        # Term 3: A(1,j0) * B(0,k) -> C(1,k)
        add_rank1({(a_i0 + 1)*m + (a_j0 + 0): 1},
                  {(b_i0 + 0)*p + (b_j0): 1},
                  {(c_k0)*n + (c_i0 + 1): 1})
        
        # Term 4: A(1,j0+1) * B(1,k) -> C(1,k)
        add_rank1({(a_i0 + 1)*m + (a_j0 + 1): 1},
                  {(b_i0 + 1)*p + (b_j0): 1},
                  {(c_k0)*n + (c_i0 + 1): 1})

    # Block 1: A[0:2, 0:2] * B[0:2, 0:5]
    # Split B into cols 0,1 (2x2), 2,3 (2x2), 4 (2x1)
    add_strassen_2x2_2(0, 0, 0, 0, 0, 0)
    add_strassen_2x2_2(0, 0, 0, 2, 0, 2)
    add_schoolbook_2x2_1(0, 0, 0, 4, 0, 4)
    
    # Block 2: A[0:2, 2:4] * B[2:4, 0:5]
    # Split B into cols 0,1 (2x2), 2,3 (2x2), 4 (2x1)
    add_strassen_2x2_2(0, 2, 2, 0, 0, 0)
    add_strassen_2x2_2(0, 2, 2, 2, 0, 2)
    add_schoolbook_2x2_1(0, 2, 2, 4, 0, 4)

    return {"U": U, "V": V, "W": W}
