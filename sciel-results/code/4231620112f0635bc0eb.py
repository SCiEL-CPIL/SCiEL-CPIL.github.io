import numpy as np

def construct():
    n, m, p = 2, 5, 6
    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Helper to add a rank-1 component
    def add_term(u_idx, v_idx, w_idx, val):
        nonlocal col
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val
        col += 1

    # Helper for [2, 2, 1]
    def add_221(i_off, j_off, k_off):
        nonlocal col
        # i in 0..1, j in 0..1, k in 0..0
        # Terms:
        # 0,0,0
        add_term(i_off*5 + (j_off+0), (j_off+0)*6 + (k_off+0), (k_off+0)*2 + i_off, 1)
        # 0,1,0
        add_term(i_off*5 + (j_off+1), (j_off+1)*6 + (k_off+0), (k_off+0)*2 + i_off, 1)
        # 1,0,0
        add_term((i_off+1)*5 + (j_off+0), (j_off+0)*6 + (k_off+0), (k_off+0)*2 + (i_off+1), 1)
        # 1,1,0
        add_term((i_off+1)*5 + (j_off+1), (j_off+1)*6 + (k_off+0), (k_off+0)*2 + (i_off+1), 1)

    # Helper for [2, 2, 2]
    def add_222(i_off, j_off, k_off):
        nonlocal col
        # Strassen decomposition
        # P1
        # U: (1,0,0,1) for i,j
        # V: (1,0,0,1) for j,k
        # W: C00+=1, C11+=1
        # We need to add 1 term for each non-zero in the product?
        # No, P1 is ONE rank-1 term in the decomposition of the SUB-PROBLEM?
        # No, P1 is a product of sums.
        # $P_1 = (\sum u_{ij} A_{ij}) (\sum v_{jk} B_{jk})$.
        # This is a single rank-1 component of the tensor T.
        # So we add ONE column to U, V, W.
        
        # Let's define the vectors for P1
        u_vec = {
            (0,0): 1, (1,1): 1
        }
        v_vec = {
            (0,0): 1, (1,1): 1
        }
        w_vec = {
            (0,0): 1, (1,1): 1
        }
        # Fill U
        for (li, lj), val in u_vec.items():
            U[i_off*5 + (j_off+lj)][col] += val
        # Fill V
        for (lj, lk), val in v_vec.items():
            V[(j_off+lj)*6 + (k_off+lk)][col] += val
        # Fill W
        for (lk, li), val in w_vec.items():
            W[(k_off+lk)*2 + (i_off+li)][col] += val
        col += 1
        
        # P2
        u_vec = {(1,0): 1, (1,1): 1}
        v_vec = {(0,0): 1}
        w_vec = {(0,1): 1}
        for (li, lj), val in u_vec.items():
            U[i_off*5 + (j_off+lj)][col] += val
        for (lj, lk), val in v_vec.items():
            V[(j_off+lj)*6 + (k_off+lk)][col] += val
        for (lk, li), val in w_vec.items():
            W[(k_off+lk)*2 + (i_off+li)][col] += val
        col += 1
        
        # P3
        u_vec = {(0,0): 1}
        v_vec = {(0,1): 1, (1,1): -1}
        w_vec = {(1,0): 1, (1,1): -1}
        for (li, lj), val in u_vec.items():
            U[i_off*5 + (j_off+lj)][col] += val
        for (lj, lk), val in v_vec.items():
            V[(j_off+lj)*6 + (k_off+lk)][col] += val
        for (lk, li), val in w_vec.items():
            W[(k_off+lk)*2 + (i_off+li)][col] += val
        col += 1
        
        # P4
        u_vec = {(1,1): 1}
        v_vec = {(0,0): -1, (1,0): 1}
        w_vec = {(0,0): -1, (0,1): 1}
        for (li, lj), val in u_vec.items():
            U[i_off*5 + (j_off+lj)][col] += val
        for (lj, lk), val in v_vec.items():
            V[(j_off+lj)*6 + (k_off+lk)][col] += val
        for (lk, li), val in w_vec.items():
            W[(k_off+lk)*2 + (i_off+li)][col] += val
        col += 1
        
        # P5
        u_vec = {(0,0): 1, (0,1): 1}
        v_vec = {(1,1): 1}
        w_vec = {(1,1): 1}
        for (li, lj), val in u_vec.items():
            U[i_off*5 + (j_off+lj)][col] += val
        for (lj, lk), val in v_vec.items():
            V[(j_off+lj)*6 + (k_off+lk)][col] += val
        for (lk, li), val in w_vec.items():
            W[(k_off+lk)*2 + (i_off+li)][col] += val
        col += 1
        
        # P6
        u_vec = {(0,0): -1, (1,0): 1}
        v_vec = {(0,0): 1, (0,1): 1}
        w_vec = {(0,0): 1}
        for (li, lj), val in u_vec.items():
            U[i_off*5 + (j_off+lj)][col] += val
        for (lj, lk), val in v_vec.items():
            V[(j_off+lj)*6 + (k_off+lk)][col] += val
        for (lk, li), val in w_vec.items():
            W[(k_off+lk)*2 + (i_off+li)][col] += val
        col += 1
        
        # P7
        u_vec = {(0,1): 1, (1,1): -1}
        v_vec = {(1,0): 1, (1,1): 1}
        w_vec = {(1,0): 1}
        for (li, lj), val in u_vec.items():
            U[i_off*5 + (j_off+lj)][col] += val
        for (lj, lk), val in v_vec.items():
            V[(j_off+lj)*6 + (k_off+lk)][col] += val
        for (lk, li), val in w_vec.items():
            W[(k_off+lk)*2 + (i_off+li)][col] += val
        col += 1

    # Helper for [2, 3, 3]
    def add_233(i_off, j_off, k_off):
        nonlocal col
        # Split j into 1 and 2
        # Part 1: [2, 1, 3] for j=0
        # Part 2: [2, 2, 3] for j=1,2
        
        # Add [2, 1, 3]
        # i in 0..1, j in 0..0, k in 0..2
        # Terms:
        # 0,0,0
        add_term(i_off*5 + (j_off+0), (j_off+0)*6 + (k_off+0), (k_off+0)*2 + i_off, 1)
        # 0,0,1
        add_term(i_off*5 + (j_off+0), (j_off+0)*6 + (k_off+1), (k_off+1)*2 + i_off, 1)
        # 0,0,2
        add_term(i_off*5 + (j_off+0), (j_off+0)*6 + (k_off+2), (k_off+2)*2 + i_off, 1)
        # 1,0,0
        add_term((i_off+1)*5 + (j_off+0), (j_off+0)*6 + (k_off+0), (k_off+0)*2 + (i_off+1), 1)
        # 1,0,1
        add_term((i_off+1)*5 + (j_off+0), (j_off+0)*6 + (k_off+1), (k_off+1)*2 + (i_off+1), 1)
        # 1,0,2
        add_term((i_off+1)*5 + (j_off+0), (j_off+0)*6 + (k_off+2), (k_off+2)*2 + (i_off+1), 1)
        
        # Add [2, 2, 3] for j=1,2
        # This is [2, 2, 1] + [2, 2, 2]
        # [2, 2, 1] for k=0
        add_221(i_off, j_off+1, k_off)
        # [2, 2, 2] for k=1,2
        add_222(i_off, j_off+1, k_off+1)

    # Combine
    # J1={0,1}, K1={0,1,2} -> [2,2,3]
    add_221(0, 0, 0) # [2,2,1] part of first [2,2,3]
    add_222(0, 0, 1) # [2,2,2] part of first [2,2,3]
    
    # J1={0,1}, K2={3,4,5} -> [2,2,3]
    add_221(0, 0, 3)
    add_222(0, 0, 4)
    
    # J2={2,3,4}, K1={0,1,2} -> [2,3,3]
    add_233(0, 2, 0)
    
    # J2={2,3,4}, K2={3,4,5} -> [2,3,3]
    add_233(0, 2, 3)
    
    return {"U": U, "V": V, "W": W}
