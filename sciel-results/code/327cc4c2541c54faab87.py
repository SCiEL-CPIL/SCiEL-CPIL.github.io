"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # 2, 5, 5
    
    U = {}
    V = {}
    W = {}
    col = 0
    
    def add_rank1(u_entries, v_entries, w_entries):
        nonlocal col
        for idx, val in u_entries:
            U[(idx, col)] = U.get((idx, col), 0) + val
        for idx, val in v_entries:
            V[(idx, col)] = V.get((idx, col), 0) + val
        for idx, val in w_entries:
            W[(idx, col)] = W.get((idx, col), 0) + val
        col += 1
    
    def add_strassen(i0, j0, k0):
        i1, j1, k1 = i0+1, j0+1, k0+1
        # Term 1
        add_rank1([(i0*m+j0,1),(i1*m+j1,1)], [(j0*p+k0,1),(j1*p+k1,1)], [(k0*n+i0,1),(k1*n+i1,1)])
        # Term 2
        add_rank1([(i1*m+j0,1),(i1*m+j1,1)], [(j0*p+k0,1)], [(k0*n+i1,1)])
        # Term 3
        add_rank1([(i0*m+j0,1)], [(j0*p+k1,1),(j1*p+k1,-1)], [(k1*n+i0,1)])
        # Term 4
        add_rank1([(i1*m+j1,1)], [(j1*p+k0,1),(j0*p+k0,-1)], [(k0*n+i1,1)])
        # Term 5
        add_rank1([(i0*m+j0,1),(i1*m+j0,1)], [(j1*p+k1,1)], [(k1*n+i0,1)])
        # Term 6
        add_rank1([(i0*m+j1,1),(i0*m+j0,-1)], [(j0*p+k0,1),(j1*p+k0,1)], [(k1*n+i1,1)])
        # Term 7
        add_rank1([(i1*m+j0,1),(i1*m+j1,-1)], [(j0*p+k1,1),(j1*p+k1,1)], [(k1*n+i1,1)])
    
    def add_simple(i0, j0, k0, di, dj, dk):
        for a in range(di):
            for b in range(dj):
                for c in range(dk):
                    ii, jj, kk = i0+a, j0+b, k0+c
                    add_rank1([(ii*m+jj,1)], [(jj*p+kk,1)], [(kk*n+ii,1)])
    
    # Apply decomposition
    add_strassen(0, 0, 0)   # j:{0,1}, k:{0,1}
    add_strassen(0, 0, 2)   # j:{0,1}, k:{2,3}
    add_simple(0, 0, 4, 2, 1, 1)   # j:{0,1}, k:{4}
    add_strassen(0, 2, 0)   # j:{2,3}, k:{0,1}
    add_strassen(0, 2, 2)   # j:{2,3}, k:{2,3}
    add_simple(0, 2, 4, 2, 1, 1)   # j:{2,3}, k:{4}
    add_simple(0, 4, 0, 2, 1, 2)   # j:{4}, k:{0,1}
    add_simple(0, 4, 2, 2, 1, 2)   # j:{4}, k:{2,3}
    add_simple(0, 4, 4, 2, 1, 1)   # j:{4}, k:{4}
    
    rank = col
    
    U_mat = [[0]*rank for _ in range(n*m)]
    V_mat = [[0]*rank for _ in range(m*p)]
    W_mat = [[0]*rank for _ in range(p*n)]
    
    for (idx, c), val in U.items():
        U_mat[idx][c] = val
    for (idx, c), val in V.items():
        V_mat[idx][c] = val
    for (idx, c), val in W.items():
        W_mat[idx][c] = val
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
