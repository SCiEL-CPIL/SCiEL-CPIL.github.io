"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = 120  # seconds
    
    n, m, p = 3, 5, 5  # A is n x m, B is m x p, C is n x p
    
    # Partition indices
    i_groups = [(0, 1), (2,)]
    j_groups = [(0, 1), (2, 3), (4,)]
    k_groups = [(0, 1), (2, 3), (4,)]
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    for ig_idx, i_g in enumerate(i_groups):
        if time.time() - start_time > deadline - 5:
            break
        for jg_idx, j_g in enumerate(j_groups):
            if time.time() - start_time > deadline - 5:
                break
            for kg_idx, k_g in enumerate(k_groups):
                if time.time() - start_time > deadline - 5:
                    break
                ni = len(i_g)
                nj = len(j_g)
                nk = len(k_g)
                
                if ni == 2 and nj == 2 and nk == 2:
                    i0, i1 = i_g
                    j0, j1 = j_g
                    k0, k1 = k_g
                    
                    strassen_products = [
                        ({i0*m+k0: 1, i1*m+k1: 1}, {k0*p+j0: 1, k1*p+j1: 1}, {j0*n+i0: 1, j1*n+i1: 1}),
                        ({i1*m+k0: 1, i1*m+k1: 1}, {k0*p+j0: 1}, {j0*n+i1: 1, j1*n+i1: -1}),
                        ({i0*m+k0: 1}, {k1*p+j0: 1, k1*p+j1: -1}, {j1*n+i0: 1, j1*n+i1: 1}),
                        ({i1*m+k1: 1}, {k0*p+j1: 1, k0*p+j0: -1}, {j0*n+i0: 1, j0*n+i1: 1}),
                        ({i0*m+k0: 1, i0*m+k1: 1}, {k1*p+j1: 1}, {j0*n+i0: -1, j1*n+i0: 1}),
                        ({i1*m+k0: 1, i0*m+k0: -1}, {k0*p+j0: 1, k0*p+j1: 1}, {j1*n+i1: 1}),
                        ({i0*m+k1: 1, i1*m+k1: -1}, {k1*p+j0: 1, k1*p+j1: 1}, {j0*n+i0: 1}),
                    ]
                    
                    for U_d, V_d, W_d in strassen_products:
                        U_cols.append(U_d)
                        V_cols.append(V_d)
                        W_cols.append(W_d)
                else:
                    for i in i_g:
                        for j in j_g:
                            for k in k_g:
                                U_cols.append({i*m+k: 1})
                                V_cols.append({k*p+j: 1})
                                W_cols.append({j*n+i: 1})
    
    rank = len(U_cols)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for col in range(rank):
        for idx, val in U_cols[col].items():
            U[idx][col] = val
        for idx, val in V_cols[col].items():
            V[idx][col] = val
        for idx, val in W_cols[col].items():
            W[idx][col] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
