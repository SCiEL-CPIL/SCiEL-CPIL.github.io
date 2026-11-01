"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vals, v_vals, w_vals):
        U_cols.append(u_vals)
        V_cols.append(v_vals)
        W_cols.append(w_vals)
    
    def strassen_2x2(j0, j1, k0, k1):
        # M1 = (A[0,j0]+A[1,j1]) * (B[j0,k0]+B[j1,k1])
        add_term({0*m+j0: 1, 1*m+j1: 1},
                 {j0*p+k0: 1, j1*p+k1: 1},
                 {k0*n+0: 1, k1*n+1: 1, k1*n+0: -1, k0*n+1: -1})
        
        # M2 = (A[1,j0]+A[1,j1]) * B[j0,k0]
        add_term({1*m+j0: 1, 1*m+j1: 1},
                 {j0*p+k0: 1},
                 {k0*n+1: 1})
        
        # M3 = A[0,j0] * (B[j1,k0]-B[j1,k1])
        add_term({0*m+j0: 1},
                 {j1*p+k0: 1, j1*p+k1: -1},
                 {k1*n+0: 1, k1*n+1: 1})
        
        # M4 = A[1,j1] * (B[j0,k1]-B[j0,k0])
        add_term({1*m+j1: 1},
                 {j0*p+k1: 1, j0*p+k0: -1},
                 {k0*n+0: 1, k0*n+1: 1})
        
        # M5 = (A[0,j0]+A[0,j1]) * B[j1,k1]
        add_term({0*m+j0: 1, 0*m+j1: 1},
                 {j1*p+k1: 1},
                 {k0*n+0: -1, k1*n+0: 1})
        
        # M6 = (A[1,j0]-A[0,j0]) * (B[j0,k0]+B[j1,k0])
        add_term({1*m+j0: 1, 0*m+j0: -1},
                 {j0*p+k0: 1, j1*p+k0: 1},
                 {k1*n+1: 1})
        
        # M7 = (A[0,j1]-A[1,j1]) * (B[j0,k1]+B[j1,k1])
        add_term({0*m+j1: 1, 1*m+j1: -1},
                 {j0*p+k1: 1, j1*p+k1: 1},
                 {k0*n+0: 1})
    
    def schoolbook_term(i, j, k):
        add_term({i*m+j: 1}, {j*p+k: 1}, {k*n+i: 1})
    
    j_groups = [(0, 1), (2, 3), (4,)]
    k_groups = [(0, 1), (2, 3), (4,)]
    
    for jg in j_groups:
        for kg in k_groups:
            if len(jg) == 2 and len(kg) == 2:
                strassen_2x2(jg[0], jg[1], kg[0], kg[1])
            else:
                for i in range(n):
                    for j in jg:
                        for k in kg:
                            schoolbook_term(i, j, k)
    
    rank = len(U_cols)
    U_mat = [[0] * rank for _ in range(n * m)]
    V_mat = [[0] * rank for _ in range(m * p)]
    W_mat = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        for idx, val in U_cols[r].items():
            U_mat[idx][r] = val
        for idx, val in V_cols[r].items():
            V_mat[idx][r] = val
        for idx, val in W_cols[r].items():
            W_mat[idx][r] = val
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
