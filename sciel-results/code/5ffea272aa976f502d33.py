"""Matrix multiplication 4x5 x 5x6 tensor decomposition using block Strassen."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # Split indices:
    # i: [0,1] and [2,3] (two blocks of size 2)
    # j: [0,1], [2,3], [4] (blocks of sizes 2, 2, 1)
    # k: [0,1], [2,3], [4,5] (blocks of sizes 2, 2, 2)
    
    i_blocks = [(0, 1), (2, 3)]
    j_blocks = [(0, 1), (2, 3), (4, 4)]
    k_blocks = [(0, 1), (2, 3), (4, 5)]
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(U_dict, V_dict, W_dict):
        U_cols.append(U_dict)
        V_cols.append(V_dict)
        W_cols.append(W_dict)
    
    for i_start, i_end in i_blocks:
        for j_start, j_end in j_blocks:
            for k_start, k_end in k_blocks:
                i_size = i_end - i_start + 1
                j_size = j_end - j_start + 1
                k_size = k_end - k_start + 1
                
                if i_size == 2 and j_size == 2 and k_size == 2:
                    # Strassen 7 terms
                    # Local (i_l, j_l, k_l) -> global (i_start+i_l, j_start+j_l, k_start+k_l)
                    # W index: (k_l, i_l)
                    strassen_terms = [
                        # M1: U=(A00+A11), V=(B00+B11), W: C00+=1, C11+=1
                        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
                        # M2: U=(A10+A11), V=B00, W: C10+=1
                        ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1}),
                        # M3: U=A00, V=(B10-B11), W: C01+=1
                        ({(0,0):1}, {(1,0):1, (1,1):-1}, {(1,0):1}),
                        # M4: U=A11, V=(B01-B00), W: C00+=1, C10+=1
                        ({(1,1):1}, {(0,1):1, (0,0):-1}, {(0,0):1, (0,1):1}),
                        # M5: U=(A00+A01), V=B11, W: C00-=1, C01+=1
                        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1}),
                        # M6: U=(A10-A00), V=(B00+B01), W: C11+=1
                        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
                        # M7: U=(A01-A11), V=(B10+B11), W: C00+=1, C11+=1
                        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
                    ]
                    
                    for U_l, V_l, W_l in strassen_terms:
                        U_dict = {}
                        for (li, lj), coeff in U_l.items():
                            U_dict[(i_start + li, j_start + lj)] = coeff
                        V_dict = {}
                        for (lj, lk), coeff in V_l.items():
                            V_dict[(j_start + lj, k_start + lk)] = coeff
                        W_dict = {}
                        for (lk, li), coeff in W_l.items():
                            W_dict[(k_start + lk, i_start + li)] = coeff
                        add_term(U_dict, V_dict, W_dict)
                        
                elif i_size == 2 and j_size == 1 and k_size == 2:
                    # Schoolbook: 4 terms
                    for li in range(2):
                        for lk in range(2):
                            gi = i_start + li
                            gj = j_start
                            gk = k_start + lk
                            add_term({(gi, gj): 1}, {(gj, gk): 1}, {(gk, gi): 1})
                else:
                    # Fallback schoolbook
                    for li in range(i_size):
                        for lj in range(j_size):
                            for lk in range(k_size):
                                gi = i_start + li
                                gj = j_start + lj
                                gk = k_start + lk
                                add_term({(gi, gj): 1}, {(gj, gk): 1}, {(gk, gi): 1})
    
    rank = len(U_cols)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        for (i, j), val in U_cols[c].items():
            U[i * m + j][c] = val
        for (j, k), val in V_cols[c].items():
            V[j * p + k][c] = val
        for (k, i), val in W_cols[c].items():
            W[k * n + i][c] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
