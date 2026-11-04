import numpy as np

def construct():
    n, m, p = 4, 4, 7
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    def add_term(u_dict, v_dict, w_dict):
        U_rows.append(u_dict)
        V_rows.append(v_dict)
        W_rows.append(w_dict)
    
    def schoolbook_2x2x1(i0, j0, k0):
        for i in range(2):
            for j in range(2):
                add_term(
                    {(i0+i, j0+j): 1},
                    {(j0+j, k0): 1},
                    {(k0, i0+i): 1}
                )
    
    def strassen_2x2x2(i0, j0, k0):
        add_term(
            {(i0+0,j0+0): 1, (i0+1,j0+1): 1},
            {(j0+0,k0+0): 1, (j0+1,k0+1): 1},
            {(k0+0, i0+0): 1}
        )
        add_term(
            {(i0+1,j0+0): 1, (i0+1,j0+1): 1},
            {(j0+0,k0+0): 1},
            {(k0+0, i0+1): 1}
        )
        add_term(
            {(i0+0,j0+0): 1},
            {(j0+1,k0+0): 1, (j0+1,k0+1): -1},
            {(k0+1, i0+0): 1}
        )
        add_term(
            {(i0+1,j0+1): 1},
            {(j0+0,k0+1): 1, (j0+0,k0+0): -1},
            {(k0+0, i0+0): 1, (k0+0, i0+1): 1}
        )
        add_term(
            {(i0+0,j0+0): 1, (i0+0,j0+1): 1},
            {(j0+1,k0+1): 1},
            {(k0+1, i0+0): 1, (k0+1, i0+1): -1}
        )
        add_term(
            {(i0+1,j0+0): 1, (i0+0,j0+0): -1},
            {(j0+0,k0+0): 1, (j0+0,k0+1): 1},
            {(k0+1, i0+1): 1}
        )
        add_term(
            {(i0+0,j0+1): 1, (i0+1,j0+1): -1},
            {(j0+1,k0+0): 1, (j0+1,k0+1): 1},
            {(k0+0, i0+0): 1}
        )
    
    def sub_2x2x3(i0, j0, k0):
        strassen_2x2x2(i0, j0, k0)
        schoolbook_2x2x1(i0, j0, k0+2)
    
    def sub_2x2x4(i0, j0, k0):
        sub_2x2x3(i0, j0, k0)
        schoolbook_2x2x1(i0, j0, k0+3)
    
    def sub_2x2x7(i0, j0, k0):
        sub_2x2x3(i0, j0, k0)
        sub_2x2x4(i0, j0, k0+3)
    
    def sub_2x4x7(i0, j0, k0):
        sub_2x2x7(i0, j0, k0)
        sub_2x2x7(i0, j0+2, k0)
    
    def sub_4x4x7(i0, j0, k0):
        sub_2x4x7(i0, j0, k0)
        sub_2x4x7(i0+2, j0, k0)
    
    sub_4x4x7(0, 0, 0)
    
    rank = len(U_rows)
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for c in range(rank):
        for (i,j), val in U_rows[c].items():
            U[i*m+j][c] = val
        for (j,k), val in V_rows[c].items():
            V[j*p+k][c] = val
        for (k,i), val in W_rows[c].items():
            W[k*n+i][c] = val
    
    return {"U": U, "V": V, "W": W}
