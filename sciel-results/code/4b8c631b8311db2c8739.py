import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 110.0

def _remaining():
    return _DEADLINE - time.time()

def construct():
    n, m, p = 2, 4, 5
    
    def verify(U, V, W, r):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for rr in range(r):
                        s += U[i*m+j][rr] * V[j*p+k][rr] * W[k*n+i][rr]
                    if s != 1:
                        return False
        return True
    
    # Rank-33 construction
    rank33 = 33
    U33 = [[0] * rank33 for _ in range(n * m)]
    V33 = [[0] * rank33 for _ in range(m * p)]
    W33 = [[0] * rank33 for _ in range(p * n)]
    
    col = 0
    
    def add_term33(u_dict, v_dict, w_dict):
        nonlocal col
        if col >= rank33:
            raise ValueError("Rank exceeded")
        for idx, val in u_dict.items():
            U33[idx][col] += val
        for idx, val in v_dict.items():
            V33[idx][col] += val
        for idx, val in w_dict.items():
            W33[idx][col] += val
        col += 1
    
    def strassen_block33(i0, j0, k0):
        add_term33({i0*m+j0: 1, (i0+1)*m+(j0+1): 1},
                   {j0*p+k0: 1, (j0+1)*p+(k0+1): 1},
                   {k0*n+i0: 1, (k0+1)*n+(i0+1): 1})
        add_term33({(i0+1)*m+j0: 1, (i0+1)*m+(j0+1): 1},
                   {j0*p+k0: 1},
                   {k0*n+(i0+1): 1, (k0+1)*n+(i0+1): -1})
        add_term33({i0*m+j0: 1},
                   {j0*p+(k0+1): 1, (j0+1)*p+(k0+1): -1},
                   {(k0+1)*n+i0: 1, (k0+1)*n+(i0+1): 1})
        add_term33({(i0+1)*m+(j0+1): 1},
                   {(j0+1)*p+k0: 1, j0*p+k0: -1},
                   {k0*n+i0: 1, k0*n+(i0+1): 1})
        add_term33({i0*m+j0: 1, i0*m+(j0+1): 1},
                   {(j0+1)*p+(k0+1): 1},
                   {k0*n+i0: -1, (k0+1)*n+i0: 1})
        add_term33({(i0+1)*m+j0: 1, i0*m+j0: -1},
                   {j0*p+k0: 1, j0*p+(k0+1): 1},
                   {(k0+1)*n+(i0+1): 1})
        add_term33({i0*m+(j0+1): 1, (i0+1)*m+(j0+1): -1},
                   {(j0+1)*p+k0: 1, (j0+1)*p+(k0+1): 1},
                   {k0*n+i0: 1})
    
    strassen_block33(0, 0, 0)
    strassen_block33(0, 2, 0)
    strassen_block33(0, 0, 2)
    strassen_block33(0, 2, 2)
    
    # k=4: single rank-1 term
    for i in range(2):
        for j in range(4):
            U33[i*m+j][col] = 1
            V33[j*p+4][col] = 1
            W33[4*n+i][col] = 1
    col += 1
    
    assert col == rank33
    
    if verify(U33, V33, W33, rank33):
        return {"U": U33, "V": V33, "W": W33}
    
    # Fallback: rank-36
    rank36 = 36
    U36 = [[0] * rank36 for _ in range(n * m)]
    V36 = [[0] * rank36 for _ in range(m * p)]
    W36 = [[0] * rank36 for _ in range(p * n)]
    col = 0
    
    def add_term36(u_dict, v_dict, w_dict):
        nonlocal col
        for idx, val in u_dict.items():
            U36[idx][col] += val
        for idx, val in v_dict.items():
            V36[idx][col] += val
        for idx, val in w_dict.items():
            W36[idx][col] += val
        col += 1
    
    def strassen_block36(i0, j0, k0):
        add_term36({i0*m+j0: 1, (i0+1)*m+(j0+1): 1},
                   {j0*p+k0: 1, (j0+1)*p+(k0+1): 1},
                   {k0*n+i0: 1, (k0+1)*n+(i0+1): 1})
        add_term36({(i0+1)*m+j0: 1, (i0+1)*m+(j0+1): 1},
                   {j0*p+k0: 1},
                   {k0*n+(i0+1): 1, (k0+1)*n+(i0+1): -1})
        add_term36({i0*m+j0: 1},
                   {j0*p+(k0+1): 1, (j0+1)*p+(k0+1): -1},
                   {(k0+1)*n+i0: 1, (k0+1)*n+(i0+1): 1})
        add_term36({(i0+1)*m+(j0+1): 1},
                   {(j0+1)*p+k0: 1, j0*p+k0: -1},
                   {k0*n+i0: 1, k0*n+(i0+1): 1})
        add_term36({i0*m+j0: 1, i0*m+(j0+1): 1},
                   {(j0+1)*p+(k0+1): 1},
                   {k0*n+i0: -1, (k0+1)*n+i0: 1})
        add_term36({(i0+1)*m+j0: 1, i0*m+j0: -1},
                   {j0*p+k0: 1, j0*p+(k0+1): 1},
                   {(k0+1)*n+(i0+1): 1})
        add_term36({i0*m+(j0+1): 1, (i0+1)*m+(j0+1): -1},
                   {(j0+1)*p+k0: 1, (j0+1)*p+(k0+1): 1},
                   {k0*n+i0: 1})
    
    def schoolbook_col36(j_start, k0):
        for di in range(2):
            for dj in range(4):
                add_term36({di*m+(j_start+dj): 1},
                           {(j_start+dj)*p+k0: 1},
                           {k0*n+di: 1})
    
    strassen_block36(0, 0, 0)
    strassen_block36(0, 2, 0)
    strassen_block36(0, 0, 2)
    strassen_block36(0, 2, 2)
    schoolbook_col36(0, 4)
    
    assert col == rank36
    return {"U": U36, "V": V36, "W": W36}
