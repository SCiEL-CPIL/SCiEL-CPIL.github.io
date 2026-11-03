DIMENSIONS = [2, 4, 8]
import time

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    U, V, W = _construct_rank56(n, m, p)
    
    if time.time() > deadline:
        return _construct_schoolbook(n, m, p)
    
    try:
        if _verify(n, m, p, U, V, W):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    return _construct_schoolbook(n, m, p)

def _construct_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _construct_rank56(n, m, p):
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    col = 0
    for ai in range(2):
        for bj in range(2):
            for ck in range(4):
                for (u_loc, v_loc, w_loc) in strassen_terms:
                    a_glob = [ai*2, ai*2+1, 4+ai*2, 4+ai*2+1]
                    b_row0 = 2*bj
                    b_row1 = 2*bj+1
                    b_col0 = 2*ck
                    b_col1 = 2*ck+1
                    b_glob = [b_row0*8+b_col0, b_row0*8+b_col1, b_row1*8+b_col0, b_row1*8+b_col1]
                    c_glob = [4*ai, 4*ai+2, 4*ai+1, 4*ai+3]
                    
                    for l in range(4):
                        if u_loc[l] != 0:
                            U[a_glob[l]][col] += u_loc[l]
                        if v_loc[l] != 0:
                            V[b_glob[l]][col] += v_loc[l]
                        if w_loc[l] != 0:
                            W[c_glob[l]][col] += w_loc[l]
                    col += 1
    
    return U, V, W

def _verify(n, m, p, U, V, W):
    import random
    random.seed(42)
    rank = len(U[0])
    for _ in range(5):
        A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
        C_expected = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        
        C_computed = [[0] * p for _ in range(n)]
        for t in range(rank):
            s_t = sum(U[i*m+j][t] * A[i][j] for i in range(n) for j in range(m))
            t_t = sum(V[j*p+k][t] * B[j][k] for j in range(m) for k in range(p))
            for i in range(n):
                for k in range(p):
                    C_computed[i][k] += W[k*n+i][t] * s_t * t_t
        
        for i in range(n):
            for k in range(p):
                if C_computed[i][k] != C_expected[i][k]:
                    return False
    return True
