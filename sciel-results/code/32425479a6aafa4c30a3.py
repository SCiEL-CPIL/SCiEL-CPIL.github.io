import time
import numpy as np

DIMENSIONS = [2, 4, 8]
_START_TIME = time.time()
_DEADLINE = 120.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _build_strassen_block(j0, k0, col_start, U, V, W, n, m, p):
    """
    Builds a 2x2x2 Strassen block.
    A is 2x2 (rows i=0,1; cols j=j0, j0+1)
    B is 2x2 (rows j=j0, j0+1; cols k=k0, k0+1)
    C is 2x2 (rows i=0,1; cols k=k0, k0+1)
    
    U indices: i*m + j
    V indices: j*p + k
    W indices: k*n + i
    """
    col = col_start
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for (i, j), val in u_coeffs.items():
            if val != 0:
                U[i*m + j][col] += val
        for (j, k), val in v_coeffs.items():
            if val != 0:
                V[j*p + k][col] += val
        for (k, i), val in w_coeffs.items():
            if val != 0:
                W[k*n + i][col] += val
        col += 1

    # M1 = (A00 + A11)(B00 + B11)
    add_term(
        {(0, j0): 1, (1, j0+1): 1},
        {(j0, k0): 1, (j0+1, k0+1): 1},
        {(k0, 0): 1, (k0+1, 1): 1}
    )
    
    # M2 = (A10 + A11)B00
    add_term(
        {(1, j0): 1, (1, j0+1): 1},
        {(j0, k0): 1},
        {(k0, 1): 1, (k0+1, 1): -1}
    )
    
    # M3 = A00(B01 - B11)
    add_term(
        {(0, j0): 1},
        {(j0, k0+1): 1, (j0+1, k0+1): -1},
        {(k0+1, 0): 1, (k0+1, 1): 1}
    )
    
    # M4 = A11(B10 - B00)
    add_term(
        {(1, j0+1): 1},
        {(j0+1, k0): 1, (j0, k0): -1},
        {(k0, 0): 1, (k0, 1): 1}
    )
    
    # M5 = (A00 + A01)B11
    add_term(
        {(0, j0): 1, (0, j0+1): 1},
        {(j0+1, k0+1): 1},
        {(k0, 0): -1, (k0+1, 0): 1}
    )
    
    # M6 = (A10 - A00)(B00 + B01)
    add_term(
        {(1, j0): 1, (0, j0): -1},
        {(j0, k0): 1, (j0, k0+1): 1},
        {(k0+1, 1): 1}
    )
    
    # M7 = (A01 - A11)(B10 + B11)
    add_term(
        {(0, j0+1): 1, (1, j0+1): -1},
        {(j0+1, k0): 1, (j0+1, k0+1): 1},
        {(k0, 0): 1}
    )
    
    return col

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    
    # 2 blocks in m dimension (size 2 each)
    # 4 blocks in p dimension (size 2 each)
    # Total 8 blocks, each rank 7. Total rank 56.
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j0 in [0, 2]:
        for k0 in [0, 2, 4, 6]:
            col = _build_strassen_block(j0, k0, col, U, V, W, n, m, p)
            
    # Validation
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    T_recon = np.einsum('ur,vr,wr->uvw', U_np, V_np, W_np)
    
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_target[u, v, w] = 1.0
                
    if not np.allclose(T_recon, T_target, atol=1e-9):
        # Fallback to schoolbook (rank 64)
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i*m+j
                    v = j*p+k
                    w = k*n+i
                    U[u][col] = 1
                    V[v][col] = 1
                    W[w][col] = 1
                    col += 1

    return {
        "U": U,
        "V": V,
        "W": W
    }

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result['U'][0]))
