import time
import numpy as np

DIMENSIONS = [2, 4, 8]
_START_TIME = time.time()
_DEADLINE = 120.0  # Safety margin before 150s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _build_strassen_block(j0, k0, col_start, U, V, W, n, m, p):
    """Build one 2x2x2 Strassen block at matrix positions j0..j0+1, k0..k0+1."""
    col = col_start
    # Local indices: i in {0, 1}, j in {j0, j0+1}, k in {k0, k0+1}
    # W indices: k*n + i
    # C00: k=k0, i=0 -> W[k0*n+0]
    # C01: k=k0+1, i=0 -> W[(k0+1)*n+0]
    # C10: k=k0, i=1 -> W[k0*n+1]
    # C11: k=k0+1, i=1 -> W[(k0+1)*n+1]
    
    def add_term(u_coeffs, v_coeffs, w_coeffs, col_idx):
        for (i, j), val in u_coeffs.items():
            if val != 0:
                U[i*m + j][col_idx] += val
        for (j, k), val in v_coeffs.items():
            if val != 0:
                V[j*p + k][col_idx] += val
        for (k, i), val in w_coeffs.items():
            if val != 0:
                W[k*n + i][col_idx] += val
    
    # M1 = (A00 + A11)(B00 + B11)
    add_term(
        {(0, j0): 1, (1, j0+1): 1},
        {(j0, k0): 1, (j0+1, k0+1): 1},
        {(k0, 0): 1, (k0+1, 1): 1},
        col
    )
    col += 1
    
    # M2 = (A10 + A11)B00
    add_term(
        {(1, j0): 1, (1, j0+1): 1},
        {(j0, k0): 1},
        {(k0, 1): 1, (k0+1, 1): -1},
        col
    )
    col += 1
    
    # M3 = A00(B01 - B11)
    add_term(
        {(0, j0): 1},
        {(j0, k0+1): 1, (j0+1, k0+1): -1},
        {(k0+1, 0): 1, (k0+1, 1): 1},
        col
    )
    col += 1
    
    # M4 = A11(B10 - B00)
    add_term(
        {(1, j0+1): 1},
        {(j0+1, k0): 1, (j0, k0): -1},
        {(k0, 0): 1, (k0, 1): 1},
        col
    )
    col += 1
    
    # M5 = (A00 + A01)B11
    add_term(
        {(0, j0): 1, (0, j0+1): 1},
        {(j0+1, k0+1): 1},
        {(k0, 0): -1, (k0+1, 0): 1},
        col
    )
    col += 1
    
    # M6 = (A10 - A00)(B00 + B01)
    add_term(
        {(1, j0): 1, (0, j0): -1},
        {(j0, k0): 1, (j0, k0+1): 1},
        {(k0+1, 1): 1},
        col
    )
    col += 1
    
    # M7 = (A01 - A11)(B10 + B11)
    add_term(
        {(0, j0+1): 1, (1, j0+1): -1},
        {(j0+1, k0): 1, (j0+1, k0+1): 1},
        {(k0, 0): 1},
        col
    )
    col += 1
    
    return col

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    
    # Strassen-based rank: 8 blocks * 7 = 56
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j0 in [0, 2]:
        for k0 in [0, 2, 4, 6]:
            col = _build_strassen_block(j0, k0, col, U, V, W, n, m, p)
    
    # Convert to numpy for validation
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
        # Fallback to schoolbook
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
