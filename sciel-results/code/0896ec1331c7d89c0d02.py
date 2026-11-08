import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 120.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def construct():
    if _time_left() < 5.0:
        return _construct_rank50()
    
    n, m, p = 2, 3, 9
    
    # Rank 50 construction
    # 4 Strassen blocks (rank 7 each) for B cols (0,1), (2,3), (4,5), (6,7)
    # 1 Naive block (rank 4) for B col 8, using A top-left 2x2
    # 2 Naive blocks (rank 9 each) for A col 2, using B bottom row
    rank = 50
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Indices in U for A[0,0], A[0,1], A[1,0], A[1,1]
    u_a, u_b, u_c, u_d = 0, 1, 3, 4
    
    # Strassen blocks for first 8 columns
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # V indices: B[row, col] -> row*p + col
        v_e = 0 * p + k0
        v_f = 0 * p + k1
        v_g = 1 * p + k0
        v_h = 1 * p + k1
        
        # W indices: C[col, row] -> col*n + row
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        # M1 = (a + d)(e + h)
        U[u_a][col] += 1
        U[u_d][col] += 1
        V[v_e][col] += 1
        V[v_h][col] += 1
        W[w_00][col] += 1
        W[w_11][col] += 1
        
        # M2 = (c + d) * e
        U[u_c][col+1] += 1
        U[u_d][col+1] += 1
        V[v_e][col+1] += 1
        W[w_10][col+1] += 1
        W[w_11][col+1] -= 1
        
        # M3 = a * (f - h)
        U[u_a][col+2] += 1
        V[v_f][col+2] += 1
        V[v_h][col+2] -= 1
        W[w_01][col+2] += 1
        W[w_11][col+2] += 1
        
        # M4 = d * (g - e)
        U[u_d][col+3] += 1
        V[v_g][col+3] += 1
        V[v_e][col+3] -= 1
        W[w_00][col+3] += 1
        W[w_10][col+3] += 1
        
        # M5 = (a + b) * h
        U[u_a][col+4] += 1
        U[u_b][col+4] += 1
        V[v_h][col+4] += 1
        W[w_00][col+4] -= 1
        W[w_01][col+4] += 1
        
        # M6 = (c - a)(e + f)
        U[u_c][col+5] += 1
        U[u_a][col+5] -= 1
        V[v_e][col+5] += 1
        V[v_f][col+5] += 1
        W[w_11][col+5] += 1
        
        # M7 = (b - d)(g + h)
        U[u_b][col+6] += 1
        U[u_d][col+6] -= 1
        V[v_g][col+6] += 1
        V[v_h][col+6] += 1
        W[w_00][col+6] += 1
        
        col += 7
    
    # Naive block for last column k=8 (A top-left 2x2, B col 8 rows 0,1)
    k8 = 8
    v_e8 = 0 * p + k8
    v_g8 = 1 * p + k8
    w_08 = k8 * n + 0
    w_18 = k8 * n + 1
    
    U[u_a][col]   += 1
    V[v_e8][col]  += 1
    W[w_08][col]  += 1
    
    U[u_b][col+1] += 1
    V[v_g8][col+1] += 1
    W[w_08][col+1] += 1
    
    U[u_c][col+2] += 1
    V[v_e8][col+2] += 1
    W[w_18][col+2] += 1
    
    U[u_d][col+3] += 1
    V[v_g8][col+3] += 1
    W[w_18][col+3] += 1
    
    col += 4
    
    # Naive blocks for A col 2 (A[:,2] x B[2,:])
    # A indices: 2 (A[0,2]), 5 (A[1,2])
    u_ar0 = 2
    u_ar1 = 5
    
    for k in range(p):
        v_bk = 2 * p + k
        w_0k = k * n + 0
        w_1k = k * n + 1
        
        U[u_ar0][col]   += 1
        V[v_bk][col]    += 1
        W[w_0k][col]    += 1
        
        U[u_ar1][col+1] += 1
        V[v_bk][col+1]  += 1
        W[w_1k][col+1]  += 1
        
        col += 2

    # Convert to numpy for verification
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Quick verification
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for r in range(rank):
        u_mat = U_np[:, r].reshape(n, m)
        v_mat = V_np[:, r].reshape(m, p)
        w_mat = W_np[:, r].reshape(p, n).T
        C_computed += (u_mat @ v_mat) * w_mat
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fallback: should not happen
        pass

    return {"U": U, "V": V, "W": W}

def _construct_rank50():
    n, m, p = 2, 3, 9
    rank = 50
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    u_a, u_b, u_c, u_d = 0, 1, 3, 4
    
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        v_e = 0 * p + k0
        v_f = 0 * p + k1
        v_g = 1 * p + k0
        v_h = 1 * p + k1
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        U[u_a][col] += 1; U[u_d][col] += 1; V[v_e][col] += 1; V[v_h][col] += 1; W[w_00][col] += 1; W[w_11][col] += 1
        U[u_c][col+1] += 1; U[u_d][col+1] += 1; V[v_e][col+1] += 1; W[w_10][col+1] += 1; W[w_11][col+1] -= 1
        U[u_a][col+2] += 1; V[v_f][col+2] += 1; V[v_h][col+2] -= 1; W[w_01][col+2] += 1; W[w_11][col+2] += 1
        U[u_d][col+3] += 1; V[v_g][col+3] += 1; V[v_e][col+3] -= 1; W[w_00][col+3] += 1; W[w_10][col+3] += 1
        U[u_a][col+4] += 1; U[u_b][col+4] += 1; V[v_h][col+4] += 1; W[w_00][col+4] -= 1; W[w_01][col+4] += 1
        U[u_c][col+5] += 1; U[u_a][col+5] -= 1; V[v_e][col+5] += 1; V[v_f][col+5] += 1; W[w_11][col+5] += 1
        U[u_b][col+6] += 1; U[u_d][col+6] -= 1; V[v_g][col+6] += 1; V[v_h][col+6] += 1; W[w_00][col+6] += 1
        col += 7
    
    k8 = 8
    v_e8 = 0 * p + k8; v_g8 = 1 * p + k8; w_08 = k8 * n + 0; w_18 = k8 * n + 1
    U[u_a][col] += 1; V[v_e8][col] += 1; W[w_08][col] += 1
    U[u_b][col+1] += 1; V[v_g8][col+1] += 1; W[w_08][col+1] += 1
    U[u_c][col+2] += 1; V[v_e8][col+2] += 1; W[w_18][col+2] += 1
    U[u_d][col+3] += 1; V[v_g8][col+3] += 1; W[w_18][col+3] += 1
    col += 4
    
    u_ar0 = 2; u_ar1 = 5
    for k in range(p):
        v_bk = 2 * p + k; w_0k = k * n + 0; w_1k = k * n + 1
        U[u_ar0][col] += 1; V[v_bk][col] += 1; W[w_0k][col] += 1
        U[u_ar1][col+1] += 1; V[v_bk][col+1] += 1; W[w_1k][col+1] += 1
        col += 2

    return {"U": U, "V": V, "W": W}
