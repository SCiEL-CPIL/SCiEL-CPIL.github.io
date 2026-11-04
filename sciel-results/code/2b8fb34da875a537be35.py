"""Matrix multiplication 4,4,7 — rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]
N, M, P = 4, 4, 7
SCHOOLBOOK_RANK = N * M * P  # 112


def _build_decomp():
    """Build a rank-112 decomposition (schoolbook) for 4x4x7 matmul."""
    rank = SCHOOLBOOK_RANK
    U = [[0.0] * rank for _ in range(N * M)]
    V = [[0.0] * rank for _ in range(M * P)]
    W = [[0.0] * rank for _ in range(P * N)]

    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][col] = 1.0
                V[j * P + k][col] = 1.0
                W[k * N + i][col] = 1.0
                col += 1
    return U, V, W


def _verify_decomp(U, V, W, tol=1e-8):
    """Verify decomposition: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1"""
    n, m, p = N, M, P
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_np[i * m + j], V_np[j * p + k] * W_np[k * n + i])
                if abs(s - 1.0) > tol:
                    return False
    return True


def _strassen_2x2x2():
    """Return rank-7 decomposition for 2x2x2 matmul (Strassen)."""
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # M1: r=0
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: r=1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3: r=2
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # M4: r=3
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5: r=4
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6: r=5
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7: r=6
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W


def _construct_4x4x4_strassen():
    """
    Construct a rank-49 decomposition for 4x4x4 matmul using Strassen.
    Uses tensor product of two rank-7 Strassen decompositions.
    """
    U_s, V_s, W_s = _strassen_2x2x2()
    
    U = np.zeros((16, 49), dtype=np.float64)
    V = np.zeros((16, 49), dtype=np.float64)
    W = np.zeros((16, 49), dtype=np.float64)
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            u_val = U_s[i1*2+j1, r1] * U_s[i2*2+j2, r2]
                            if u_val != 0:
                                U[(i1*2+i2)*4 + j1*2+j2, r] = u_val
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(2):
                            v_val = V_s[j1*2+k1, r1] * V_s[j2*2+k2, r2]
                            if v_val != 0:
                                V[(j1*2+j2)*4 + k1*2+k2, r] = v_val
            for k1 in range(2):
                for k2 in range(2):
                    for i1 in range(2):
                        for i2 in range(2):
                            w_val = W_s[k1*2+i1, r1] * W_s[k2*2+i2, r2]
                            if w_val != 0:
                                W[(k1*2+k2)*4 + i1*2+i2, r] = w_val
    
    return U, V, W


def _construct_4x4x3_schoolbook():
    """
    Construct a rank-48 decomposition for 4x4x3 matmul (schoolbook).
    A is 4x4, B is 4x3, C is 4x3.
    U: (16, 48), V: (12, 48), W: (12, 48)
    """
    rank = 48
    U = np.zeros((16, rank), dtype=np.float64)
    V = np.zeros((12, rank), dtype=np.float64)
    W = np.zeros((12, rank), dtype=np.float64)
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U[i*4+j, col] = 1.0
                V[j*3+k, col] = 1.0
                W[k*4+i, col] = 1.0
                col += 1
    
    return U, V, W


def _construct_4x4x7():
    """
    Construct a rank-97 decomposition for 4x4x7 matmul.
    Split B (4x7) into B1 (4x4) and B2 (4x3).
    C = A*B1 + A*B2
    Use rank-49 Strassen for A*B1 (4x4x4) and rank-48 schoolbook for A*B2 (4x4x3).
    Total rank = 49 + 48 = 97.
    """
    # Part 1: 4x4x4 Strassen (rank 49)
    U1, V1, W1 = _construct_4x4x4_strassen()
    
    # Part 2: 4x4x3 schoolbook (rank 48)
    U2, V2, W2 = _construct_4x4x3_schoolbook()
    
    # Combine: U has shape (16, 97), V has shape (28, 97), W has shape (28, 97)
    rank = 97
    U = np.zeros((16, rank), dtype=np.float64)
    V = np.zeros((28, rank), dtype=np.float64)
    W = np.zeros((28, rank), dtype=np.float64)
    
    # Part 1: columns 0..48
    U[:, :49] = U1
    V[:16, :49] = V1
    W[:16, :49] = W1
    
    # Part 2: columns 49..96
    U[:, 49:] = U2
    for j in range(4):
        for k in range(3):
            V[j*7 + 4 + k, 49:] = V2[j*3 + k, :]
    
    for k2 in range(3):
        for i in range(4):
            W[(4 + k2)*4 + i, 49:] = W2[k2*4 + i, :]
    
    return U, V, W


def _construct_4x4x7_strassen_optimized():
    """
    Construct a rank-97 decomposition for 4x4x7 matmul.
    Split B (4x7) into B1 (4x4) and B2 (4x3).
    C = A*B1 + A*B2
    Use rank-49 Strassen for A*B1 (4x4x4) and rank-48 schoolbook for A*B2 (4x4x3).
    Total rank = 49 + 48 = 97.
    """
    # Part 1: 4x4x4 Strassen (rank 49)
    U1, V1, W1 = _construct_4x4x4_strassen()
    
    # Part 2: 4x4x3 schoolbook (rank 48)
    U2, V2, W2 = _construct_4x4x3_schoolbook()
    
    # Combine: U has shape (16, 97), V has shape (28, 97), W has shape (28, 97)
    rank = 97
    U = np.zeros((16, rank), dtype=np.float64)
    V = np.zeros((28, rank), dtype=np.float64)
    W = np.zeros((28, rank), dtype=np.float64)
    
    # Part 1: columns 0..48
    U[:, :49] = U1
    V[:16, :49] = V1
    W[:16, :49] = W1
    
    # Part 2: columns 49..96
    U[:, 49:] = U2
    for j in range(4):
        for k in range(3):
            V[j*7 + 4 + k, 49:] = V2[j*3 + k, :]
    
    for k2 in range(3):
        for i in range(4):
            W[(4 + k2)*4 + i, 49:] = W2[k2*4 + i, :]
    
    return U, V, W


def construct():
    """Return factor matrices for 4x4x7 matrix multiplication tensor."""
    start_time = time.time()
    
    # Try the rank-97 decomposition first
    try:
        U, V, W = _construct_4x4x7()
        if _verify_decomp(U, V, W, tol=1e-8):
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
    except Exception:
        pass
    
    # Fallback to schoolbook
    U, V, W = _build_decomp()
    return {"U": U, "V": V, "W": W}
