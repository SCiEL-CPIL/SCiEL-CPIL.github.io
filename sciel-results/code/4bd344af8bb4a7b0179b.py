"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 5

    # ---- Strassen 2x2x2 (rank 7) ----
    # M1=A00+B11, M2=(A01+A11)*B00, M3=A11*(B10-B00), M4=A00*(B01-B11),
    # M5=(A00+A10)*B11, M6=(A10-A00)*(B00+B01), M7=(A01-A11)*(B10+B11)
    # C00=M1+M4-M5+M7, C01=M2+M3-M5-M6, C10=M2+M4+M5-M7, C11=M1-M2+M3+M6
    # (row = input index, col = rank-1 slot)
    def strassen2x2x2():
        # U indexed by (i,j) i in {0,1}, j in {0,1} -> 4 rows
        U = np.zeros((4, 7))
        # A00=(0,0), A01=(0,1), A10=(1,0), A11=(1,1)
        U[0, 0] = 1; U[0, 3] = 1; U[0, 4] = 1; U[0, 5] = -1; U[0, 6] = 1
        U[1, 1] = 1; U[1, 5] = 1; U[1, 6] = -1
        U[2, 4] = 1; U[2, 5] = 1; U[2, 6] = -1
        U[3, 0] = 1; U[3, 1] = 1; U[3, 2] = 1
        # V indexed by (j,k) j in {0,1}, k in {0,1} -> 4 rows
        V = np.zeros((4, 7))
        # B00=(0,0), B01=(0,1), B10=(1,0), B11=(1,1)
        V[0, 0] = 1; V[0, 1] = 1; V[0, 2] = -1; V[0, 6] = 1
        V[1, 3] = 1; V[1, 4] = -1; V[1, 5] = 1
        V[2, 2] = 1; V[2, 6] = 1
        V[3, 0] = 1; V[3, 4] = 1; V[3, 5] = 1
        # W indexed by (k,i) k in {0,1}, i in {0,1} -> 4 rows
        W = np.zeros((4, 7))
        # C00=(0,0), C01=(0,1), C10=(1,0), C11=(1,1)
        W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1
        W[1, 1] = 1; W[1, 2] = 1; W[1, 4] = -1; W[1, 5] = -1
        W[2, 1] = 1; W[2, 3] = 1; W[2, 4] = 1; W[2, 6] = -1
        W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1
        return U, V, W

    # ---- Build rank-43 decomposition via linear embedding ----
    R = 43
    U = np.zeros((n * m, R))   # 9 x R
    V = np.zeros((m * p, R))   # 15 x R
    W = np.zeros((p * n, R))   # 15 x R

    sU, sV, sW = strassen2x2x2()  # each 4 x 7

    def add_submat(mat, r0, c0, sub):
        mat[r0:r0 + sub.shape[0], c0:c0 + sub.shape[1]] = sub

    # Part A: k in {0,1,2}
    #  A1: Strassen core (i in {0,1}, j in {0,1}, k in {0,1}), cols 0..6
    add_submat(U, 0, 0, sU)              # rows (i*m+j) for i,j in {0,1}
    add_submat(V, 0, 0, sV)              # rows (j*p+k) for j,k in {0,1}
    add_submat(W, 0, 0, sW)              # rows (k*n+i) for k,i in {0,1}
    #  A2: k=2 (i in {0,1,2}, j in {0,1,2}), cols 7..15
    for t in range(9):
        i = t // 3; j = t % 3
        col = 7 + t
        U[i * m + j, col] = 1
        V[j * p + 2, col] = 1
        W[2 * n + i, col] = 1
    #  A3: i=2, k in {0,1} (j in {0,1,2}), cols 16..21
    for t in range(6):
        j = t // 2; k = t % 2
        col = 16 + t
        U[2 * m + j, col] = 1
        V[j * p + k, col] = 1
        W[k * n + 2, col] = 1
    #  A4: j=2, i in {0,1}, k in {0,1}, cols 22..25
    for t in range(4):
        i = t // 2; k = t % 2
        col = 22 + t
        U[i * m + 2, col] = 1
        V[2 * p + k, col] = 1
        W[k * n + i, col] = 1
    # Part B: k in {3,4}
    #  B1: Strassen core (i in {0,1}, j in {0,1}, k in {3,4}), cols 26..32
    add_submat(U, 0, 26, sU)
    add_submat(V, 3 * n, 26, sV)          # j*p+k for j in {0,1}, k in {3,4}
    add_submat(W, 3 * n, 26, sW)          # k*n+i for k in {3,4}, i in {0,1}
    #  B2: i=2, k in {3,4} (j in {0,1,2}), cols 33..38
    for t in range(6):
        j = t // 2; k = 3 + (t % 2)
        col = 33 + t
        U[2 * m + j, col] = 1
        V[j * p + k, col] = 1
        W[k * n + 2, col] = 1
    #  B3: j=2, i in {0,1}, k in {3,4}, cols 39..42
    for t in range(4):
        i = t // 2; k = 3 + (t % 2)
        col = 39 + t
        U[i * m + 2, col] = 1
        V[2 * p + k, col] = 1
        W[k * n + i, col] = 1

    # ---- Verify with random matrices (guard against a construction bug) ----
    rng = np.random.default_rng(0)
    for _ in range(20):
        A = rng.integers(-5, 6, size=(n, m)).astype(float)
        B = rng.integers(-5, 6, size=(m, p)).astype(float)
        Ctrue = A @ B
        Ucol = A @ U          # (n,R)  : Ucol[i,r] = sum_j A[i,j] U[i*m+j, r]
        Vcol = B @ V          # (p,R)  : Vcol[k,r] = sum_j B[j,k] V[j*p+k, r]
        Wt = W.T              # (R,p*n)
        C = (Ucol * Vcol) @ Wt
        C = C.reshape(n, p)
        if not np.allclose(C, Ctrue, atol=1e-6):
            # fall back to schoolbook rank 45 (always valid)
            rank = n * m * p
            Us = np.zeros((n * m, rank)); Vs = np.zeros((m * p, rank)); Ws = np.zeros((p * n, rank))
            c = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        Us[i * m + j, c] = 1
                        Vs[j * p + k, c] = 1
                        Ws[k * n + i, c] = 1
                        c += 1
            return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
