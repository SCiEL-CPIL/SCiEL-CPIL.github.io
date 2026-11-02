import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 148

    n, m, p = DIMENSIONS
    N, M, P = n * m, m * p, p * n

    def matmul_tensor():
        T = np.zeros((N, M, P), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1.0
        return T

    T_target = matmul_tensor()

    def check_decomp(U, V, W):
        T = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.max(np.abs(T - T_target))

    RANK = 50
    U = [np.zeros(RANK) for _ in range(N)]
    V = [np.zeros(RANK) for _ in range(M)]
    W = [np.zeros(RANK) for _ in range(P)]
    col = 0

    def add_strassen_222(i0, j0, k0):
        nonlocal col
        if col + 7 > RANK:
            return False
        
        A00 = i0 * m + j0
        A01 = i0 * m + (j0 + 1)
        A10 = (i0 + 1) * m + j0
        A11 = (i0 + 1) * m + (j0 + 1)
        
        B00 = j0 * p + k0
        B01 = j0 * p + (k0 + 1)
        B10 = (j0 + 1) * p + k0
        B11 = (j0 + 1) * p + (k0 + 1)
        
        C00 = k0 * n + i0
        C01 = (k0 + 1) * n + i0
        C10 = k0 * n + (i0 + 1)
        C11 = (k0 + 1) * n + (i0 + 1)

        # M1 = (A00+A11)(B00+B11)
        U[A00][col] += 1; U[A11][col] += 1
        V[B00][col] += 1; V[B11][col] += 1
        W[C00][col] += 1; W[C11][col] += 1
        col += 1

        # M2 = (A10+A11)B00
        U[A10][col] += 1; U[A11][col] += 1
        V[B00][col] += 1
        W[C10][col] += 1; W[C11][col] -= 1
        col += 1

        # M3 = A01(B11-B00)
        U[A01][col] += 1
        V[B11][col] += 1; V[B00][col] -= 1
        W[C01][col] += 1; W[C11][col] += 1
        col += 1

        # M4 = A11(B01-B10)  <-- Corrected sign
        U[A11][col] += 1
        V[B01][col] += 1; V[B10][col] -= 1
        W[C00][col] -= 1; W[C10][col] += 1
        col += 1

        # M5 = (A00-A11)(B01+B11)
        U[A00][col] += 1; U[A11][col] -= 1
        V[B01][col] += 1; V[B11][col] += 1
        W[C00][col] -= 1; W[C01][col] += 1
        col += 1

        # M6 = (A10-A00)(B00+B01)
        U[A10][col] += 1; U[A00][col] -= 1
        V[B00][col] += 1; V[B01][col] += 1
        W[C11][col] += 1
        col += 1

        # M7 = (A01-A11)(B10+B11)
        U[A01][col] += 1; U[A11][col] -= 1
        V[B10][col] += 1; V[B11][col] += 1
        W[C00][col] += 1
        col += 1
        return True

    def add_schoolbook_221(i0, j0, k0):
        nonlocal col
        if col + 4 > RANK:
            return False
        A00 = i0 * m + j0
        A01 = i0 * m + (j0 + 1)
        A10 = (i0 + 1) * m + j0
        A11 = (i0 + 1) * m + (j0 + 1)
        B0 = j0 * p + k0
        B1 = (j0 + 1) * p + k0
        C0 = k0 * n + i0
        C1 = k0 * n + (i0 + 1)

        # C0 = A00*B0 + A01*B1
        U[A00][col] += 1; V[B0][col] += 1; W[C0][col] += 1
        col += 1
        U[A01][col] += 1; V[B1][col] += 1; W[C0][col] += 1
        col += 1
        # C1 = A10*B0 + A11*B1
        U[A10][col] += 1; V[B0][col] += 1; W[C1][col] += 1
        col += 1
        U[A11][col] += 1; V[B1][col] += 1; W[C1][col] += 1
        col += 1
        return True

    # Construct
    for j0 in [0, 2]:
        for k0 in [0, 2, 4]:
            add_strassen_222(0, j0, k0)
        add_schoolbook_221(0, j0, 6)

    # Verify
    err = check_decomp(U, V, W)
    if err > 1e-10:
        # Fallback to pure schoolbook rank 56
        RANK56 = 56
        U56 = [np.zeros(RANK56) for _ in range(N)]
        V56 = [np.zeros(RANK56) for _ in range(M)]
        W56 = [np.zeros(RANK56) for _ in range(P)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    cc = k * n + i
                    U56[a][c] = 1.0
                    V56[b][c] = 1.0
                    W56[cc][c] = 1.0
                    c += 1
        U_out = [[float(x) for x in row] for row in U56]
        V_out = [[float(x) for x in row] for row in V56]
        W_out = [[float(x) for x in row] for row in W56]
        return {"U": U_out, "V": V_out, "W": W_out}

    U_out = [[float(x) for x in row] for row in U]
    V_out = [[float(x) for x in row] for row in V]
    W_out = [[float(x) for x in row] for row in W]
    
    return {"U": U_out, "V": V_out, "W": W_out}
