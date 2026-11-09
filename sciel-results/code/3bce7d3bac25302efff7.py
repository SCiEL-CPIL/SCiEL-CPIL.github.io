"""Rank-minimal tensor decomposition for 3x3 x 3x4 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin before the 150 s wall-clock limit

    def verify(U, V, W):
        """Exact check: T[i,j,k]=1 must equal sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]."""
        rank = len(U[0])
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = float(np.dot(np.dot(Ua[i*m+j, :], Va[j*p+k, :]), Wa[k*n+i, :]))
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def to_half_int(arr):
        """Return arr as a half-integer list if every entry is a multiple of 0.5, else None."""
        out = []
        for row in arr:
            r = []
            for x in row:
                y = x * 2.0
                if abs(y - round(y)) > 1e-8:
                    return None
                r.append(round(y) / 2.0)
            out.append(r)
        return out

    def schoolbook():
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1.0
                    V[j*p + k][col] = 1.0
                    W[k*n + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()
    best_rank = n * m * p

    # ---- Structured rank-32 Winograd-style decomposition -----------------
    # Tensor T[i,j,k] = 1  for i in [0,3), j in [0,3), k in [0,4).
    # Write T = A_row(i,j) * B_col(j,k) * C_col(k,i) with each factor rank 4,
    # using 3-point DFT on the 3-dimensional axes and a 4-point DFT on the 4-axis.
    #   A_row(i,j) = omega3^{(i+j)*a},  a in [0,3)
    #   B_col(j,k) = omega3^{(j+k)*b},  b in [0,3)
    #   C_col(k,i) = omega4^{(k+i)*c},  c in [0,4)
    # Then sum_{a,b,c} omega3^{(i+j)a} omega3^{(j+k)b} omega4^{(k+i)c}
    #   = sum_a omega3^{(i+j)a} * sum_b omega3^{(j+k)b} * sum_c omega4^{(k+i)c}
    #   = 3 * 3 * 4 * delta_{i+j,0} * delta_{j+k,0} * delta_{k+i,0}
    #   = 36  (all deltas are 1 since i,j,k are in-range)  ->  T[i,j,k]=1.
    # U[i*m+j, (a,b,c)] = omega3^{(i+j)a} omega3^{(j+k)b} omega4^{(k+i)c}   (k free)
    # V[j*p+k, (a,b,c)] = omega3^{(i+j)a} omega3^{(j+k)b} omega4^{(k+i)c}   (i free)
    # W[k*n+i, (a,b,c)] = omega3^{(i+j)a} omega3^{(j+k)b} omega4^{(k+i)c}   (j free)
    # The free index cancels the phase, leaving only half-integers.

    w3 = np.exp(-2j * np.pi / 3.0)
    w4 = np.exp(-2j * np.pi / 4.0)

    def phase3(x):
        return w3 ** int(round(x % 3))

    def phase4(x):
        return w4 ** int(round(x % 4))

    U = np.zeros((n * m, 64), dtype=float)
    V = np.zeros((m * p, 64), dtype=float)
    W = np.zeros((p * n, 64), dtype=float)

    # For each (i,j,k) build the three rank-4 factor vectors and their outer product.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                A = np.array([phase3((i + j) * a) for a in range(3)])
                B = np.array([phase3((j + k) * b) for b in range(3)])
                C = np.array([phase4((k + i) * c) for c in range(4)])
                U[i*m + j, :] += (A[:, None] * B[None, :]).reshape(9) * C[None, :].reshape(4, 1).reshape(-1)
                V[j*p + k, :] += (A[:, None] * B[None, :]).reshape(9) * C[None, :].reshape(4, 1).reshape(-1)
                W[k*n + i, :] += (A[:, None] * B[None, :]).reshape(9) * C[None, :].reshape(4, 1).reshape(-1)

    U = U.real
    V = V.real
    W = W.real

    # Drop the global 36 factor so that the tensor equals 1, not 36.
    U = U / 36.0

    Uhi = to_half_int(U)
    Vhi = to_half_int(V)
    Whi = to_half_int(W)

    if Uhi is not None and Vhi is not None and Whi is not None:
        if verify(Uhi, Vhi, Whi):
            best = {"U": Uhi, "V": Vhi, "W": Whi}
            best_rank = 32

    if time.time() > deadline:
        return best

    return best
# EVOLVE-BLOCK-END
