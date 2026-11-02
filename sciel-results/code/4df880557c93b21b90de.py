import time
import numpy as np

def construct():
    start_time = time.time()
    n, m, p = 2, 2, 7

    def build_rank23():
        rank = 23
        U = np.zeros((n*m, rank), dtype=int)
        V = np.zeros((m*p, rank), dtype=int)
        W = np.zeros((p*n, rank), dtype=int)

        def add_strassen_block(col_start, k0, k1):
            # Standard Strassen for 2x2x2 covering B columns k0,k1
            # M1 = (A00+A11)*(B0k0+B1k1) -> C0k0, C1k1
            U[0, col_start+0] += 1; U[3, col_start+0] += 1
            V[k0, col_start+0] += 1; V[p+k1, col_start+0] += 1
            W[n*k0, col_start+0] += 1; W[n*k1+1, col_start+0] += 1
            # M2 = (A10+A11)*B0k0 -> C1k0, C1k1
            U[2, col_start+1] += 1; U[3, col_start+1] += 1
            V[k0, col_start+1] += 1
            W[n*k0+1, col_start+1] += 1; W[n*k1+1, col_start+1] -= 1
            # M3 = A00*(B0k1-B1k1) -> C0k1, C1k1
            U[0, col_start+2] += 1
            V[k1, col_start+2] += 1; V[p+k1, col_start+2] -= 1
            W[n*k1, col_start+2] += 1; W[n*k1+1, col_start+2] += 1
            # M4 = A11*(B1k0-B0k0) -> C0k0, C1k0
            U[3, col_start+3] += 1
            V[p+k0, col_start+3] += 1; V[k0, col_start+3] -= 1
            W[n*k0, col_start+3] += 1; W[n*k0+1, col_start+3] += 1
            # M5 = (A00+A01)*B1k1 -> C0k0, C0k1
            U[0, col_start+4] += 1; U[1, col_start+4] += 1
            V[p+k1, col_start+4] += 1
            W[n*k0, col_start+4] -= 1; W[n*k1, col_start+4] += 1
            # M6 = (A10-A00)*(B0k0+B0k1) -> C1k1
            U[2, col_start+5] += 1; U[0, col_start+5] -= 1
            V[k0, col_start+5] += 1; V[k1, col_start+5] += 1
            W[n*k1+1, col_start+5] += 1
            # M7 = (A01-A11)*(B1k0+B1k1) -> C0k0
            U[1, col_start+6] += 1; U[3, col_start+6] -= 1
            V[p+k0, col_start+6] += 1; V[p+k1, col_start+6] += 1
            W[n*k0, col_start+6] += 1

        def add_third_column(col_start, k):
            # C0k = A00*B0k
            U[0, col_start+0] += 1
            V[k, col_start+0] += 1
            W[n*k, col_start+0] += 1
            # C1k = A10*B0k + A11*B1k
            U[2, col_start+1] += 1
            V[k, col_start+1] += 1
            W[n*k+1, col_start+1] += 1
            U[3, col_start+2] += 1
            V[p+k, col_start+2] += 1
            W[n*k+1, col_start+2] += 1

        # Block 1: columns 0,1,2 (10 terms)
        add_strassen_block(0, 0, 1)
        add_third_column(7, 2)
        # Block 2: columns 3,4,5 (10 terms)
        add_strassen_block(10, 3, 4)
        add_third_column(17, 5)
        # Block 3: column 6 (3 terms)
        add_third_column(20, 6)

        return U.tolist(), V.tolist(), W.tolist()

    def verify(U, V, W):
        Ua = np.array(U, dtype=float); Va = np.array(V, dtype=float); Wa = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(len(Ua[0])):
                        s += Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def schoolbook():
        rank = n*m*p
        U = np.zeros((n*m, rank), dtype=int)
        V = np.zeros((m*p, rank), dtype=int)
        W = np.zeros((p*n, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1; V[j*p+k, col] = 1; W[k*n+i, col] = 1
                    col += 1
        return U.tolist(), V.tolist(), W.tolist()

    U, V, W = build_rank23()
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    # Fallback
    U, V, W = schoolbook()
    return {"U": U, "V": V, "W": W}
