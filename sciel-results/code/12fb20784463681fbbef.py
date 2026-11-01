import numpy as np
import time

def construct():
    t0 = time.time()
    deadline = t0 + 115.0

    n, m, p = 2, 2, 6
    rank = 21

    # Strassen coefficients for 2x2 matrix multiplication
    # Each entry: U (4 coeffs for A00,A01,A10,A11), V (4 coeffs for B00,B01,B10,B11), W (4 coeffs for C00,C01,C10,C11)
    strassen = [
        {'U': [1, 0, 0, 1], 'V': [1, 0, 0, 1], 'W': [1, 0, 0, 1]},   # P1
        {'U': [0, 0, 1, 1], 'V': [1, 0, 0, 0], 'W': [0, -1, 0, 1]},   # P2
        {'U': [1, 0, 0, 0], 'V': [0, 1, 0, -1], 'W': [0, 1, 0, 0]},   # P3
        {'U': [0, 0, 0, 1], 'V': [-1, 0, 1, 0], 'W': [1, 0, 0, 0]},   # P4
        {'U': [1, 1, 0, 0], 'V': [0, 0, 0, 1], 'W': [-1, 0, 0, 0]},   # P5
        {'U': [-1, 0, 1, 0], 'V': [1, 1, 0, 0], 'W': [0, 0, 1, 0]},   # P6
        {'U': [0, 1, 0, -1], 'V': [0, 0, 1, 1], 'W': [0, 0, 0, 1]},   # P7
    ]

    U = [[0] * rank for _ in range(n * m)]  # 4 rows
    V = [[0] * rank for _ in range(m * p)]  # 12 rows
    W = [[0] * rank for _ in range(p * n)]  # 12 rows

    # Three blocks of 2x2 multiplication for columns 0-1, 2-3, 4-5 of B/C
    for b in range(3):
        for s in range(7):
            col = b * 7 + s
            sc = strassen[s]

            # Fill U: indices 0..3 correspond to A[0,0], A[0,1], A[1,0], A[1,1]
            for idx in range(4):
                if sc['U'][idx] != 0:
                    U[idx][col] = sc['U'][idx]

            # Fill V: m*p = 2*6 = 12 entries, indexed as V[j*6 + k]
            # Local V indices: 0->(j=0,k=2b), 1->(j=0,k=2b+1), 2->(j=1,k=2b), 3->(j=1,k=2b+1)
            v_indices = [2*b, 2*b+1, 6+2*b, 6+2*b+1]
            for idx in range(4):
                if sc['V'][idx] != 0:
                    V[v_indices[idx]][col] = sc['V'][idx]

            # Fill W: p*n = 6*2 = 12 entries, indexed as W[k*2 + i]
            # Local W indices: 0->(k=2b,i=0), 1->(k=2b,i=1), 2->(k=2b+1,i=0), 3->(k=2b+1,i=1)
            w_indices = [2*(2*b), 2*(2*b)+1, 2*(2*b+1), 2*(2*b+1)+1]
            for idx in range(4):
                if sc['W'][idx] != 0:
                    W[w_indices[idx]][col] = sc['W'][idx]

    if time.time() > deadline:
        return None

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    if res:
        print("Rank:", len(res["U"][0]))
        np.random.seed(42)
        A = np.random.randn(2, 2)
        B = np.random.randn(2, 6)
        C_expected = A @ B

        U = np.array(res["U"])
        V = np.array(res["V"])
        W = np.array(res["W"])

        C_reconstructed = np.zeros((2, 6))
        for i in range(2):
            for k in range(6):
                total = 0.0
                for s in range(21):
                    for j in range(2):
                        total += U[i*2+j, s] * V[j*6+k, s] * W[k*2+i, s]
                C_reconstructed[i, k] = total

        print("Max error:", np.max(np.abs(C_expected - C_reconstructed)))
