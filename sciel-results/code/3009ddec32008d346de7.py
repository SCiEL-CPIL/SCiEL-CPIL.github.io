import numpy as np

def construct():
    n, m, p = 2, 3, 6
    rank = 8
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to fill columns
    def fill(col, u_idx, v_idx, w_idx):
        for u in u_idx:
            U[u, col] += 1
        for v in v_idx:
            V[v, col] += 1
        for w in w_idx:
            W[w, col] += 1
            
    # P1 = (A00 + A11) * (B0k + B1k)
    # U: A00(0), A11(4)
    # V: B0k(0-5), B1k(6-11)
    # W: C0k(0-5), C1k(6-11)
    fill(0, [0, 4], list(range(0, 6)) + list(range(6, 12)), list(range(0, 6)) + list(range(6, 12)))
    
    # P2 = (A10 + A11) * B0k
    # U: A10(3), A11(4)
    # V: B0k(0-5)
    # W: C1k(6-11)
    fill(1, [3, 4], list(range(0, 6)), list(range(6, 12)))
    
    # P3 = A00 * (B1k - B0k)
    # U: A00(0)
    # V: B1k(6-11), -B0k(0-5)
    # W: C0k(0-5)
    U[0, 2] = 1
    V[list(range(6, 12)), 2] = 1
    V[list(range(0, 6)), 2] = -1
    W[list(range(0, 6)), 2] = 1
    
    # P4 = A11 * (B0k - B1k)
    # U: A11(4)
    # V: B0k(0-5), -B1k(6-11)
    # W: C0k(0-5), C1k(6-11)
    U[4, 3] = 1
    V[list(range(0, 6)), 3] = 1
    V[list(range(6, 12)), 3] = -1
    W[list(range(0, 6)), 3] = 1
    W[list(range(6, 12)), 3] = 1
    
    # P5 = (A00 + A01) * B1k
    # U: A00(0), A01(1)
    # V: B1k(6-11)
    # W: -C0k(0-5), C1k(6-11)
    U[0, 4] = 1
    U[1, 4] = 1
    V[list(range(6, 12)), 4] = 1
    W[list(range(0, 6)), 4] = -1
    W[list(range(6, 12)), 4] = 1
    
    # P6 = (A10 - A00) * (B0k + B1k)
    # U: A10(3), -A00(0)
    # V: B0k(0-5), B1k(6-11)
    # W: C1k(6-11)
    U[3, 5] = 1
    U[0, 5] = -1
    V[list(range(0, 6)), 5] = 1
    V[list(range(6, 12)), 5] = 1
    W[list(range(6, 12)), 5] = 1
    
    # P7 = (A01 - A11) * (B1k + B0k)
    # U: A01(1), -A11(4)
    # V: B1k(6-11), B0k(0-5)
    # W: C0k(0-5)
    U[1, 6] = 1
    U[4, 6] = -1
    V[list(range(6, 12)), 6] = 1
    V[list(range(0, 6)), 6] = 1
    W[list(range(0, 6)), 6] = 1
    
    # P8 = A_i2 * B_2k
    # U: A02(2), A12(5)
    # V: B2k(12-17)
    # W: C0k(0-5), C1k(6-11)
    U[2, 7] = 1
    U[5, 7] = 1
    V[list(range(12, 18)), 7] = 1
    W[list(range(0, 6)), 7] = 1
    W[list(range(6, 12)), 7] = 1

    return {"U": U, "V": V, "W": W}
