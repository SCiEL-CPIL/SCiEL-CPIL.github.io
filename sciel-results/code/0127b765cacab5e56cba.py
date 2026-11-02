import numpy as np

def construct():
    n, m, p = 2, 3, 10
    rank = 55
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to get indices
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    col = 0
    # Part 1: Strassen for A[:, 0:2] * B[0:2, :]
    for b in range(5):
        # Columns 2b and 2b+1
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # P1: (A00 + A11) * (B0k0 + B1k1)
        U[u_idx(0, 0), col] = 1
        U[u_idx(1, 1), col] = 1
        V[v_idx(0, k0), col] = 1
        V[v_idx(1, k1), col] = 1
        W[w_idx(k0, 0), col] = 1   # R00
        W[w_idx(k1, 1), col] = 1   # R11
        col += 1
        
        # P2: (A10 + A11) * B0k0
        U[u_idx(1, 0), col] = 1
        U[u_idx(1, 1), col] = 1
        V[v_idx(0, k0), col] = 1
        W[w_idx(k0, 1), col] = 1   # R10
        W[w_idx(k1, 1), col] = -1  # R11
        col += 1
        
        # P3: A00 * (B0k1 - B1k1)
        U[u_idx(0, 0), col] = 1
        V[v_idx(0, k1), col] = 1
        V[v_idx(1, k1), col] = -1
        W[w_idx(k1, 0), col] = 1   # R01
        W[w_idx(k1, 1), col] = 1   # R11
        col += 1
        
        # P4: A11 * (B1k0 - B0k0)
        U[u_idx(1, 1), col] = 1
        V[v_idx(1, k0), col] = 1
        V[v_idx(0, k0), col] = -1
        W[w_idx(k0, 0), col] = 1   # R00
        W[w_idx(k0, 1), col] = 1   # R10
        col += 1
        
        # P5: (A00 + A01) * B1k1
        U[u_idx(0, 0), col] = 1
        U[u_idx(0, 1), col] = 1
        V[v_idx(1, k1), col] = 1
        W[w_idx(k0, 0), col] = -1  # R00
        W[w_idx(k1, 0), col] = 1   # R01
        col += 1
        
        # P6: (A10 - A00) * (B0k0 + B0k1)
        U[u_idx(1, 0), col] = 1
        U[u_idx(0, 0), col] = -1
        V[v_idx(0, k0), col] = 1
        V[v_idx(0, k1), col] = 1
        W[w_idx(k1, 1), col] = 1   # R11
        col += 1
        
        # P7: (A01 - A11) * (B1k0 + B1k1)
        U[u_idx(0, 1), col] = 1
        U[u_idx(1, 1), col] = -1
        V[v_idx(1, k0), col] = 1
        V[v_idx(1, k1), col] = 1
        W[w_idx(k0, 0), col] = 1   # R00
        col += 1

    # Part 2: Schoolbook for A[:, 2] * B[2, :]
    for k in range(10):
        for i in range(2):
            U[u_idx(i, 2), col] = 1
            V[v_idx(2, k), col] = 1
            W[w_idx(k, i), col] = 1
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
