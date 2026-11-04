import time
import numpy as np

DIMENSIONS = [2, 4, 8]
_START_TIME = time.time()
_DEADLINE = 115.0  # Safety margin before 150s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # The optimal known rank for 2x4x8 using Strassen blocks is 56.
    # Structure: 4 independent 2x4x2 blocks, each of rank 14 (2x2x2 Strassen x 2).
    # Or 8 independent 2x2x2 blocks? No, 2x4x8 = 4 * (2x4x2).
    # Rank(2x4x2) = 14. Total 56.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
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

    col = 0
    
    # We iterate over the 4 blocks of B (size 4x2 each)
    # k0 in {0, 2, 4, 6} corresponds to the start of the 2 columns in B
    for k0 in [0, 2, 4, 6]:
        # We split A into A_left (cols 0,1) and A_right (cols 2,3)
        # This creates two 2x2x2 Strassen multiplications per block.
        
        # --- Part 1: A_left (2x2) * B_block_top (2x2) ---
        # A indices: i in {0,1}, j in {0,1}
        # B indices: j in {0,1}, k in {k0, k0+1}
        # C indices: i in {0,1}, k in {k0, k0+1}
        
        # Strassen terms for 2x2x2
        # M1 = (A00 + A11)(B00 + B11)
        add_term(
            {(0, 0): 1, (1, 1): 1},
            {(0, k0): 1, (1, k0+1): 1},
            {(k0, 0): 1, (k0+1, 1): 1},
            col
        )
        col += 1
        
        # M2 = (A10 + A11)B00
        add_term(
            {(1, 0): 1, (1, 1): 1},
            {(0, k0): 1},
            {(k0, 1): 1, (k0+1, 1): -1},
            col
        )
        col += 1
        
        # M3 = A00(B01 - B11)
        add_term(
            {(0, 0): 1},
            {(0, k0+1): 1, (1, k0+1): -1},
            {(k0+1, 0): 1, (k0+1, 1): 1},
            col
        )
        col += 1
        
        # M4 = A11(B10 - B00)
        add_term(
            {(1, 1): 1},
            {(1, k0): 1, (0, k0): -1},
            {(k0, 0): 1, (k0, 1): 1},
            col
        )
        col += 1
        
        # M5 = (A00 + A01)B11
        add_term(
            {(0, 0): 1, (0, 1): 1},
            {(1, k0+1): 1},
            {(k0, 0): -1, (k0+1, 0): 1},
            col
        )
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        add_term(
            {(1, 0): 1, (0, 0): -1},
            {(0, k0): 1, (0, k0+1): 1},
            {(k0+1, 1): 1},
            col
        )
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        add_term(
            {(0, 1): 1, (1, 1): -1},
            {(1, k0): 1, (1, k0+1): 1},
            {(k0, 0): 1},
            col
        )
        col += 1
        
        # --- Part 2: A_right (2x2) * B_block_bottom (2x2) ---
        # A indices: i in {0,1}, j in {2,3}
        # B indices: j in {2,3}, k in {k0, k0+1}
        # C indices: i in {0,1}, k in {k0, k0+1}
        
        # M1 = (A02 + A13)(B20 + B31)
        add_term(
            {(0, 2): 1, (1, 3): 1},
            {(2, k0): 1, (3, k0+1): 1},
            {(k0, 0): 1, (k0+1, 1): 1},
            col
        )
        col += 1
        
        # M2 = (A12 + A13)B20
        add_term(
            {(1, 2): 1, (1, 3): 1},
            {(2, k0): 1},
            {(k0, 1): 1, (k0+1, 1): -1},
            col
        )
        col += 1
        
        # M3 = A02(B21 - B31)
        add_term(
            {(0, 2): 1},
            {(2, k0+1): 1, (3, k0+1): -1},
            {(k0+1, 0): 1, (k0+1, 1): 1},
            col
        )
        col += 1
        
        # M4 = A13(B30 - B20)
        add_term(
            {(1, 3): 1},
            {(3, k0): 1, (2, k0): -1},
            {(k0, 0): 1, (k0, 1): 1},
            col
        )
        col += 1
        
        # M5 = (A02 + A03)B31
        add_term(
            {(0, 2): 1, (0, 3): 1},
            {(3, k0+1): 1},
            {(k0, 0): -1, (k0+1, 0): 1},
            col
        )
        col += 1
        
        # M6 = (A12 - A02)(B20 + B21)
        add_term(
            {(1, 2): 1, (0, 2): -1},
            {(2, k0): 1, (2, k0+1): 1},
            {(k0+1, 1): 1},
            col
        )
        col += 1
        
        # M7 = (A03 - A13)(B30 + B31)
        add_term(
            {(0, 3): 1, (1, 3): -1},
            {(3, k0): 1, (3, k0+1): 1},
            {(k0, 0): 1},
            col
        )
        col += 1

    # Validation
    if _time_left() < 5.0:
        return {"U": U, "V": V, "W": W}

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
        # Fallback to schoolbook if validation fails (should not happen)
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
