import numpy as np
import time

def construct():
    n, m, p = 2, 4, 4
    rank = 28

    # Target tensor T
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                x = i * m + k
                y = k * p + j
                z = i + j * n
                T[x, y, z] = 1.0

    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)

    # Strassen patterns for 2x2 matrix multiplication
    # A = [[a0, a1], [a2, a3]], B = [[b0, b1], [b2, b3]], C = [[c0, c1], [c2, c3]]
    # Indices: 0:TL, 1:TR, 2:BL, 3:BR
    # c0 = M1 + M4 - M5 + M7
    # c1 = M3 + M5
    # c2 = M2 + M4
    # c3 = M1 - M2 + M3 + M6
    
    # Patterns: (A_coeffs, B_coeffs, C_coeffs)
    # Coeffs are lists of 4 elements [TL, TR, BL, BR]
    strassen_patterns = [
        # M1 = (a0+a3)(b0+b3)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2 = (a2+a3)(b0)
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # M3 = (a0)(b1-b3)
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # M4 = (a3)(b2-b0)
        ([0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 1, 0]),
        # M5 = (a0+a1)(b3)
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6 = (a2-a0)(b0+b1)
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7 = (a1-a3)(b2+b3)
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]

    # Define 4 blocks
    # A is 2x4. Split into A1 (cols 0,1) and A2 (cols 2,3).
    # B is 4x4. Split into 4 2x2 blocks.
    # C is 2x4. Split into C1 (cols 0,1) and C2 (cols 2,3).
    
    # Block 1: A1 * B11 -> C1
    # A1: rows 0,1; cols 0,1. Indices in A (row*4+col): 0,1,4,5
    # B11: rows 0,1; cols 0,1. Indices in B (row*4+col): 0,1,4,5
    # C1: rows 0,1; cols 0,1. Indices in C (col*2+row): 0,1,2,3
    
    # Block 2: A2 * B21 -> C1
    # A2: rows 0,1; cols 2,3. Indices in A: 2,3,6,7
    # B21: rows 2,3; cols 0,1. Indices in B: 8,9,12,13
    # C1: rows 0,1; cols 0,1. Indices in C: 0,1,2,3
    
    # Block 3: A1 * B12 -> C2
    # A1: rows 0,1; cols 0,1. Indices in A: 0,1,4,5
    # B12: rows 0,1; cols 2,3. Indices in B: 2,3,6,7
    # C2: rows 0,1; cols 2,3. Indices in C: 4,5,6,7
    
    # Block 4: A2 * B22 -> C2
    # A2: rows 0,1; cols 2,3. Indices in A: 2,3,6,7
    # B22: rows 2,3; cols 2,3. Indices in B: 10,11,14,15
    # C2: rows 0,1; cols 2,3. Indices in C: 4,5,6,7

    # Mapping for local 2x2 indices (0:TL, 1:TR, 2:BL, 3:BR) to global indices
    # A: local (r,c) -> global r*4 + (base_a + c)
    # B: local (r,c) -> global (base_br + r)*4 + (base_bc + c)
    # C: local (r,c) -> global (base_cc + c)*2 + r
    
    blocks = [
        # (base_a, base_br, base_bc, base_cc)
        (0, 0, 0, 0), # Block 1
        (2, 2, 0, 0), # Block 2
        (0, 0, 2, 2), # Block 3
        (2, 2, 2, 2), # Block 4
    ]

    col = 0
    for base_a, base_br, base_bc, base_cc in blocks:
        for a_coeffs, b_coeffs, c_coeffs in strassen_patterns:
            # Fill U
            for local_idx, coeff in enumerate(a_coeffs):
                if coeff != 0:
                    r = local_idx // 2
                    c = local_idx % 2
                    global_idx = r * m + (base_a + c)
                    U[global_idx, col] = coeff
            
            # Fill V
            for local_idx, coeff in enumerate(b_coeffs):
                if coeff != 0:
                    r = local_idx // 2
                    c = local_idx % 2
                    global_idx = (base_br + r) * p + (base_bc + c)
                    V[global_idx, col] = coeff
            
            # Fill W
            for local_idx, coeff in enumerate(c_coeffs):
                if coeff != 0:
                    r = local_idx // 2
                    c = local_idx % 2
                    global_idx = (base_cc + c) * n + r
                    W[global_idx, col] = coeff
            
            col += 1

    # Verification
    def check_solution(U, V, W, rank):
        T_est = np.zeros((8, 16, 8), dtype=float)
        for r in range(rank):
            T_est += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        if not np.allclose(T_est, T, atol=1e-9):
            return False
        return True

    if check_solution(U, V, W, rank):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback if something went wrong (should not happen)
        return {"U": np.zeros((8, 28)).tolist(), "V": np.zeros((16, 28)).tolist(), "W": np.zeros((8, 28)).tolist()}

# Test locally
if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    T_est = np.zeros((8, 16, 8), dtype=float)
    for r in range(28):
        T_est += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
    
    T_true = np.zeros((8, 16, 8), dtype=float)
    for i in range(2):
        for k in range(4):
            for j in range(4):
                x = i * 4 + k
                y = k * 4 + j
                z = i + j * 2
                T_true[x, y, z] = 1.0
                
    print("Max diff:", np.max(np.abs(T_est - T_true)))
    if np.allclose(T_est, T_true, atol=1e-9):
        print("SUCCESS")
    else:
        print("FAILURE")
