import numpy as np

def construct():
    n, m, p = 3, 5, 7
    
    # Total rank estimate: 6 * 7 (Strassen) + (105 - 6 * 8) (Schoolbook) = 42 + 57 = 99
    R = 99
    
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    # Helper to get linear indices
    def idx_A(i, j): return i * m + j
    def idx_B(j, k): return j * p + k
    def idx_C(k, i): return k * n + i
    
    col = 0
    
    # Define the 6 disjoint 2x2x2 blocks
    # i in {0, 1}, j in {0, 1} or {2, 3}, k in {0, 1}, {2, 3}, {4, 5}
    blocks = []
    for i0 in [0]:
        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                blocks.append((i0, j0, k0))
                
    # Strassen coefficients for a 2x2x2 block
    # Each entry is (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict {(i,j): coeff}
    # V_coeffs: dict {(j,k): coeff}
    # W_coeffs: dict {(k,i): coeff}
    
    strassen_terms = [
        # M1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}),
        # M3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5
        ({(0,0): 1, (1,0): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6
        ({(0,1): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}, {(1,1): 1}),
        # M7
        ({(1,0): 1, (1,1): -1}, {(1,1): 1, (0,1): 1}, {(0,0): 1}),
    ]
    
    # Track which (i,j,k) triples are covered by Strassen
    covered = set()
    
    for i0, j0, k0 in blocks:
        for u_dict, v_dict, w_dict in strassen_terms:
            for (di, dj), cu in u_dict.items():
                i, j = i0 + di, j0 + dj
                U[idx_A(i, j), col] += cu
            for (dj, dk), cv in v_dict.items():
                j, k = j0 + dj, k0 + dk
                V[idx_B(j, k), col] += cv
            for (dk, di), cw in w_dict.items():
                k, i = k0 + dk, i0 + di
                W[idx_C(k, i), col] += cw
            col += 1
            
        # Mark covered triples
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0+di, j0+dj, k0+dk))
                    
    # Add schoolbook terms for the rest
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[idx_A(i, j), col] = 1
                    V[idx_B(j, k), col] = 1
                    W[idx_C(k, i), col] = 1
                    col += 1
                    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
