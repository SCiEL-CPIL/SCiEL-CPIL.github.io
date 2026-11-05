import numpy as np

def construct():
    n, m, p = 2, 4, 8
    rank = 56
    
    # Build the multiplication tensor T of shape (n*m, m*p, p*n) = (8, 32, 16)
    # T[a, b, c] = 1 if a = i*m + j, b = j*p + k, c = k*n + i
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # Use the 2x2x2 Strassen decomposition to build a rank-56 decomposition
    # We decompose the 2x4x8 multiplication into 8 blocks of 2x2x2 (each with rank 7)
    # A is 2x4: split into 2 blocks of 2x2 along columns
    # B is 4x8: split into 4 blocks of 2x2 along rows (and 2 along cols)
    # C is 2x8: split into 4 blocks of 2x2 along columns
    
    # For each pair (j_block, k_block) where j_block in {0,1} and k_block in {0,1,2,3}:
    #   A_block = A[:, 2*j_block : 2*j_block+2]  (2x2)
    #   B_block = B[2*j_block : 2*j_block+2, 2*k_block : 2*k_block+2]  (2x2)
    #   C_block contribution = A_block * B_block  (2x2)
    #   This contributes to C[:, 2*k_block : 2*k_block+2]
    
    # Strassen 7 terms for 2x2x2:
    # P1 = (A00+A11)(B00+B11) -> C00 += P1, C11 += P1
    # P2 = (A10+A11)B00      -> C10 += P2, C11 -= P2
    # P3 = A00(B10-B11)      -> C01 += P3, C11 += P3
    # P4 = A11(B01-B00)      -> C00 += P4, C10 += P4
    # P5 = (A00+A01)B11      -> C00 -= P5, C01 += P5
    # P6 = (A10-A00)(B00+B10)-> C11 += P6
    # P7 = (A01-A11)(B01+B11)-> C00 += P7
    
    # U coefficients for A (2x2): indices (i, j) -> flat i*2+j
    # V coefficients for B (2x2): indices (j, k) -> flat j*2+k
    # W coefficients for C (2x2): indices (i, k) -> flat k*2+i (trace convention)
    
    strassen = [
        # (U_coeffs, V_coeffs, W_coeffs) where each is list of (idx, val)
        # P1
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        # P2
        ([(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)]),
        # P3
        ([(0,1)], [(2,1),(3,-1)], [(1,1),(3,1)]),
        # P4
        ([(3,1)], [(1,1),(0,-1)], [(0,1),(2,1)]),
        # P5
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]),
        # P6
        ([(2,1),(0,-1)], [(0,1),(2,1)], [(3,1)]),
        # P7
        ([(1,1),(3,-1)], [(1,1),(3,1)], [(0,1)]),
    ]
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for jb in range(2):  # j block 0 or 1
        for kb in range(4):  # k block 0,1,2,3
            for u_coeffs, v_coeffs, w_coeffs in strassen:
                # Fill U: A indices (i, j) -> global (i, 2*jb + j)
                for idx, val in u_coeffs:
                    i = idx // 2
                    j_local = idx % 2
                    j_global = 2 * jb + j_local
                    U[i * m + j_global][col] += val
                
                # Fill V: B indices (j, k) -> global (2*jb + j_local, 2*kb + k_local)
                for idx, val in v_coeffs:
                    j_local = idx // 2
                    k_local = idx % 2
                    j_global = 2 * jb + j_local
                    k_global = 2 * kb + k_local
                    V[j_global * p + k_global][col] += val
                
                # Fill W: C indices (i, k) -> global k*n + i
                for idx, val in w_coeffs:
                    i = idx % 2  # In trace convention, flat idx = k*n + i, so i = idx % n
                    k_local = idx // 2
                    k_global = 2 * kb + k_local
                    W[k_global * n + i][col] += val
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
