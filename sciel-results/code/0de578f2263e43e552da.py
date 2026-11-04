"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 63
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # X = [[x00, x01], [x10, x11]]
    # Y = [[y00, y01], [y10, y11]]
    # M_s = (sum cX_s * X) * (sum cY_s * Y)
    # Z_ab += cZ_s * M_s
    
    # s, cX, cY, cZ
    # cX = [c00, c01, c10, c11]
    # cY = [c00, c01, c10, c11]
    # cZ = [c00, c01, c10, c11]
    
    strassen_terms = [
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # M3
        ([1, 0, 0, 0], [0, -1, 1, 0], [0, 1, 0, 1]),
        # M4
        ([0, 0, 0, 1], [-1, 1, 0, 0], [1, 0, 1, 0]),
        # M5
        ([1, 0, 1, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6
        ([-1, 0, 0, 1], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7
        ([0, 1, -1, 0], [0, 1, 0, 1], [1, 0, 0, 1])
    ]
    
    # Mapping from block indices to matrix indices
    # A[i, j] is at index i*m + j
    # B[j, k] is at index j*p + k
    # C[k, i] is at index k*n + i
    
    # We have 3x3 blocks of size 2x2.
    # Block A_ik: rows 2i, 2i+1; cols 2k, 2k+1
    # Block B_kj: rows 2k, 2k+1; cols 2j, 2j+1
    # Block C_ij: rows 2i, 2i+1; cols 2j, 2j+1
    
    # r = ((i*3 + j)*3 + k)*7 + s
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for s in range(7):
                    r = ((i * 3 + j) * 3 + k) * 7 + s
                    cX, cY, cZ = strassen_terms[s]
                    
                    # Fill U
                    # X_ab corresponds to A[2i+a, 2k+b]
                    # Index in U: (2i+a)*m + (2k+b)
                    for a in range(2):
                        for b in range(2):
                            idx_u = (2 * i + a) * m + (2 * k + b)
                            U[idx_u][r] = cX[a * 2 + b]
                            
                    # Fill V
                    # Y_cd corresponds to B[2k+c, 2j+d]
                    # Index in V: (2k+c)*p + (2j+d)
                    for c in range(2):
                        for d in range(2):
                            idx_v = (2 * k + c) * p + (2 * j + d)
                            V[idx_v][r] = cY[c * 2 + d]
                            
                    # Fill W
                    # Z_ab contributes to C[2i+a, 2j+b]
                    # Index in W: (2j+b)*n + (2i+a)  <-- Note: W[k*n+i, r] convention?
                    # Task says: W[k*n+i, r] for C[i, k]?
                    # "output factor follows the official trace convention W[k*n+i,r]"
                    # This usually means C[i, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
                    # So W index is k*n + i.
                    # Here C is 6x6. C[2i+a, 2j+b].
                    # So i_C = 2i+a, k_C = 2j+b.
                    # Index in W: k_C*n + i_C = (2j+b)*n + (2i+a).
                    for a in range(2):
                        for b in range(2):
                            idx_w = (2 * j + b) * n + (2 * i + a)
                            W[idx_w][r] = cZ[a * 2 + b]
                            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
