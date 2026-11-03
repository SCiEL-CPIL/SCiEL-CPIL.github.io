"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # U shape: (n*m, rank) = (8, 56)
    # V shape: (m*p, rank) = (32, 56)
    # W shape: (p*n, rank) = (16, 56)
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # M_1 .. M_7
    # A_coeffs[m][i][j] where i,j in 0..1
    # B_coeffs[m][j][k] where j,k in 0..1
    # C_coeffs[m][i][k] where i,k in 0..1
    
    A_coeffs = [
        [[1, 0], [0, 1]], # M1: (a00 + a11) -> a00:1, a11:1
        [[0, 0], [1, 1]], # M2: (a10 + a11) -> a10:1, a11:1
        [[1, 0], [0, 0]], # M3: a00
        [[0, 0], [0, 1]], # M4: a11
        [[1, 1], [0, 0]], # M5: (a00 + a01)
        [[-1, 0], [1, 0]],# M6: (a10 - a00)
        [[0, 1], [0, -1]] # M7: (a01 - a11)
    ]
    
    B_coeffs = [
        [[1, 0], [0, 1]], # M1: (b00 + b11)
        [[1, 0], [0, 0]], # M2: b00
        [[0, 0], [1, -1]],# M3: (b10 - b11)
        [[-1, 1], [0, 0]],# M4: (b01 - b00)
        [[0, 0], [0, 1]], # M5: b11
        [[1, 1], [0, 0]], # M6: (b00 + b01)
        [[0, 0], [1, 1]]  # M7: (b10 + b11)
    ]
    
    C_coeffs = [
        [[1, 0], [0, 1]], # M1: c00 += 1, c11 += 1
        [[0, 0], [1, -1]],# M2: c10 += 1, c11 -= 1
        [[0, 1], [0, 1]], # M3: c01 += 1, c11 += 1
        [[1, 0], [1, 0]], # M4: c00 += 1, c10 += 1
        [[-1, 1], [0, 0]],# M5: c00 -= 1, c01 += 1
        [[0, 0], [0, 1]], # M6: c11 += 1
        [[1, 0], [0, 0]]  # M7: c00 += 1
    ]
    
    for g in range(2): # j-groups: 0, 1
        for h in range(4): # k-groups: 0, 1, 2, 3
            for m_idx in range(7):
                c = 7 * (g * 4 + h) + m_idx
                
                # Fill U
                # U row index: i * m + j_global
                # j_global = 2*g + j_local
                for i in range(2):
                    for j_local in range(2):
                        j_global = 2 * g + j_local
                        u_row = i * m + j_global
                        U[u_row][c] = A_coeffs[m_idx][i][j_local]
                        
                # Fill V
                # V row index: j_global * p + k_global
                # j_global = 2*g + j_local
                # k_global = 2*h + k_local
                for j_local in range(2):
                    j_global = 2 * g + j_local
                    for k_local in range(2):
                        k_global = 2 * h + k_local
                        v_row = j_global * p + k_global
                        V[v_row][c] = B_coeffs[m_idx][j_local][k_local]
                        
                # Fill W
                # W row index: k_global * n + i
                # k_global = 2*h + k_local
                for i in range(2):
                    for k_local in range(2):
                        k_global = 2 * h + k_local
                        w_row = k_global * n + i
                        W[w_row][c] += C_coeffs[m_idx][i][k_local]
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
