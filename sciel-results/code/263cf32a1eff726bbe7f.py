"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # We decompose into 6 blocks of 2x2 multiplication
    # Each block uses Strassen's algorithm (rank 7)
    # Total rank = 6 * 7 = 42
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's algorithm for 2x2 matrix multiplication
    # C = A * B
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U vectors (coefficients for A)
    # Index in U: i * m + j
    # A00 -> 0, A01 -> 1, A10 -> 2, A11 -> 3
    
    U_M1 = [1, 0, 0, 1]
    U_M2 = [0, 0, 1, 1]
    U_M3 = [1, 0, 0, 0]
    U_M4 = [0, 0, 0, 1]
    U_M5 = [1, 1, 0, 0]
    U_M6 = [1, 0, -1, 0]
    U_M7 = [0, 1, 0, -1]
    
    U_strassen = [U_M1, U_M2, U_M3, U_M4, U_M5, U_M6, U_M7]
    
    # V vectors (coefficients for B) for a single 2x2 block
    # Index in V: j * p + k
    # For a block starting at column k_start:
    # B00 -> k_start + 0
    # B01 -> k_start + 1
    # B10 -> k_start + 2
    # B11 -> k_start + 3
    # Wait, B is 2x12.
    # B[j, k].
    # Block i uses B[:, 2*i : 2*i+2].
    # So B00 is B[0, 2*i], B01 is B[0, 2*i+1]
    # B10 is B[1, 2*i], B11 is B[1, 2*i+1]
    
    # V_M1 = B00 + B11 -> [1, 0, 1, 0] at offsets 0, 1, 2, 3?
    # V index: j * 12 + k.
    # B[0, k] is index k.
    # B[1, k] is index 12 + k.
    
    # Let's define V vectors for a block at offset k0
    def get_V(k0):
        # M1: B00 + B11 -> B[0,k0] + B[1,k0+1]
        V_M1 = [0] * (m * p)
        V_M1[k0] = 1
        V_M1[12 + k0 + 1] = 1
        
        # M2: B00 -> B[0,k0]
        V_M2 = [0] * (m * p)
        V_M2[k0] = 1
        
        # M3: B01 - B11 -> B[0,k0+1] - B[1,k0+1]
        V_M3 = [0] * (m * p)
        V_M3[k0 + 1] = 1
        V_M3[12 + k0 + 1] = -1
        
        # M4: B10 - B00 -> B[1,k0] - B[0,k0]
        V_M4 = [0] * (m * p)
        V_M4[12 + k0] = 1
        V_M4[k0] = -1
        
        # M5: B11 -> B[1,k0+1]
        V_M5 = [0] * (m * p)
        V_M5[12 + k0 + 1] = 1
        
        # M6: B00 + B01 -> B[0,k0] + B[0,k0+1]
        V_M6 = [0] * (m * p)
        V_M6[k0] = 1
        V_M6[k0 + 1] = 1
        
        # M7: B10 + B11 -> B[1,k0] + B[1,k0+1]
        V_M7 = [0] * (m * p)
        V_M7[12 + k0] = 1
        V_M7[12 + k0 + 1] = 1
        
        return [V_M1, V_M2, V_M3, V_M4, V_M5, V_M6, V_M7]

    # W vectors (coefficients for C) for a single 2x2 block
    # Index in W: k * n + i
    # C[i, k].
    # Block i uses C[:, 2*i : 2*i+2].
    # C[0, 2*i] is index 2*i * 2 + 0 = 4i
    # C[0, 2*i+1] is index (2*i+1) * 2 + 0 = 4i + 2
    # C[1, 2*i] is index 2*i * 2 + 1 = 4i + 1
    # C[1, 2*i+1] is index (2*i+1) * 2 + 1 = 4i + 3
    
    def get_W(k0):
        # k0 is the starting column index in C (0, 2, 4, 6, 8, 10)
        # C00 -> index k0 * 2 + 0
        # C01 -> index (k0+1) * 2 + 0
        # C10 -> index k0 * 2 + 1
        # C11 -> index (k0+1) * 2 + 1
        
        idx_C00 = k0 * 2 + 0
        idx_C01 = (k0 + 1) * 2 + 0
        idx_C10 = k0 * 2 + 1
        idx_C11 = (k0 + 1) * 2 + 1
        
        # C00 = M1 + M4 - M5 + M7
        W_M1 = [0] * (p * n)
        W_M1[idx_C00] = 1
        W_M1[idx_C01] = 0
        W_M1[idx_C10] = 0
        W_M1[idx_C11] = 0
        # Wait, W vectors are for the OUTPUT of the multiplication M_s.
        # The term is U_s * V_s * W_s.
        # The output is sum of these terms.
        # So W_s should have 1 at the positions where M_s contributes.
        
        # Let's re-read Strassen:
        # C00 = M1 + M4 - M5 + M7
        # So M1 contributes to C00 with coeff 1.
        # M4 contributes to C00 with coeff 1.
        # M5 contributes to C00 with coeff -1.
        # M7 contributes to C00 with coeff 1.
        
        # We need to construct W_s for each s.
        
        W_1 = [0] * (p * n)
        W_1[idx_C00] = 1
        
        W_2 = [0] * (p * n)
        W_2[idx_C10] = 1
        
        W_3 = [0] * (p * n)
        W_3[idx_C01] = 1
        W_3[idx_C11] = 1
        
        W_4 = [0] * (p * n)
        W_4[idx_C00] = 1
        W_4[idx_C10] = 1
        
        W_5 = [0] * (p * n)
        W_5[idx_C00] = -1
        W_5[idx_C01] = 1
        
        W_6 = [0] * (p * n)
        W_6[idx_C11] = 1
        
        W_7 = [0] * (p * n)
        W_7[idx_C00] = 1
        W_7[idx_C11] = 1
        
        return [W_1, W_2, W_3, W_4, W_5, W_6, W_7]

    # Fill U, V, W
    col = 0
    for block in range(6):
        k0 = block * 2
        V_block = get_V(k0)
        W_block = get_W(k0)
        
        for s in range(7):
            U_s = U_strassen[s]
            V_s = V_block[s]
            W_s = W_block[s]
            
            for i in range(n * m):
                U[i][col] = U_s[i]
            for j in range(m * p):
                V[j][col] = V_s[j]
            for k in range(p * n):
                W[k][col] = W_s[k]
            
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
