"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 3 blocks * 14 terms per block = 42
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y01
    # M3 = X00 Y10
    # M4 = X11 (Y11-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y01)
    # M7 = (X01-X11)(Y10+Y11)
    
    # We define the terms as (U_indices, coeffs, V_indices, coeffs)
    # U_indices are relative to the 2x2 block of A
    # V_indices are relative to the 2x2 block of B
    
    # For A_L (Top-Left 2x2 of A):
    # A00 -> U[0], A01 -> U[1], A10 -> U[4], A11 -> U[5]
    # For A_R (Top-Right 2x2 of A):
    # A00 -> U[2], A01 -> U[3], A10 -> U[6], A11 -> U[7]
    
    # For B_T (Top 2x2 of B block):
    # B00 -> V[row], B01 -> V[row+1], B10 -> V[row+6], B11 -> V[row+7]
    # For B_B (Bottom 2x2 of B block):
    # B00 -> V[row+12], B01 -> V[row+13], B10 -> V[row+18], B11 -> V[row+19]
    
    # Output C block b:
    # C00 -> W[4b], C01 -> W[4b+1], C10 -> W[4b+2], C11 -> W[4b+3]
    
    for b in range(3):
        base_col = b * 14
        
        # --- Top Part (A_L * B_T) ---
        # A_L indices in U: 0, 1, 4, 5
        u_top = [0, 1, 4, 5] # A00, A01, A10, A11
        # B_T indices in V (offset by 2b for columns)
        # B00: 0*6 + 2b = 2b
        # B01: 0*6 + 2b+1 = 2b+1
        # B10: 1*6 + 2b = 6+2b
        # B11: 1*6 + 2b+1 = 6+2b+1
        v_top = [2*b, 2*b+1, 6+2*b, 6+2*b+1]
        
        # Term 1: (A00+A11)(B00+B11)
        c = base_col + 0
        U[u_top[0]][c] += 1; U[u_top[3]][c] += 1
        V[v_top[0]][c] += 1; V[v_top[3]][c] += 1
        
        # Term 2: (A10+A11)B01
        c = base_col + 1
        U[u_top[2]][c] += 1; U[u_top[3]][c] += 1
        V[v_top[1]][c] += 1
        
        # Term 3: A00 B10
        c = base_col + 2
        U[u_top[0]][c] += 1
        V[v_top[2]][c] += 1
        
        # Term 4: A11 (B11-B00)
        c = base_col + 3
        U[u_top[3]][c] += 1
        V[v_top[3]][c] += 1; V[v_top[0]][c] -= 1
        
        # Term 5: (A00+A01)B11
        c = base_col + 4
        U[u_top[0]][c] += 1; U[u_top[1]][c] += 1
        V[v_top[3]][c] += 1
        
        # Term 6: (A10-A00)(B00+B01)
        c = base_col + 5
        U[u_top[2]][c] += 1; U[u_top[0]][c] -= 1
        V[v_top[0]][c] += 1; V[v_top[1]][c] += 1
        
        # Term 7: (A01-A11)(B10+B11)
        c = base_col + 6
        U[u_top[1]][c] += 1; U[u_top[3]][c] -= 1
        V[v_top[2]][c] += 1; V[v_top[3]][c] += 1
        
        # --- Bottom Part (A_R * B_B) ---
        # A_R indices in U: 2, 3, 6, 7
        u_bot = [2, 3, 6, 7]
        # B_B indices in V
        # B00: 2*6 + 2b = 12+2b
        # B01: 2*6 + 2b+1 = 12+2b+1
        # B10: 3*6 + 2b = 18+2b
        # B11: 3*6 + 2b+1 = 18+2b+1
        v_bot = [12+2*b, 12+2*b+1, 18+2*b, 18+2*b+1]
        
        # Term 8 (local 1): (A00+A11)(B00+B11)
        c = base_col + 7
        U[u_bot[0]][c] += 1; U[u_bot[3]][c] += 1
        V[v_bot[0]][c] += 1; V[v_bot[3]][c] += 1
        
        # Term 9 (local 2): (A10+A11)B01
        c = base_col + 8
        U[u_bot[2]][c] += 1; U[u_bot[3]][c] += 1
        V[v_bot[1]][c] += 1
        
        # Term 10 (local 3): A00 B10
        c = base_col + 9
        U[u_bot[0]][c] += 1
        V[v_bot[2]][c] += 1
        
        # Term 11 (local 4): A11 (B11-B00)
        c = base_col + 10
        U[u_bot[3]][c] += 1
        V[v_bot[3]][c] += 1; V[v_bot[0]][c] -= 1
        
        # Term 12 (local 5): (A00+A01)B11
        c = base_col + 11
        U[u_bot[0]][c] += 1; U[u_bot[1]][c] += 1
        V[v_bot[3]][c] += 1
        
        # Term 13 (local 6): (A10-A00)(B00+B01)
        c = base_col + 12
        U[u_bot[2]][c] += 1; U[u_bot[0]][c] -= 1
        V[v_bot[0]][c] += 1; V[v_bot[1]][c] += 1
        
        # Term 14 (local 7): (A01-A11)(B10+B11)
        c = base_col + 13
        U[u_bot[1]][c] += 1; U[u_bot[3]][c] -= 1
        V[v_bot[2]][c] += 1; V[v_bot[3]][c] += 1
        
        # --- W Matrix (Output Reconstruction) ---
        # C00 = M1 + M4 - M5 + M7 + M8 + M11 - M12 + M14
        # C01 = M2 + M5 + M9 + M12
        # C10 = M3 + M6 + M10 + M13
        # C11 = M1 - M2 + M4 + M6 + M8 - M9 + M11 + M13
        
        # W rows for block b:
        # C00 -> W[4b]
        # C01 -> W[4b+1]
        # C10 -> W[4b+2]
        # C11 -> W[4b+3]
        
        w_00 = 4*b
        w_01 = 4*b+1
        w_10 = 4*b+2
        w_11 = 4*b+3
        
        # Indices in global columns
        # Top: 0..6
        # Bot: 7..13
        
        # C00
        W[w_00][base_col + 0] += 1   # M1
        W[w_00][base_col + 3] += 1   # M4
        W[w_00][base_col + 4] -= 1   # -M5
        W[w_00][base_col + 6] += 1   # M7
        W[w_00][base_col + 7] += 1   # M8
        W[w_00][base_col + 10] += 1  # M11
        W[w_00][base_col + 11] -= 1  # -M12
        W[w_00][base_col + 13] += 1  # M14
        
        # C01
        W[w_01][base_col + 1] += 1   # M2
        W[w_01][base_col + 4] += 1   # M5
        W[w_01][base_col + 8] += 1   # M9
        W[w_01][base_col + 11] += 1  # M12
        
        # C10
        W[w_10][base_col + 2] += 1   # M3
        W[w_10][base_col + 5] += 1   # M6
        W[w_10][base_col + 9] += 1   # M10
        W[w_10][base_col + 12] += 1  # M13
        
        # C11
        W[w_11][base_col + 0] += 1   # M1
        W[w_11][base_col + 1] -= 1   # -M2
        W[w_11][base_col + 3] += 1   # M4
        W[w_11][base_col + 5] += 1   # M6
        W[w_11][base_col + 7] += 1   # M8
        W[w_11][base_col + 8] -= 1   # -M9
        W[w_11][base_col + 10] += 1  # M11
        W[w_11][base_col + 12] += 1  # M13

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
