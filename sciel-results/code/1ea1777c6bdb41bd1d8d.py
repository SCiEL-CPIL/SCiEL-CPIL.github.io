import numpy as np
import time

def construct():
    # Dimensions
    n, m, p = 2, 4, 7
    
    # We will build the decomposition by combining smaller decompositions.
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (28, rank)
    # W shape: (p*n, rank) = (14, rank)
    
    terms = []
    
    def add_term(u, v, w):
        # u, v, w are lists of length 8, 28, 14 respectively
        terms.append((u, v, w))

    # --- Helper: Strassen 2x2 * 2x2 -> 2x2 ---
    # A indices: 0,1,2,3
    # B indices: 0,1,2,3
    # C indices: 0,1,2,3
    def strassen_2x2(A_off, B_off, C_off):
        # A_off, B_off, C_off are offsets into the global indices
        # A = [A00, A01, A10, A11]
        # B = [B00, B01, B10, B11]
        # C = [C00, C01, C10, C11]
        
        # M1 = (A00+A11)(B00+B11)
        u = [0]*8; v = [0]*28; w = [0]*14
        u[A_off] = 1; u[A_off+3] = 1
        v[B_off] = 1; v[B_off+3] = 1
        w[C_off] = 1; w[C_off+3] = 1
        add_term(u, v, w)
        
        # M2 = (A10+A11)B00
        u = [0]*8; v = [0]*28; w = [0]*14
        u[A_off+2] = 1; u[A_off+3] = 1
        v[B_off] = 1
        w[C_off+2] = 1; w[C_off+3] = -1
        add_term(u, v, w)
        
        # M3 = A00(B01-B11)
        u = [0]*8; v = [0]*28; w = [0]*14
        u[A_off] = 1
        v[B_off+1] = 1; v[B_off+3] = -1
        w[C_off+1] = 1; w[C_off+3] = 1
        add_term(u, v, w)
        
        # M4 = A11(B10-B00)
        u = [0]*8; v = [0]*28; w = [0]*14
        u[A_off+3] = 1
        v[B_off+2] = 1; v[B_off] = -1
        w[C_off] = 1; w[C_off+2] = 1
        add_term(u, v, w)
        
        # M5 = (A00+A01)B11
        u = [0]*8; v = [0]*28; w = [0]*14
        u[A_off] = 1; u[A_off+1] = 1
        v[B_off+3] = 1
        w[C_off] = -1; w[C_off+1] = 1
        add_term(u, v, w)
        
        # M6 = (A10-A00)(B00+B01)
        u = [0]*8; v = [0]*28; w = [0]*14
        u[A_off+2] = 1; u[A_off] = -1
        v[B_off] = 1; v[B_off+1] = 1
        w[C_off+3] = 1
        add_term(u, v, w)
        
        # M7 = (A01-A11)(B10+B11)
        u = [0]*8; v = [0]*28; w = [0]*14
        u[A_off+1] = 1; u[A_off+3] = -1
        v[B_off+2] = 1; v[B_off+3] = 1
        w[C_off] = 1; w[C_off+3] = 1
        add_term(u, v, w)

    # --- Helper: Schoolbook 2x2 * 2x1 -> 2x1 ---
    # A indices: 0,1,2,3
    # B indices: 0,1
    # C indices: 0,1
    def schoolbook_2x2_2x1(A_off, B_off, C_off):
        # C0 = A00 B0 + A10 B1
        u = [0]*8; v = [0]*28; w = [0]*14
        u[A_off] = 1
        v[B_off] = 1
        w[C_off] = 1
        add_term(u, v, w)
        
        u = [0]*8; v = [0]*28; w = [0]*14
        u[A_off+2] = 1
        v[B_off+1] = 1
        w[C_off] = 1
        add_term(u, v, w)
        
        # C1 = A01 B0 + A11 B1
        u = [0]*8; v = [0]*28; w = [0]*14
        u[A_off+1] = 1
        v[B_off] = 1
        w[C_off+1] = 1
        add_term(u, v, w)
        
        u = [0]*8; v = [0]*28; w = [0]*14
        u[A_off+3] = 1
        v[B_off+1] = 1
        w[C_off+1] = 1
        add_term(u, v, w)

    # --- Construct 2x2 * 2x4 -> 2x4 ---
    # A: 2x2, B: 2x4, C: 2x4
    # Split B into B1 (2x2) and B2 (2x2)
    def mult_2x2_2x4(A_off, B_off, C_off):
        # B1 is B[0:2], B2 is B[2:4]
        # C1 is C[0:2], C2 is C[2:4]
        strassen_2x2(A_off, B_off, C_off)
        strassen_2x2(A_off, B_off+2, C_off+2)

    # --- Construct 2x2 * 2x3 -> 2x3 ---
    # A: 2x2, B: 2x3, C: 2x3
    # Split B into B1 (2x2) and B2 (2x1)
    def mult_2x2_2x3(A_off, B_off, C_off):
        # B1 is B[0:2], B2 is B[2:3]
        # C1 is C[0:2], C2 is C[2:3]
        strassen_2x2(A_off, B_off, C_off)
        schoolbook_2x2_2x1(A_off, B_off+2, C_off+2)

    # --- Construct 2x2 * 4x4 -> 2x4 ---
    # A: 2x2, B: 4x4, C: 2x4
    # Split B into B_top (2x4) and B_bot (2x4)
    def mult_2x2_4x4(A_off, B_off, C_off):
        # B_top is B[0:2, :], B_bot is B[2:4, :]
        # In flattened B (4x4), B_top is B[0:8], B_bot is B[8:16]
        # Wait, B is 4x4. Flattened size 16.
        # B_top rows 0,1. Cols 0,1,2,3. Indices 0,1,2,3, 4,5,6,7.
        # B_bot rows 2,3. Cols 0,1,2,3. Indices 8,9,10,11, 12,13,14,15.
        # But our V vector is length 28 (4x7).
        # We are calling this helper as part of the main logic.
        # We need to map the local 2x4 B to the global V indices.
        
        # This helper is for local logic. We will call the lower level helpers directly in main.
        pass

    # --- Main Logic ---
    # A is 2x4. Flattened: A00, A01, A10, A11 (for A1) and A02, A03, A12, A13 (for A2).
    # Indices in U (length 8):
    # A1: 0, 1, 2, 3
    # A2: 4, 5, 6, 7
    
    # B is 4x7. Flattened:
    # Row 0: B00, B01, B02, B03, B04, B05, B06 (Indices 0-6)
    # Row 1: B10, B11, B12, B13, B14, B15, B16 (Indices 7-13)
    # Row 2: B20, B21, B22, B23, B24, B25, B26 (Indices 14-20)
    # Row 3: B30, B31, B32, B33, B34, B35, B36 (Indices 21-27)
    
    # C is 2x7. Flattened:
    # Row 0: C00, C01, C02, C03, C04, C05, C06 (Indices 0-6)
    # Row 1: C10, C11, C12, C13, C14, C15, C16 (Indices 7-13)
    
    # C = A1 * B_top + A2 * B_bot
    # B_top is rows 0,1 of B. (2x7)
    # B_bot is rows 2,3 of B. (2x7)
    
    # A1 * B_top:
    # A1 is 2x2. B_top is 2x7.
    # Split B_top into B_top_left (2x4) and B_top_right (2x3).
    # B_top_left: Rows 0,1. Cols 0,1,2,3.
    #   Indices in V:
    #   Row 0 Cols 0-3: 0,1,2,3
    #   Row 1 Cols 0-3: 7,8,9,10
    #   Flattened B_top_left (2x4) for helper:
    #   We need to map these to a contiguous 2x4 block in a virtual buffer?
    #   No, the helpers expect contiguous indices.
    #   We must manually construct the terms or adapt helpers to take index maps.
    
    # Given the complexity of index mapping, it is safer to generate the terms explicitly
    # by calling the low-level Strassen/Schoolbook with correct global indices.
    
    # Let's define a generic function to add a term with specific global indices.
    
    def add_strassen_term(A_idx_map, B_idx_map, C_idx_map, u_coeffs, v_coeffs, w_coeffs):
        u = [0]*8
        v = [0]*28
        w = [0]*14
        for i, c in zip(A_idx_map, u_coeffs):
            if c != 0: u[i] = c
        for i, c in zip(B_idx_map, v_coeffs):
            if c != 0: v[i] = c
        for i, c in zip(C_idx_map, w_coeffs):
            if c != 0: w[i] = c
        add_term(u, v, w)

    # Strassen Coefficients
    # A indices: 0,1,2,3
    # B indices: 0,1,2,3
    # C indices: 0,1,2,3
    
    # M1: A(0,3) B(0,3) C(0,3)
    # M2: A(2,3) B(0) C(2,-1 at 3)
    # M3: A(0) B(1,-1 at 3) C(1,1 at 3)
    # M4: A(3) B(2,-1 at 0) C(1 at 0, 1 at 2)
    # M5: A(0,1) B(3) C(-1 at 0, 1 at 1)
    # M6: A(2,-1 at 0) B(0,1) C(1 at 3)
    # M7: A(1,-1 at 3) B(2,3) C(1 at 0, 1 at 3)
    
    def emit_strassen(A_g_idx, B_g_idx, C_g_idx):
        # A_g_idx: list of 4 global indices for A
        # B_g_idx: list of 4 global indices for B
        # C_g_idx: list of 4 global indices for C
        
        # M1
        add_strassen_term(A_g_idx, B_g_idx, C_g_idx, [1,0,0,1], [1,0,0,1], [1,0,0,1])
        # M2
        add_strassen_term(A_g_idx, B_g_idx, C_g_idx, [0,0,1,1], [1,0,0,0], [0,0,1,-1])
        # M3
        add_strassen_term(A_g_idx, B_g_idx, C_g_idx, [1,0,0,0], [0,1,0,-1], [0,1,0,1])
        # M4
        add_strassen_term(A_g_idx, B_g_idx, C_g_idx, [0,0,0,1], [-1,0,1,0], [1,0,1,0])
        # M5
        add_strassen_term(A_g_idx, B_g_idx, C_g_idx, [1,1,0,0], [0,0,0,1], [-1,1,0,0])
        # M6
        add_strassen_term(A_g_idx, B_g_idx, C_g_idx, [-1,0,1,0], [1,1,0,0], [0,0,0,1])
        # M7
        add_strassen_term(A_g_idx, B_g_idx, C_g_idx, [0,1,0,-1], [0,0,1,1], [1,0,0,1])

    def emit_schoolbook_2x1(A_g_idx, B_g_idx, C_g_idx):
        # A_g_idx: 4 indices
        # B_g_idx: 2 indices
        # C_g_idx: 2 indices
        # C0 = A0 B0 + A2 B1
        # C1 = A1 B0 + A3 B1
        
        # Term 1: A0 B0 -> C0
        u = [0]*8; v = [0]*28; w = [0]*14
        u[A_g_idx[0]] = 1; v[B_g_idx[0]] = 1; w[C_g_idx[0]] = 1
        add_term(u, v, w)
        
        # Term 2: A2 B1 -> C0
        u = [0]*8; v = [0]*28; w = [0]*14
        u[A_g_idx[2]] = 1; v[B_g_idx[1]] = 1; w[C_g_idx[0]] = 1
        add_term(u, v, w)
        
        # Term 3: A1 B0 -> C1
        u = [0]*8; v = [0]*28; w = [0]*14
        u[A_g_idx[1]] = 1; v[B_g_idx[0]] = 1; w[C_g_idx[1]] = 1
        add_term(u, v, w)
        
        # Term 4: A3 B1 -> C1
        u = [0]*8; v = [0]*28; w = [0]*14
        u[A_g_idx[3]] = 1; v[B_g_idx[1]] = 1; w[C_g_idx[1]] = 1
        add_term(u, v, w)

    # Global Indices
    # A: 0..7
    # B: 0..27
    # C: 0..13
    
    # A1: [0, 1, 2, 3]
    # A2: [4, 5, 6, 7]
    
    # B Top (Rows 0,1):
    # B_TL (2x4): Rows 0,1 Cols 0,1,2,3
    #   Indices: 
    #   Row 0: 0, 1, 2, 3
    #   Row 1: 7, 8, 9, 10
    #   Mapped to local 2x4 order: [0,1,2,3, 7,8,9,10]
    
    # B TR (2x3): Rows 0,1 Cols 4,5,6
    #   Indices:
    #   Row 0: 4, 5, 6
    #   Row 1: 11, 12, 13
    #   Mapped to local 2x3 order: [4,5,6, 11,12,13]
    
    # B Bot (Rows 2,3):
    # B_BL (2x4): Rows 2,3 Cols 0,1,2,3
    #   Indices:
    #   Row 2: 14, 15, 16, 17
    #   Row 3: 21, 22, 23, 24
    #   Mapped to local 2x4 order: [14,15,16,17, 21,22,23,24]
    
    # B BR (2x3): Rows 2,3 Cols 4,5,6
    #   Indices:
    #   Row 2: 18, 19, 20
    #   Row 3: 25, 26, 27
    #   Mapped to local 2x3 order: [18,19,20, 25,26,27]
    
    # C:
    # C Left (2x4): Cols 0,1,2,3
    #   Indices:
    #   Row 0: 0, 1, 2, 3
    #   Row 1: 7, 8, 9, 10
    #   Mapped to local 2x4 order: [0,1,2,3, 7,8,9,10]
    
    # C Right (2x3): Cols 4,5,6
    #   Indices:
    #   Row 0: 4, 5, 6
    #   Row 1: 11, 12, 13
    #   Mapped to local 2x3 order: [4,5,6, 11,12,13]
    
    # 1. A1 * B_TL (2x2 * 2x4)
    # A1: [0,1,2,3]
    # B_TL: [0,1,2,3, 7,8,9,10] -> Split into 2x2 and 2x2
    #   B_TL_1: [0,1, 7,8] (Cols 0,1)
    #   B_TL_2: [2,3, 9,10] (Cols 2,3)
    # C_Left: [0,1,2,3, 7,8,9,10] -> Split into 2x2 and 2x2
    #   C_L_1: [0,1, 7,8]
    #   C_L_2: [2,3, 9,10]
    
    emit_strassen([0,1,2,3], [0,1,7,8], [0,1,7,8])
    emit_strassen([0,1,2,3], [2,3,9,10], [2,3,9,10])
    
    # 2. A1 * B_TR (2x2 * 2x3)
    # A1: [0,1,2,3]
    # B_TR: [4,5,6, 11,12,13] -> Split into 2x2 and 2x1
    #   B_TR_1: [4,5, 11,12] (Cols 4,5)
    #   B_TR_2: [6, 13] (Col 6)
    # C_Right: [4,5,6, 11,12,13] -> Split into 2x2 and 2x1
    #   C_R_1: [4,5, 11,12]
    #   C_R_2: [6, 13]
    
    emit_strassen([0,1,2,3], [4,5,11,12], [4,5,11,12])
    emit_schoolbook_2x1([0,1,2,3], [6,13], [6,13])
    
    # 3. A2 * B_BL (2x2 * 2x4)
    # A2: [4,5,6,7]
    # B_BL: [14,15,16,17, 21,22,23,24] -> Split into 2x2 and 2x2
    #   B_BL_1: [14,15, 21,22]
    #   B_BL_2: [16,17, 23,24]
    # C_Left: [0,1,2,3, 7,8,9,10] -> Split into 2x2 and 2x2
    #   C_L_1: [0,1, 7,8]
    #   C_L_2: [2,3, 9,10]
    
    emit_strassen([4,5,6,7], [14,15,21,22], [0,1,7,8])
    emit_strassen([4,5,6,7], [16,17,23,24], [2,3,9,10])
    
    # 4. A2 * B_BR (2x2 * 2x3)
    # A2: [4,5,6,7]
    # B_BR: [18,19,20, 25,26,27] -> Split into 2x2 and 2x1
    #   B_BR_1: [18,19, 25,26]
    #   B_BR_2: [20, 27]
    # C_Right: [4,5,6, 11,12,13] -> Split into 2x2 and 2x1
    #   C_R_1: [4,5, 11,12]
    #   C_R_2: [6, 13]
    
    emit_strassen([4,5,6,7], [18,19,25,26], [4,5,11,12])
    emit_schoolbook_2x1([4,5,6,7], [20,27], [6,13])
    
    rank = len(terms)
    
    U = []
    V = []
    W = []
    for u, v, w in terms:
        U.append(u)
        V.append(v)
        W.append(w)
        
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'])}")
