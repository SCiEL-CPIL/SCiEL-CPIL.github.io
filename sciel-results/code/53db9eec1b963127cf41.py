import numpy as np
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_terms.append(u_vec)
        V_terms.append(v_vec)
        W_terms.append(w_vec)
        
    # Strassen 2x2x2
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}
    # U index: i*2+j
    # V index: j*2+k
    # W index: k*2+i
    
    # Term 1: (A00-A11)(B11-B00)
    # A part: (1,0,0,-1) -> i0j0=1, i1j1=-1
    # B part: (0,-1,1,0) -> j0k0=0, j0k1=-1, j1k0=1, j1k1=0
    # C part: (1,0,0,1) -> k0i0=1, k1i1=1?
    # C00 = M1... C11 = M1...
    # So W: k0i0=1, k1i1=1.
    # W index: k*2+i.
    # k0i0 -> 0*2+0=0.
    # k1i1 -> 1*2+1=3.
    
    # I will define the 7 Strassen terms explicitly.
    
    # T1
    u = [1, 0, 0, -1] # A00, A01, A10, A11
    v = [0, -1, 1, 0] # B00, B01, B10, B11
    w = [1, 0, 0, 1]   # C00, C01, C10, C11 (k0i0, k0i1, k1i0, k1i1)
    # Wait, W index is k*2+i.
    # C00 (k0i0) -> 0
    # C11 (k1i1) -> 3
    # So w = [1, 0, 0, 1] is correct for indices 0 and 3.
    
    # I will just list the 7 vectors.
    
    strassen_terms = [
        ([1, 0, 0, -1], [0, -1, 1, 0], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]), # M2: (A10+A11)B00 -> C10
        # C10 is k0i1 -> index 1.
        # W: k0i1=1.
        # So w = [0, 1, 0, 0].
        # Let's re-verify M2.
        # M2 = (A21+A22)B11. In 0-index: (A10+A11)B00.
        # C21 = M2... In 0-index C10 = M2.
        # So W index 1 is 1.
        
        # I will use a known list of 7 Strassen terms in (u,v,w) format.
    ]
    
    # To be safe, I will use the schoolbook for 2x2x2 if I'm not sure?
    # No, 8 terms for 2x2x2 -> 2x4x8 would be 2*8*4 = 64?
    # 2x4x8 schoolbook is 64.
    # 1x4x8 schoolbook is 32.
    # Total 96.
    # So I MUST use Strassen.
    
    # I will hardcode the 7 Strassen terms carefully.
    
    # 1. M1 = (A00-A11)(B11-B00) -> C00, C11
    # u: A00=1, A11=-1. [1,0,0,-1]
    # v: B11=1, B00=-1. [0,0,-1,1] (B00, B01, B10, B11)
    # w: C00=1, C11=1. [1,0,0,1] (C00, C01, C10, C11)
    
    # 2. M2 = (A10+A11)B00 -> C10
    # u: A10=1, A11=1. [0,0,1,1]
    # v: B00=1. [1,0,0,0]
    # w: C10=1. [0,1,0,0]
    
    # 3. M3 = A00(B01-B11) -> C01
    # u: A00=1. [1,0,0,0]
    # v: B01=1, B11=-1. [0,1,0,-1]
    # w: C01=1. [0,1,0,0]
    
    # 4. M4 = A11(B00-B01) -> C00, C10
    # u: A11=1. [0,0,0,1]
    # v: B00=1, B01=-1. [1,-1,0,0]
    # w: C00=1, C10=1. [1,1,0,0]
    
    # 5. M5 = (A00+A01)B11 -> C00, C01
    # u: A00=1, A01=1. [1,1,0,0]
    # v: B11=1. [0,0,0,1]
    # w: C00=1, C01=1. [1,1,0,0]
    
    # 6. M6 = (A10-A00)(B00+B01) -> C11
    # u: A10=1, A00=-1. [-1,0,1,0]
    # v: B00=1, B01=1. [1,1,0,0]
    # w: C11=1. [0,0,0,1]
    
    # 7. M7 = (A01-A11)(B10+B11) -> C00, C11
    # u: A01=1, A11=-1. [0,1,0,-1]
    # v: B10=1, B11=1. [0,0,1,1]
    # w: C00=1, C11=1. [1,0,0,1]
    
    strassen = [
        ([1,0,0,-1], [0,0,-1,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,1,0,0]),
        ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),
        ([0,0,0,1], [1,-1,0,0], [1,1,0,0]),
        ([1,1,0,0], [0,0,0,1], [1,1,0,0]),
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
        ([0,1,0,-1], [0,0,1,1], [1,0,0,1])
    ]
    
    # Helper to expand 2x2x2 terms to larger dimensions
    # We will build 2x4x8 using 2x2x2 blocks.
    
    # 2x4x8
    # A is 2x4. Split into A1 (2x2) cols 0,1 and A2 (2x2) cols 2,3.
    # B is 4x8. Split into B11 (2x4) rows 0,1 and B21 (2x4) rows 2,3?
    # No, B is 4x8.
    # C = A B.
    # C = A1 B_top + A2 B_bottom.
    # B_top is 2x8. B_bottom is 2x8.
    # A1 B_top is 2x2 * 2x8 = 2x8.
    # This is 4 blocks of 2x2x2.
    
    # I will generate the 56 terms for 2x4x8.
    # For each block (i_block in 0,1 for A rows? No, A is 2 rows).
    # A rows 0,1.
    # A cols 0,1 (A1) and 2,3 (A2).
    # B rows 0,1 (B1) and 2,3 (B2).
    # B cols 0..7. Split into 4 blocks of 2 cols.
    
    # This is getting complex to hardcode manually.
    # I will write a helper to generate the terms.
    
    U_248 = []
    V_248 = []
    W_248 = []
    
    # A indices: i in {0,1}, j in {0,1,2,3}
    # B indices: j in {0,1,2,3}, k in {0..7}
    # C indices: i in {0,1}, k in {0..7}
    
    # Block structure:
    # A is split into A_left (cols 0,1) and A_right (cols 2,3).
    # B is split into B_top (rows 0,1) and B_bottom (rows 2,3).
    # B is also split into 4 col blocks of size 2.
    
    # Term generation:
    # For each A part (left/right), B part (top/bottom), B col block (0-3)
    # We have a 2x2x2 multiplication.
    
    # A_left (2x2) * B_top_left (2x2) -> C_left (2x2)
    # A_left (2x2) * B_top_right (2x2) -> C_right (2x2)
    # A_right (2x2) * B_bottom_left (2x2) -> C_left (2x2)
    # A_right (2x2) * B_bottom_right (2x2) -> C_right (2x2)
    
    # Wait, C = A B.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # j=0,1 use A_left and B_top.
    # j=2,3 use A_right and B_bottom.
    
    # So:
    # 1. A_left * B_top (cols 0,1) -> C (cols 0,1)
    # 2. A_left * B_top (cols 2,3) -> C (cols 2,3)
    # 3. A_left * B_top (cols 4,5) -> C (cols 4,5)
    # 4. A_left * B_top (cols 6,7) -> C (cols 6,7)
    # 5. A_right * B_bottom (cols 0,1) -> C (cols 0,1)
    # ...
    
    # Total 8 blocks of 2x2x2.
    # 8 * 7 = 56 terms.
    
    for a_part in [0, 1]: # 0: left (j=0,1), 1: right (j=2,3)
        for b_row_part in [0, 1]: # 0: top (j=0,1), 1: bottom (j=2,3)
            if (a_part == 0 and b_row_part == 1) or (a_part == 1 and b_row_part == 0):
                continue # A_left uses B_top, A_right uses B_bottom
                
            for b_col_block in range(4): # k blocks: 0-1, 2-3, 4-5, 6-7
                # Map to 2x2x2
                # A indices: i in {0,1}, j_local in {0,1}
                # B indices: j_local in {0,1}, k_local in {0,1}
                # C indices: i in {0,1}, k_local in {0,1}
                
                for u, v, w in strassen:
                    U_new = [0] * 8 # 2x4
                    V_new = [0] * 32 # 4x8
                    W_new = [0] * 16 # 8x2 (k*2+i)
                    
                    for i in range(2):
                        for j_local in range(2):
                            j_global = a_part * 2 + j_local
                            U_new[i * 4 + j_global] = u[i * 2 + j_local]
                            
                    for j_local in range(2):
                        for k_local in range(2):
                            j_global = b_row_part * 2 + j_local
                            k_global = b_col_block * 2 + k_local
                            V_new[j_global * 8 + k_global] = v[j_local * 2 + k_local]
                            
                    for k_local in range(2):
                        for i in range(2):
                            k_global = b_col_block * 2 + k_local
                            W_new[k_global * 2 + i] = w[k_local * 2 + i]
                            
                    U_248.append(U_new)
                    V_248.append(V_new)
                    W_248.append(W_new)
                    
    # Now 1x4x8 (32 terms)
    U_148 = []
    V_148 = []
    W_148 = []
    
    for i in range(1): # i=0
        for j in range(4):
            for k in range(8):
                U_new = [0] * 12 # 3x4
                V_new = [0] * 32 # 4x8
                W_new = [0] * 24 # 8x3
                
                U_new[i * 4 + j] = 1
                V_new[j * 8 + k] = 1
                W_new[k * 3 + i] = 1
                
                U_148.append(U_new)
                V_148.append(V_new)
                W_148.append(W_new)
                
    # Combine
    # U is 12x88, V is 32x88, W is 24x88
    
    U_final = np.zeros((12, 88))
    V_final = np.zeros((32, 88))
    W_final = np.zeros((24, 88))
    
    for idx in range(56):
        U_final[:, idx] = U_248[idx]
        V_final[:, idx] = V_248[idx]
        W_final[:, idx] = W_248[idx]
        
    for idx in range(32):
        U_final[:, 56 + idx] = U_148[idx]
        V_final[:, 56 + idx] = V_148[idx]
        W_final[:, 56 + idx] = W_148[idx]
        
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
# EVOLVE-BLOCK-END
