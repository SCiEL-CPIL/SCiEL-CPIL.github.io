import time
import sys

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=6
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2 * 2x2
    # M1: (x00+x11)(y00+y11)
    # M2: (x10+x11)y00
    # M3: x00(y10-y11)
    # M4: x11(y01-y00)
    # M5: (x00+x01)y11
    # M6: (x10-x00)(y00+y01)
    # M7: (x01-x11)(y10+y11)
    
    # We need to map local x, y indices to global A, B indices
    # and local C indices to global C indices (which map to W)
    
    def add_strassen_term(A_offsets, B_offsets, C_offsets, term_type):
        nonlocal col
        # A_offsets: dict or list of (global_idx, coeff) for A
        # B_offsets: dict or list of (global_idx, coeff) for B
        # C_offsets: dict or list of (global_idx, coeff) for C (W)
        
        # term_type 1..7
        # We define the linear forms for X (A part) and Y (B part)
        # and the contribution to C.
        
        # Let's hardcode the 7 types
        # X indices: 00, 01, 10, 11
        # Y indices: 00, 01, 10, 11
        
        # Types:
        # 1: X: 00,11 (1,1); Y: 00,11 (1,1); C: 00,11 (1,1)
        # 2: X: 10,11 (1,1); Y: 00 (1); C: 10,11 (1,-1) -> Wait, c10=M2+M4, c11=M1-M2+M3+M6.
        #    So M2 contributes +1 to c10, -1 to c11.
        # 3: X: 00 (1); Y: 10,11 (1,-1); C: 01,11 (1,1)
        #    c01=M3+M5, c11=...+M3...
        # 4: X: 11 (1); Y: 01,00 (1,-1); C: 00,10 (1,1)
        #    c00=...+M4..., c10=M2+M4
        # 5: X: 00,01 (1,1); Y: 11 (1); C: 00,01 (-1,1)
        #    c00=...-M5..., c01=M3+M5
        # 6: X: 10,00 (1,-1); Y: 00,01 (1,1); C: 11 (1)
        #    c11=...+M6...
        # 7: X: 01,11 (1,-1); Y: 10,11 (1,1); C: 00 (1)
        #    c00=M1+M4-M5+M7
        
        # Let's verify Strassen formulas:
        # M1 = (x00+x11)(y00+y11)
        # M2 = (x10+x11)y00
        # M3 = x00(y10-y11)
        # M4 = x11(y01-y00)
        # M5 = (x00+x01)y11
        # M6 = (x10-x00)(y00+y01)
        # M7 = (x01-x11)(y10+y11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Now map to global indices.
        
        # A_offsets is a mapping from local (0,0)..(1,1) to global index in U
        # U index = i*m + j
        # B_offsets is a mapping from local (0,0)..(1,1) to global index in V
        # V index = j*p + k
        # C_offsets is a mapping from local (0,0)..(1,1) to global index in W
        # W index = k*n + i
        
        # We pass in the global indices for the 4 local Xs and 4 local Ys.
        # x00_g, x01_g, x10_g, x11_g
        # y00_g, y01_g, y10_g, y11_g
        # c00_g, c01_g, c10_g, c11_g
        
        # And the term type.
        
        # It's easier to just pass the global indices directly.
        pass

    # Let's define the 7 types explicitly in terms of coefficients
    # For each type, we have:
    # U_coeffs: list of (local_x_idx, coeff)
    # V_coeffs: list of (local_y_idx, coeff)
    # W_coeffs: list of (local_c_idx, coeff)
    
    # Local indices: 0:00, 1:01, 2:10, 3:11
    
    types = [
        # 1
        {
            'U': [(0, 1), (3, 1)],
            'V': [(0, 1), (3, 1)],
            'W': [(0, 1), (3, 1)]
        },
        # 2
        {
            'U': [(2, 1), (3, 1)],
            'V': [(0, 1)],
            'W': [(2, 1), (3, -1)]
        },
        # 3
        {
            'U': [(0, 1)],
            'V': [(2, 1), (3, -1)],
            'W': [(1, 1), (3, 1)]
        },
        # 4
        {
            'U': [(3, 1)],
            'V': [(1, 1), (0, -1)],
            'W': [(0, 1), (2, 1)]
        },
        # 5
        {
            'U': [(0, 1), (1, 1)],
            'V': [(3, 1)],
            'W': [(0, -1), (1, 1)]
        },
        # 6
        {
            'U': [(2, 1), (0, -1)],
            'V': [(0, 1), (1, 1)],
            'W': [(3, 1)]
        },
        # 7
        {
            'U': [(1, 1), (3, -1)],
            'V': [(2, 1), (3, 1)],
            'W': [(0, 1)]
        }
    ]
    
    def process_block(A_local_to_global, B_local_to_global, C_local_to_global):
        nonlocal col
        for t in range(7):
            # A_local_to_global: list of 4 global U indices for x00, x01, x10, x11
            # B_local_to_global: list of 4 global V indices for y00, y01, y10, y11
            # C_local_to_global: list of 4 global W indices for c00, c01, c10, c11
            
            for (lx, cx) in types[t]['U']:
                U[A_local_to_global[lx]][col] += cx
            for (ly, cy) in types[t]['V']:
                V[B_local_to_global[ly]][col] += cy
            for (lc, cw) in types[t]['W']:
                W[C_local_to_global[lc]][col] += cw
            col += 1

    # Left Part
    # A_L: x00=A[0,0], x01=A[0,1], x10=A[1,0], x11=A[1,1]
    # U indices: i*m+j
    A_L_global = [
        0*4+0, # 0
        0*4+1, # 1
        1*4+0, # 4
        1*4+1  # 5
    ]
    
    # For each of the 3 blocks of B_top
    for b_block in range(3):
        # B_top block b_block uses cols 2*b_block, 2*b_block+1
        # Rows 0, 1
        # y00 = B[0, 2*b_block], y01 = B[0, 2*b_block+1]
        # y10 = B[1, 2*b_block], y11 = B[1, 2*b_block+1]
        # V indices: j*p+k
        B_top_global = [
            0*6 + 2*b_block, # 2*b_block
            0*6 + 2*b_block + 1, # 2*b_block+1
            1*6 + 2*b_block, # 6+2*b_block
            1*6 + 2*b_block + 1  # 6+2*b_block+1
        ]
        
        # C result for this block
        # c00 = C[0, 2*b_block], c01 = C[0, 2*b_block+1]
        # c10 = C[1, 2*b_block], c11 = C[1, 2*b_block+1]
        # W indices: k*n+i
        # c00 (i=0, k=2*b_block) -> W[2*b_block*2+0]
        # c01 (i=0, k=2*b_block+1) -> W[(2*b_block+1)*2+0]
        # c10 (i=1, k=2*b_block) -> W[2*b_block*2+1]
        # c11 (i=1, k=2*b_block+1) -> W[(2*b_block+1)*2+1]
        
        C_global = [
            (2*b_block)*2 + 0,
            (2*b_block+1)*2 + 0,
            (2*b_block)*2 + 1,
            (2*b_block+1)*2 + 1
        ]
        
        process_block(A_L_global, B_top_global, C_global)

    # Right Part
    # A_R: x00=A[0,2], x01=A[0,3], x10=A[1,2], x11=A[1,3]
    A_R_global = [
        0*4+2, # 2
        0*4+3, # 3
        1*4+2, # 6
        1*4+3  # 7
    ]
    
    for b_block in range(3):
        # B_bottom block b_block uses cols 2*b_block, 2*b_block+1
        # Rows 2, 3
        # y00 = B[2, 2*b_block], y01 = B[2, 2*b_block+1]
        # y10 = B[3, 2*b_block], y11 = B[3, 2*b_block+1]
        # V indices: j*p+k
        B_bottom_global = [
            2*6 + 2*b_block, # 12+2*b_block
            2*6 + 2*b_block + 1, # 12+2*b_block+1
            3*6 + 2*b_block, # 18+2*b_block
            3*6 + 2*b_block + 1  # 18+2*b_block+1
        ]
        
        C_global = [
            (2*b_block)*2 + 0,
            (2*b_block+1)*2 + 0,
            (2*b_block)*2 + 1,
            (2*b_block+1)*2 + 1
        ]
        
        process_block(A_R_global, B_bottom_global, C_global)

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Simple test
    res = construct()
    U = res['U']
    V = res['V']
    W = res['W']
    print("Rank:", len(U[0]))
