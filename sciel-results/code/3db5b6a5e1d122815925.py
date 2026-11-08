"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Target rank 29 using Strassen hybrid
    # Structure:
    # A (2x3) split into A_left (2x2) and A_right (2x1)
    # B (3x5) split into B_top (2x5) and B_bottom (1x5)
    # B_top split into B_tl (2x2) and B_tr (2x3)
    #
    # C = A_left * B_tl + A_left * B_tr + A_right * B_bottom
    #
    # Part 1: A_left (2x2) * B_tl (2x2) -> Rank 7 (Strassen)
    # Part 2: A_left (2x2) * B_tr (2x3) -> Rank 12 (Schoolbook)
    # Part 3: A_right (2x1) * B_bottom (1x5) -> Rank 10 (Schoolbook)
    # Total Rank = 29
    
    rank = 29
    U = [[0] * rank for _ in range(n * m)] # 6x29
    V = [[0] * rank for _ in range(m * p)] # 15x29
    W = [[0] * rank for _ in range(p * n)] # 10x29
    
    col = 0
    
    # --- Part 1: Strassen for 2x2x2 ---
    # A_left indices in U: rows (0,1) for i=0, (2,3) for i=1? 
    # U index: i*m + j. i in [0,1], j in [0,1].
    # V index: j*p + k. j in [0,1], k in [0,1].
    # W index: k*n + i. k in [0,1], i in [0,1].
    
    # Strassen coefficients for 2x2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We need to map these to our global indices.
    # U rows for A_left: 
    # A11: i=0,j=0 -> U[0]
    # A12: i=0,j=1 -> U[1]
    # A21: i=1,j=0 -> U[2]
    # A22: i=1,j=1 -> U[3]
    
    # V rows for B_tl (j in 0,1; k in 0,1):
    # B11: j=0,k=0 -> V[0]
    # B12: j=0,k=1 -> V[1]
    # B21: j=1,k=0 -> V[5] (j*p+k = 1*5+0? No, p=5. j=1, k=0 -> V[5]. j=1, k=1 -> V[6])
    # Wait, V index is j*p + k.
    # j=0: V[0], V[1], V[2], V[3], V[4]
    # j=1: V[5], V[6], V[7], V[8], V[9]
    # j=2: V[10]...
    # So B11 is V[0], B12 is V[1], B21 is V[5], B22 is V[6].
    
    # W rows for C_top_left (k in 0,1; i in 0,1):
    # W index is k*n + i.
    # k=0: W[0] (i=0, C11), W[1] (i=1, C21)
    # k=1: W[2] (i=0, C12), W[3] (i=1, C22)
    
    strassen_ops = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # M1 = (A11+A22)(B11+B22)
        # U: A11(1), A22(1). V: B11(1), B22(1).
        # W: C11(1), C12(1), C21(1), C22(1)
        # This is not a single multiplication, it's a linear combination.
        # In tensor decomposition, each term is a rank-1 tensor u v^T w^T.
        # Strassen uses 7 multiplications. Each multiplication corresponds to a rank-1 term.
        # The inputs are linear combinations of A and B.
        # So for M1:
        # u vector has 1 at A11 and A22.
        # v vector has 1 at B11 and B22.
        # w vector has 1 at C11, C12, C21, C22? No.
        # M1 contributes to C11, C12, C21, C22 with coefficients.
        # C11 += M1, C12 -= M1, C21 += M1, C22 -= M1? 
        # Let's check Strassen formulas:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # So for M1:
        # u: A11=1, A22=1
        # v: B11=1, B22=1
        # w: C11=1, C12=-1, C21=1, C22=-1
        
        # M2 = (A21+A22)B11
        # u: A21=1, A22=1
        # v: B11=1
        # w: C21=1, C22=-1
        
        # M3 = A11(B21-B22)
        # u: A11=1
        # v: B21=1, B22=-1
        # w: C12=1, C22=1
        
        # M4 = A22(B12-B11)
        # u: A22=1
        # v: B12=1, B11=-1
        # w: C11=1, C21=1
        
        # M5 = (A11+A12)B22
        # u: A11=1, A12=1
        # v: B22=1
        # w: C11=-1, C12=1
        
        # M6 = (A21-A11)(B11+B12)
        # u: A21=1, A11=-1
        # v: B11=1, B12=1
        # w: C22=1
        
        # M7 = (A12-A22)(B21+B22)
        # u: A12=1, A22=-1
        # v: B21=1, B22=1
        # w: C11=1, C22=-1
    ]
    
    # Map indices
    # U indices: 0:A11, 1:A12, 2:A21, 3:A22
    # V indices: 0:B11, 1:B12, 5:B21, 6:B22
    # W indices: 0:C11, 1:C21, 2:C12, 3:C22
    
    strassen_defs = [
        # (U_indices, V_indices, W_indices)
        ((0, 3), (0, 6), (0, 2, 1, 3)), # M1: C11(0), C12(2), C21(1), C22(3) -> signs: 1, -1, 1, -1
        ((2, 3), (0,), (1, 3)),          # M2: C21(1), C22(3) -> signs: 1, -1
        ((0,), (5, 6), (2, 3)),          # M3: C12(2), C22(3) -> signs: 1, 1
        ((3,), (1, 0), (0, 1)),          # M4: C11(0), C21(1) -> signs: 1, 1. Note V order B12, B11.
        ((0, 1), (6,), (0, 2)),          # M5: C11(0), C12(2) -> signs: -1, 1
        ((2, 0), (0, 1), (3,)),          # M6: C22(3) -> sign: 1. U order A21, A11.
        ((1, 3), (5, 6), (0, 3)),        # M7: C11(0), C22(3) -> signs: 1, -1
    ]
    strassen_signs = [
        [1, -1, 1, -1],
        [1, -1],
        [1, 1],
        [1, 1],
        [-1, 1],
        [1],
        [1, -1]
    ]
    
    for idx, (u_idx, v_idx, w_idx) in enumerate(strassen_defs):
        signs = strassen_signs[idx]
        # We need to handle signs in U, V, or W. Usually put in W or U.
        # Let's put signs in W for simplicity, assuming U and V are positive 1s.
        # But some have -1 in U or V.
        # Let's just assign coefficients directly.
        
        # Determine coefficients for U, V, W
        u_coeffs = [0.0] * len(u_idx)
        v_coeffs = [0.0] * len(v_idx)
        w_coeffs = [0.0] * len(w_idx)
        
        # Re-derive coefficients from the definition to be safe
        # This is getting complex. Let's hardcode the 7 rank-1 tensors for Strassen 2x2x2.
        
        pass
        
    # Alternative: Just use the known Strassen decomposition coefficients.
    # Rank 1 tensors for Strassen:
    # 1. u=[1,0,0,1], v=[1,0,0,1], w=[1,-1,1,-1]
    # 2. u=[0,0,1,1], v=[1,0,0,0], w=[0,1,0,-1]
    # 3. u=[1,0,0,0], v=[0,0,1,-1], w=[0,0,1,1]
    # 4. u=[0,0,0,1], v=[0,1,-1,0], w=[1,1,0,0]
    # 5. u=[1,1,0,0], v=[0,0,0,1], w=[-1,0,1,0]
    # 6. u=[0,-1,1,0], v=[1,1,0,0], w=[0,0,0,1]
    # 7. u=[0,1,0,-1], v=[0,0,1,1], w=[1,0,0,-1]
    
    # Note: u indices 0:A11, 1:A12, 2:A21, 3:A22
    # v indices 0:B11, 1:B12, 2:B21, 3:B22 (local to Strassen block)
    # w indices 0:C11, 1:C21, 2:C12, 3:C22 (local to Strassen block)
    
    strassen_tensors = [
        ([1,0,0,1], [1,0,0,1], [1,-1,1,-1]),
        ([0,0,1,1], [1,0,0,0], [0,1,0,-1]),
        ([1,0,0,0], [0,0,1,-1], [0,0,1,1]),
        ([0,0,0,1], [0,1,-1,0], [1,1,0,0]),
        ([1,1,0,0], [0,0,0,1], [-1,0,1,0]),
        ([0,-1,1,0], [1,1,0,0], [0,0,0,1]),
        ([0,1,0,-1], [0,0,1,1], [1,0,0,-1]),
    ]
    
    # Global Index Mappings
    # U: 0:A11, 1:A12, 2:A21, 3:A22
    u_map = [0, 1, 2, 3]
    # V: 0:B11, 1:B12, 2:B21, 3:B22
    # Global V indices: B11=0, B12=1, B21=5, B22=6
    v_map = [0, 1, 5, 6]
    # W: 0:C11, 1:C21, 2:C12, 3:C22
    # Global W indices: C11=0, C21=1, C12=2, C22=3
    w_map = [0, 1, 2, 3]
    
    for u_t, v_t, w_t in strassen_tensors:
        for i, val in enumerate(u_t):
            if val != 0:
                U[u_map[i]][col] = val
        for j, val in enumerate(v_t):
            if val != 0:
                V[v_map[j]][col] = val
        for k, val in enumerate(w_t):
            if val != 0:
                W[w_map[k]][col] = val
        col += 1
    
    # --- Part 2: Schoolbook for A_left (2x2) * B_tr (2x3) ---
    # A_left: i in [0,1], j in [0,1]
    # B_tr: j in [0,1], k in [2,4] (columns 3,4,5 of B, so indices 2,3,4 in k)
    # Result C: i in [0,1], k in [2,4]
    
    for i in range(2):
        for j in range(2):
            for k in range(2, 5):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # --- Part 3: Schoolbook for A_right (2x1) * B_bottom (1x5) ---
    # A_right: i in [0,1], j=2
    # B_bottom: j=2, k in [0,4]
    # Result C: i in [0,1], k in [0,4]
    
    for i in range(2):
        for k in range(5):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Check if col == 29
    assert col == 29, f"Expected rank 29, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
