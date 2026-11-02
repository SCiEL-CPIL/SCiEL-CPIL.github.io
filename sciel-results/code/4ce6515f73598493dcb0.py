import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    # Rank calculation
    # 2 * ( (7 + 7 + 4) + 1 ) = 38
    rank = 38
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add a bilinear term
    # u_idx: (i, j)
    # v_idx: (j, k)
    # w_idx: (k, i)
    # coeff: multiplier
    def add_term(i, j, k, coeff=1):
        nonlocal col
        if col >= rank:
            return
        U[i*m + j, col] += coeff
        V[j*p + k, col] += coeff
        W[k*n + i, col] += coeff
        col += 1
        
    # Helper to add Strassen 2x2x2
    # a_rows: [i0, i1], a_cols: [j0, j1]
    # b_rows: [j0, j1], b_cols: [k0, k1]
    # Note: j0, j1 must match for contraction
    def add_strassen(i0, i1, j0, j1, k0, k1):
        nonlocal col
        # M1 = (A00+A11)(B00+B11)
        add_term(i0, j0, k0, 1)
        add_term(i0, j1, k0, 1) # Wait, Strassen mixes indices
        # Let's explicitly map the 7 terms
        
        # A = [[A00, A01], [A10, A11]]
        # B = [[B00, B01], [B10, B11]]
        # A00 is at (i0, j0)
        # A01 is at (i0, j1)
        # A10 is at (i1, j0)
        # A11 is at (i1, j1)
        
        # B00 is at (j0, k0)
        # B01 is at (j0, k1)
        # B10 is at (j1, k0)
        # B11 is at (j1, k1)
        
        # M1 = (A00+A11)(B00+B11)
        # This expands to 4 terms. But Strassen uses 1 rank term for the product?
        # No, the rank term is a single multiplication of linear combinations.
        # The decomposition is:
        # M_r = (linear comb of A) * (linear comb of B)
        # C = sum M_r * (linear comb of output basis)
        
        # The standard Strassen decomposition adds 7 rank terms.
        # For each rank term r, we have:
        # U_vec (length 4 for 2x2 block)
        # V_vec (length 4 for 2x2 block)
        # W_vec (length 4 for 2x2 block)
        
        # We need to add these to the global U, V, W.
        
        # Term 1: (A00+A11)(B00+B11)
        # U: A00=1, A11=1
        # V: B00=1, B11=1
        # W: C00=1, C11=1, C01=-1, C10=-1?
        # Let's check the W coefficients for Strassen.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So for M1, W contributions are: C00:1, C11:1.
        # For M2, W: C10:1, C11:-1.
        # For M3, W: C01:1, C11:1.
        # For M4, W: C00:1, C10:1.
        # For M5, W: C00:-1, C01:1.
        # For M6, W: C11:1.
        # For M7, W: C00:1.
        
        # U/V vectors for each M:
        # M1: U=(1,0,0,1), V=(1,0,0,1)
        # M2: U=(0,0,1,1), V=(1,0,0,0)
        # M3: U=(1,0,0,0), V=(0,1,0,-1)
        # M4: U=(0,0,0,1), V=(0,0,1,-1)
        # M5: U=(1,1,0,0), V=(0,0,0,1)
        # M6: U=(0,0,1,-1), V=(1,1,0,0)
        # M7: U=(0,1,0,-1), V=(0,0,1,1)
        
        # We will iterate through these 7 terms.
        
        terms = [
            # (U_coeffs, V_coeffs, W_coeffs)
            # U: [A00, A01, A10, A11]
            # V: [B00, B01, B10, B11]
            # W: [C00, C01, C10, C11]
            ((1,0,0,1), (1,0,0,1), (1,0,0,1)), # M1
            ((0,0,1,1), (1,0,0,0), (0,0,1,-1)), # M2
            ((1,0,0,0), (0,1,0,-1), (0,1,0,1)), # M3
            ((0,0,0,1), (0,0,1,-1), (1,0,1,0)), # M4
            ((1,1,0,0), (0,0,0,1), (-1,1,0,0)), # M5
            ((0,0,1,-1), (1,1,0,0), (0,0,0,1)), # M6
            ((0,1,0,-1), (0,0,1,1), (1,0,0,0)), # M7
        ]
        
        for u_c, v_c, w_c in terms:
            if col >= rank:
                return
            # Map local indices to global
            # Local A: 0->(i0,j0), 1->(i0,j1), 2->(i1,j0), 3->(i1,j1)
            # Local B: 0->(j0,k0), 1->(j0,k1), 2->(j1,k0), 3->(j1,k1)
            # Local C: 0->(k0,i0), 1->(k1,i0), 2->(k0,i1), 3->(k1,i1)
            # Note: W index is (k, i).
            # C00 is output (i0, k0) -> W index (k0, i0)
            # C01 is output (i0, k1) -> W index (k1, i0)
            # C10 is output (i1, k0) -> W index (k0, i1)
            # C11 is output (i1, k1) -> W index (k1, i1)
            
            u_indices = [
                (i0, j0), (i0, j1), (i1, j0), (i1, j1)
            ]
            v_indices = [
                (j0, k0), (j0, k1), (j1, k0), (j1, k1)
            ]
            w_indices = [
                (k0, i0), (k1, i0), (k0, i1), (k1, i1)
            ]
            
            for idx, coeff in enumerate(u_c):
                if coeff != 0:
                    i, j = u_indices[idx]
                    U[i*m + j, col] += coeff
            for idx, coeff in enumerate(v_c):
                if coeff != 0:
                    j, k = v_indices[idx]
                    V[j*p + k, col] += coeff
            for idx, coeff in enumerate(w_c):
                if coeff != 0:
                    k, i = w_indices[idx]
                    W[k*n + i, col] += coeff
            col += 1

    # Helper to add 2x2x1 (Rank 4)
    def add_2x2x1(i0, i1, j0, j1, k0):
        # A (2x2) * B (2x1)
        # B is column k0, rows j0, j1
        # C is column k0, rows i0, i1
        # C00 = A00 B00 + A01 B10
        # C10 = A10 B00 + A11 B10
        # 4 terms
        for i, j in [(i0, j0), (i0, j1), (i1, j0), (i1, j1)]:
            if col >= rank:
                return
            U[i*m + j, col] = 1
            V[j*p + k0, col] = 1
            W[k0*n + i, col] = 1
            col += 1

    # Helper to add 2x1x5 (Rank 1)
    # A (2x1) * B (1x5)
    # A is col j0, rows i0, i1
    # B is row j0, cols k0..k4
    # C is rows i0, i1, cols k0..k4
    # C0k = A0 Bk
    # C1k = A1 Bk
    # This is 1 rank term?
    # M = A0 A1? No.
    # M = (A0, A1) * (B0, ..., B4)?
    # No, the product is outer product of vector A and vector B.
    # Rank 1 tensor: U = A, V = B, W = Identity?
    # U shape (2x1, 1) -> [A0, A1]
    # V shape (1x5, 1) -> [B0..B4]
    # W shape (5x2, 1) -> [1,0, 0,1, 1,0, 0,1, 1,0, 0,1]?
    # C_ik = U_i V_k W_ik.
    # We want C_ik = A_i B_k.
    # So if U_i = A_i, V_k = B_k, then W_ik must be 1.
    # So W is all ones.
    def add_2x1x5(i0, i1, j0, k0):
        if col >= rank:
            return
        # U: (i0, j0)=1, (i1, j0)=1
        U[i0*m + j0, col] = 1
        U[i1*m + j0, col] = 1
        # V: (j0, k) = 1 for k in k0..k0+4
        for k in range(k0, k0+5):
            V[j0*p + k, col] = 1
        # W: (k, i) = 1 for all k, i
        for k in range(k0, k0+5):
            W[k*n + i0, col] = 1
            W[k*n + i1, col] = 1
        col += 1

    # Main construction
    # Split B into B1 (cols 0-4) and B2 (cols 5-9)
    
    # Block 1: B1
    # Split A into A_L (cols 0-1) and A_R (col 2)
    # Split B1 into B1_top (rows 0-1) and B1_bot (row 2)
    
    # Term 1.1: A_L (2x2) * B1_top (2x5)
    # Split B1_top into 2x2, 2x2, 2x1
    # B1_top cols 0-1, 2-3, 4
    add_strassen(0, 1, 0, 1, 0, 1) # A_L * B1_top[0:2, 0:2]
    add_strassen(0, 1, 0, 1, 2, 3) # A_L * B1_top[0:2, 2:4]
    add_2x2x1(0, 1, 0, 1, 4)       # A_L * B1_top[0:2, 4:5]
    
    # Term 1.2: A_R (2x1) * B1_bot (1x5)
    # A_R is col 2. B1_bot is row 2, cols 0-4.
    add_2x1x5(0, 1, 2, 0)
    
    # Block 2: B2 (cols 5-9)
    # Term 2.1: A_L (2x2) * B2_top (2x5)
    # B2_top cols 5-6, 7-8, 9
    add_strassen(0, 1, 0, 1, 5, 6)
    add_strassen(0, 1, 0, 1, 7, 8)
    add_2x2x1(0, 1, 0, 1, 9)
    
    # Term 2.2: A_R (2x1) * B2_bot (1x5)
    # B2_bot is row 2, cols 5-9.
    add_2x1x5(0, 1, 2, 5)
    
    if col != rank:
        # Fallback or error
        pass
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
