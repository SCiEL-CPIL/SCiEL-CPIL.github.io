import numpy as np

def construct():
    # Dimensions
    n, m, p = 2, 2, 9
    
    # We use 4 Strassen blocks for the first 8 columns and 1 schoolbook block for the 9th.
    # Total rank = 4 * 7 + 4 = 32.
    
    rank = 32
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Helper to add a term
    # term_idx: index in U, V, W
    # u_idx, u_val, v_idx, v_val, w_idx, w_val
    def add_term(idx, u_idx, u_val, v_idx, v_val, w_idx, w_val):
        U[u_idx, idx] += u_val
        V[v_idx, idx] += v_val
        W[w_idx, idx] += w_val
        
    # Strassen block for columns k, k+1
    # A indices: 0,1,2,3
    # B indices: k, k+1, 9+k, 9+k+1
    # C indices: 2k, 2k+1, 2k+2, 2k+3 (row-major: C[0,k]=2k, C[0,k+1]=2k+1, C[1,k]=2k+2, C[1,k+1]=2k+3)
    # Wait, the trace convention is W[k*n+i, r].
    # C is p x n? No, C is n x p.
    # "output factor follows the official trace convention W[k*n+i,r]"
    # Usually W corresponds to C flattened.
    # If C is 2x9, size 18.
    # Index in W: row * n + col? Or col * n + row?
    # "W[k*n+i,r]" suggests k is the column index (0..8) and i is the row index (0..1)?
    # Let's check the previous code's indexing.
    # Previous code:
    # C[0,k0]=2*k0, C[0,k1]=2*k1, C[1,k0]=2*k0+1, C[1,k1]=2*k1+1
    # This implies Index = 2*col + row.
    # So W index = 2*col + row.
    
    for block_idx in range(4):
        k0 = 2 * block_idx
        k1 = k0 + 1
        base_term = block_idx * 7
        
        # B indices
        b_e = k0       # B[0, k0]
        b_f = k1       # B[0, k1]
        b_g = 9 + k0   # B[1, k0]
        b_h = 9 + k1   # B[1, k1]
        
        # C indices
        c_00 = 2 * k0     # C[0, k0]
        c_01 = 2 * k1     # C[0, k1]
        c_10 = 2 * k0 + 1 # C[1, k0]
        c_11 = 2 * k1 + 1 # C[1, k1]
        
        # A indices
        a_00 = 0
        a_01 = 1
        a_10 = 2
        a_11 = 3
        
        # Strassen terms
        # 1: (a00 + a11) * (b_e + b_h)
        add_term(base_term + 0, a_00, 1, b_e, 1, c_00, 1)
        add_term(base_term + 0, a_11, 1, b_h, 1, c_11, 1)
        
        # 2: (a10 + a11) * b_e
        add_term(base_term + 1, a_10, 1, b_e, 1, c_10, 1)
        add_term(base_term + 1, a_11, 1, b_e, 1, c_11, -1)
        
        # 3: a00 * (b_f - b_h)
        add_term(base_term + 2, a_00, 1, b_f, 1, c_01, 1)
        add_term(base_term + 2, a_00, 1, b_h, -1, c_01, 1)
        
        # 4: a11 * (b_g - b_e)
        add_term(base_term + 3, a_11, 1, b_g, 1, c_00, 1)
        add_term(base_term + 3, a_11, 1, b_e, -1, c_00, 1)
        # Wait, standard Strassen:
        # M4 = d(g-e). Output C10 uses M4?
        # C10 = M2 + M4.
        # C11 = M1 - M2 + M6 + M7.
        # Let's re-verify the mapping from the previous working code.
        # Previous code:
        # p4 = d(g-e) -> {3:1}, {9+k0:1, k0:-1}, {2*k0:1, 2*k0+1:1}
        # This adds to C[0,k0] and C[1,k0]?
        # 2*k0 is C[0,k0]. 2*k0+1 is C[1,k0].
        # So M4 contributes to C00 and C10?
        # Standard Strassen:
        # C00 = M1 - M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M6 + M7
        # So M4 contributes to C00 (-1) and C10 (+1).
        # My add_term above for term 0 (M1) added to c_00 and c_11.
        # M1 = (a+d)(e+h).
        # C00 uses +M1. C11 uses +M1.
        # So add_term(base, 0, 1, b_e, 1, c_00, 1) is correct for A part.
        # But I need to be careful with signs.
        
        # Let's restart the Strassen mapping carefully based on the previous working code.
        # The previous code was:
        # p1 = (a+d)(e+h) -> U:{0:1, 3:1}, V:{k0:1, 9+k1:1}, W:{2*k0:1, 2*k1+1:1}
        # p2 = (c+d)(e)   -> U:{2:1, 3:1}, V:{k0:1},       W:{2*k0+1:1, 2*k1+1:-1}
        # p3 = a(f-h)     -> U:{0:1},      V:{k1:1, 9+k1:-1},W:{2*k1:1, 2*k1+1:1}
        # p4 = d(g-e)     -> U:{3:1},      V:{9+k0:1, k0:-1},W:{2*k0:1, 2*k0+1:1}
        # p5 = (a+b)(h)   -> U:{0:1, 1:1}, V:{9+k1:1},      W:{2*k0:-1, 2*k1:1}
        # p6 = (c-a)(e+f) -> U:{2:1, 0:-1},V:{k0:1, k1:1},  W:{2*k1+1:1}
        # p7 = (b-d)(g+h) -> U:{1:1, 3:-1},V:{9+k0:1, 9+k1:1},W:{2*k0:1}
        
        # I will replicate this exact logic but with different variable names and NumPy.
        
        # Term 0: M1
        U[0, base_term] += 1
        U[3, base_term] += 1
        V[k0, base_term] += 1
        V[9+k1, base_term] += 1
        W[2*k0, base_term] += 1
        W[2*k1+1, base_term] += 1
        
        # Term 1: M2
        U[2, base_term+1] += 1
        U[3, base_term+1] += 1
        V[k0, base_term+1] += 1
        W[2*k0+1, base_term+1] += 1
        W[2*k1+1, base_term+1] += -1
        
        # Term 2: M3
        U[0, base_term+2] += 1
        V[k1, base_term+2] += 1
        V[9+k1, base_term+2] += -1
        W[2*k1, base_term+2] += 1
        W[2*k1+1, base_term+2] += 1
        
        # Term 3: M4
        U[3, base_term+3] += 1
        V[9+k0, base_term+3] += 1
        V[k0, base_term+3] += -1
        W[2*k0, base_term+3] += 1
        W[2*k0+1, base_term+3] += 1
        
        # Term 4: M5
        U[0, base_term+4] += 1
        U[1, base_term+4] += 1
        V[9+k1, base_term+4] += 1
        W[2*k0, base_term+4] += -1
        W[2*k1, base_term+4] += 1
        
        # Term 5: M6
        U[2, base_term+5] += 1
        U[0, base_term+5] += -1
        V[k0, base_term+5] += 1
        V[k1, base_term+5] += 1
        W[2*k1+1, base_term+5] += 1
        
        # Term 6: M7
        U[1, base_term+6] += 1
        U[3, base_term+6] += -1
        V[9+k0, base_term+6] += 1
        V[9+k1, base_term+6] += 1
        W[2*k0, base_term+6] += 1

    # Schoolbook for column 8
    k = 8
    base_term = 28
    
    # C[0,8] = A[0,0]B[0,8] + A[0,1]B[1,8]
    # Term 28: A[0,0] * B[0,8]
    U[0, base_term] += 1
    V[k, base_term] += 1
    W[2*k, base_term] += 1
    
    # Term 29: A[0,1] * B[1,8]
    U[1, base_term+1] += 1
    V[9+k, base_term+1] += 1
    W[2*k, base_term+1] += 1
    
    # Term 30: A[1,0] * B[0,8]
    U[2, base_term+2] += 1
    V[k, base_term+2] += 1
    W[2*k+1, base_term+2] += 1
    
    # Term 31: A[1,1] * B[1,8]
    U[3, base_term+3] += 1
    V[9+k, base_term+3] += 1
    W[2*k+1, base_term+3] += 1

    return {"U": U, "V": V, "W": W}
