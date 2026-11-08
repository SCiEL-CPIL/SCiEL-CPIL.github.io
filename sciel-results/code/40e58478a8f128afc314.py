import time
import numpy as np

DIMENSIONS = [3, 3, 6]
START_TIME = time.time()
TIME_LIMIT = 110

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    U = []
    V = []
    W = []
    
    def add_term(u_vals, v_vals, w_vals):
        # u_vals: dict of index->value
        # v_vals: dict of index->value
        # w_vals: dict of index->value
        u_vec = [0.0] * nm
        v_vec = [0.0] * mp
        w_vec = [0.0] * pn
        for idx, val in u_vals.items():
            u_vec[idx] = val
        for idx, val in v_vals.items():
            v_vec[idx] = val
        for idx, val in w_vals.items():
            w_vec[idx] = val
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)

    # Helper to get index
    def a_idx(i, j): return i * m + j
    def b_idx(j, k): return j * p + k
    def c_idx(k, i): return k * n + i

    # Build rank 26 decomposition for 3x3*3x3
    # We do this for k in 0..2 (B1) and k in 3..5 (B2)
    
    for k0 in [0, 3]:
        # k0 is the start of the k-range for this 3x3 block
        # A is 3x3, B is 3x3 (cols k0..k0+2)
        
        # Strassen for 2x2 block (i,j in 0,1; k in k0, k0+1)
        # M1 = (A00+A11)(B00+B11)
        add_term({a_idx(0,0): 1, a_idx(1,1): 1}, 
                 {b_idx(0,k0): 1, b_idx(1,k0+1): 1}, 
                 {c_idx(k0,0): 1, c_idx(k0+1,1): 1})
        
        # M2 = (A10+A11)B00
        add_term({a_idx(1,0): 1, a_idx(1,1): 1}, 
                 {b_idx(0,k0): 1}, 
                 {c_idx(k0,1): 1})
                 
        # M3 = A00(B10-B11)
        add_term({a_idx(0,0): 1}, 
                 {b_idx(1,k0): 1, b_idx(1,k0+1): -1}, 
                 {c_idx(k0,0): 1, c_idx(k0+1,0): -1})
                 
        # M4 = A11(B01-B00)
        add_term({a_idx(1,1): 1}, 
                 {b_idx(0,k0+1): 1, b_idx(0,k0): -1}, 
                 {c_idx(k0,1): 1, c_idx(k0+1,1): -1})
                 
        # M5 = (A00+A01)B11
        add_term({a_idx(0,0): 1, a_idx(0,1): 1}, 
                 {b_idx(1,k0+1): 1}, 
                 {c_idx(k0,0): 1, c_idx(k0+1,0): 1})
                 
        # M6 = (A10-A00)(B00+B01)
        add_term({a_idx(1,0): 1, a_idx(0,0): -1}, 
                 {b_idx(0,k0): 1, b_idx(0,k0+1): 1}, 
                 {c_idx(k0,1): 1, c_idx(k0+1,1): 1})
                 
        # M7 = (A01-A11)(B10+B11)
        add_term({a_idx(0,1): 1, a_idx(1,1): -1}, 
                 {b_idx(1,k0): 1, b_idx(1,k0+1): 1}, 
                 {c_idx(k0,0): 1, c_idx(k0+1,1): 1})
        
        # Now we need to combine these to get C22, C21, C12, C11?
        # No, the Strassen terms ARE the multiplication of the 2x2 block.
        # The output C22 (2x2) is formed by linear combinations of M1..M7.
        # But in tensor decomposition, we don't form linear combinations of outputs.
        # We form linear combinations of INPUTS.
        # So the 7 terms above are the rank 7 decomposition of the 2x2*2x2 sub-tensor.
        
        # Now handle the remaining parts of 3x3*3x3.
        # C22 also has contribution from A21*B12.
        # A21 is (rows 0,1; col 2). B12 is (row 2; cols k0, k0+1).
        # Product is 2x2.
        # Terms:
        # A02 * B2,k0 -> C_k0,0
        # A02 * B2,k0+1 -> C_k0+1,0
        # A12 * B2,k0 -> C_k0,1
        # A12 * B2,k0+1 -> C_k0+1,1
        
        add_term({a_idx(0,2): 1}, {b_idx(2,k0): 1}, {c_idx(k0,0): 1})
        add_term({a_idx(0,2): 1}, {b_idx(2,k0+1): 1}, {c_idx(k0+1,0): 1})
        add_term({a_idx(1,2): 1}, {b_idx(2,k0): 1}, {c_idx(k0,1): 1})
        add_term({a_idx(1,2): 1}, {b_idx(2,k0+1): 1}, {c_idx(k0+1,1): 1})
        
        # C21 (2x1) = A22*B21 + A21*B11
        # A22 (2x2) * B21 (2x1). B21 is rows 0,1; col k0+2.
        # A00*B0,k0+2 -> C_k0+2,0
        # A01*B1,k0+2 -> C_k0+2,0
        # A10*B0,k0+2 -> C_k0+2,1
        # A11*B1,k0+2 -> C_k0+2,1
        
        add_term({a_idx(0,0): 1}, {b_idx(0,k0+2): 1}, {c_idx(k0+2,0): 1})
        add_term({a_idx(0,1): 1}, {b_idx(1,k0+2): 1}, {c_idx(k0+2,0): 1})
        add_term({a_idx(1,0): 1}, {b_idx(0,k0+2): 1}, {c_idx(k0+2,1): 1})
        add_term({a_idx(1,1): 1}, {b_idx(1,k0+2): 1}, {c_idx(k0+2,1): 1})
        
        # A21 (2x1) * B11 (1x1). B11 is row 2; col k0+2.
        # A02 * B2,k0+2 -> C_k0+2,0
        # A12 * B2,k0+2 -> C_k0+2,1
        
        add_term({a_idx(0,2): 1}, {b_idx(2,k0+2): 1}, {c_idx(k0+2,0): 1})
        add_term({a_idx(1,2): 1}, {b_idx(2,k0+2): 1}, {c_idx(k0+2,1): 1})
        
        # C12 (1x2) = A12*B22 + A11*B12
        # A12 (1x2) * B22 (2x2).
        # A20*B0,k0 -> C_k0,2
        # A20*B0,k0+1 -> C_k0+1,2
        # A21*B1,k0 -> C_k0,2
        # A21*B1,k0+1 -> C_k0+1,2
        
        add_term({a_idx(2,0): 1}, {b_idx(0,k0): 1}, {c_idx(k0,2): 1})
        add_term({a_idx(2,0): 1}, {b_idx(0,k0+1): 1}, {c_idx(k0+1,2): 1})
        add_term({a_idx(2,1): 1}, {b_idx(1,k0): 1}, {c_idx(k0,2): 1})
        add_term({a_idx(2,1): 1}, {b_idx(1,k0+1): 1}, {c_idx(k0+1,2): 1})
        
        # A11 (1x1) * B12 (1x2).
        # A22 * B2,k0 -> C_k0,2
        # A22 * B2,k0+1 -> C_k0+1,2
        
        add_term({a_idx(2,2): 1}, {b_idx(2,k0): 1}, {c_idx(k0,2): 1})
        add_term({a_idx(2,2): 1}, {b_idx(2,k0+1): 1}, {c_idx(k0+1,2): 1})
        
        # C11 (1x1) = A12*B21 + A11*B11
        # A12 (1x2) * B21 (2x1).
        # A20*B0,k0+2 -> C_k0+2,2
        # A21*B1,k0+2 -> C_k0+2,2
        
        add_term({a_idx(2,0): 1}, {b_idx(0,k0+2): 1}, {c_idx(k0+2,2): 1})
        add_term({a_idx(2,1): 1}, {b_idx(1,k0+2): 1}, {c_idx(k0+2,2): 1})
        
        # A11 (1x1) * B11 (1x1).
        # A22 * B2,k0+2 -> C_k0+2,2
        
        add_term({a_idx(2,2): 1}, {b_idx(2,k0+2): 1}, {c_idx(k0+2,2): 1})

    # Total terms: 7 (Strassen) + 4 + 6 + 6 + 3 = 26 per block.
    # 2 blocks = 52 terms.
    
    # Convert to lists of lists
    U_list = [row for row in U]
    V_list = [row for row in V]
    W_list = [row for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
