import numpy as np

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=10
    # We decompose into 5 independent 2x2 multiplications.
    # Each 2x2 multiplication has rank 7 (Strassen).
    # Total rank = 5 * 7 = 35.
    
    rank = 35
    U = [[0] * rank for _ in range(n * m)]  # 4 x 35
    V = [[0] * rank for _ in range(m * p)]  # 20 x 35
    W = [[0] * rank for _ in range(p * n)]  # 20 x 35
    
    # Strassen factors for 2x2 matrix multiplication
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A11 - A01)(B10 + B11)
    
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6 - M7
    
    # Indices for A: i*m + j -> i*2 + j
    # Indices for B: j*p + k -> j*10 + k
    # Indices for C: k*n + i -> k*2 + i
    
    # We iterate over the 5 blocks l in 0..4
    # For each l, the columns of B are 2*l and 2*l+1? 
    # Wait, the grouping was k = 5c + l with c in 0,1.
    # This means for fixed l, the columns are 5*0+l and 5*1+l.
    # i.e., l and 5+l.
    # So the 2x2 matrix B_l has columns indexed by c=0 and c=1.
    # B_l[j, 0] = B[j, l]
    # B_l[j, 1] = B[j, 5+l]
    
    # Let's verify the output indices.
    # C_l[i, 0] = C[i, l]
    # C_l[i, 1] = C[i, 5+l]
    
    # Strassen coefficients
    # U part (A coefficients)
    # V part (B coefficients)
    # W part (C coefficients)
    
    # We will fill U, V, W for each of the 5 blocks.
    # Block index l: 0 to 4.
    # Strassen term index r: 0 to 6.
    # Column in U,V,W: col = 7*l + r
    
    for l in range(5):
        # Column indices for this block in B and C
        k0 = l
        k1 = 5 + l
        
        # Strassen terms
        # r=0: M1
        # A: A00 + A11
        # B: B00 + B11 (in B_l, this is B[l,0] + B[l,1] for row 0? No, B_l[j,c])
        # B_l[0,0] = B[0, k0]
        # B_l[1,1] = B[1, k1]
        # M1 = (A00+A11)(B00+B11)
        
        # Let's define the linear combinations explicitly.
        # A indices: 00, 01, 10, 11
        # B indices for this block: 
        #   j=0, c=0 -> B[0, k0]
        #   j=0, c=1 -> B[0, k1]
        #   j=1, c=0 -> B[1, k0]
        #   j=1, c=1 -> B[1, k1]
        
        # We need to map these to the global B index j*p + k.
        # B_idx(0, k0) = 0*10 + k0
        # B_idx(0, k1) = 0*10 + k1
        # B_idx(1, k0) = 1*10 + k0
        # B_idx(1, k1) = 1*10 + k1
        
        # C indices for this block:
        # C[i, k0] and C[i, k1]
        # C_idx(k0, i) = k0*2 + i
        # C_idx(k1, i) = k1*2 + i
        
        # Strassen Terms:
        # 1. M1 = (A00+A11)(B00+B11)
        # 2. M2 = (A10+A11)B00
        # 3. M3 = A00(B01-B11)
        # 4. M4 = A11(B10-B00)
        # 5. M5 = (A00+A01)B11
        # 6. M6 = (A10-A00)(B00+B01)
        # 7. M7 = (A11-A01)(B10+B11)
        
        # Note: B00 = B_l[0,0] = B[0,k0]
        #       B01 = B_l[0,1] = B[0,k1]
        #       B10 = B_l[1,0] = B[1,k0]
        #       B11 = B_l[1,1] = B[1,k1]
        
        terms = []
        
        # Helper to add a term
        def add_term(col, a_coeffs, b_coeffs, c_coeffs):
            # a_coeffs: dict {A_idx: coeff}
            # b_coeffs: dict {B_idx: coeff}
            # c_coeffs: dict {C_idx: coeff}
            for a_idx, val in a_coeffs.items():
                U[a_idx][col] += val
            for b_idx, val in b_coeffs.items():
                V[b_idx][col] += val
            for c_idx, val in c_coeffs.items():
                W[c_idx][col] += val
        
        # A indices: 0:00, 1:01, 2:10, 3:11
        # B indices for this block:
        b_00 = 0 * 10 + k0
        b_01 = 0 * 10 + k1
        b_10 = 1 * 10 + k0
        b_11 = 1 * 10 + k1
        
        # C indices for this block:
        # C[i, k] -> k*2 + i
        # C00 = C[0, k0] -> k0*2 + 0
        # C01 = C[0, k1] -> k1*2 + 0
        # C10 = C[1, k0] -> k0*2 + 1
        # C11 = C[1, k1] -> k1*2 + 1
        
        c_00 = k0 * 2 + 0
        c_01 = k1 * 2 + 0
        c_10 = k0 * 2 + 1
        c_11 = k1 * 2 + 1
        
        # M1
        add_term(7*l + 0, {0: 1, 3: 1}, {b_00: 1, b_11: 1}, {c_00: 1, c_11: 1})
        # M2
        add_term(7*l + 1, {2: 1, 3: 1}, {b_00: 1}, {c_10: 1})
        # M3
        add_term(7*l + 2, {0: 1}, {b_01: 1, b_11: -1}, {c_01: 1})
        # M4
        add_term(7*l + 3, {3: 1}, {b_10: 1, b_00: -1}, {c_00: -1, c_10: 1})
        # M5
        add_term(7*l + 4, {0: 1, 1: 1}, {b_11: 1}, {c_00: 1, c_01: 1})
        # M6
        add_term(7*l + 5, {2: 1, 0: -1}, {b_00: 1, b_01: 1}, {c_11: 1})
        # M7
        add_term(7*l + 6, {3: 1, 1: -1}, {b_10: 1, b_11: 1}, {c_00: -1, c_11: -1})
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
