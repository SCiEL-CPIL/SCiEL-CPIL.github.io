import numpy as np

def construct():
    n, m, p = 2, 2, 8
    nr, mr, pr = n*m, m*p, p*n  # 4, 16, 16
    
    rank = 28
    U = np.zeros((nr, rank), dtype=float)
    V = np.zeros((mr, rank), dtype=float)
    W = np.zeros((pr, rank), dtype=float)
    
    # Strassen terms defined as:
    # (coeffs for A, coeffs for B, coeffs for C)
    # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # C indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B10-B11)
    # P4 = A11(B01-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    strassen_terms = [
        # (A_coeffs, B_coeffs, C_coeffs)
        # P1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # P2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # P3
        ({0: 1}, {2: 1, 3: -1}, {1: 1, 3: 1}),
        # P4
        ({3: 1}, {1: 1, 0: -1}, {0: 1, 2: 1}),
        # P5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # P6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # P7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    for q in range(4):
        k1 = 2 * q
        k2 = 2 * q + 1
        
        # Map local B indices (0,1,2,3) to global B indices
        # Local 0: (0, k1) -> global k1
        # Local 1: (0, k2) -> global k2
        # Local 2: (1, k1) -> global 8 + k1
        # Local 3: (1, k2) -> global 8 + k2
        b_map = {
            0: k1,
            1: k2,
            2: 8 + k1,
            3: 8 + k2
        }
        
        # Map local C indices (0,1,2,3) to global C indices
        # Local 0: (0, k1) -> global 2*k1
        # Local 1: (0, k2) -> global 2*k2
        # Local 2: (1, k1) -> global 2*k1 + 1
        # Local 3: (1, k2) -> global 2*k2 + 1
        c_map = {
            0: 2 * k1,
            1: 2 * k2,
            2: 2 * k1 + 1,
            3: 2 * k2 + 1
        }
        
        for t_idx, (A_c, B_c, C_c) in enumerate(strassen_terms):
            t = q * 7 + t_idx
            
            for a_local, coeff in A_c.items():
                U[a_local, t] = coeff
                
            for b_local, coeff in B_c.items():
                V[b_map[b_local], t] = coeff
                
            for c_local, coeff in C_c.items():
                W[c_map[c_local], t] = coeff

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
