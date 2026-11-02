import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=5
    # U: (n*m, rank) = (6, 28)
    # V: (m*p, rank) = (15, 28)
    # W: (p*n, rank) = (10, 28)
    
    rank = 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Helper to get flat indices
    def a_idx(i, j): return i * m + j
    def b_idx(j, k): return j * p + k
    def c_idx(k, i): return k * n + i
    
    # Term definitions: (U_indices, coeffs), (V_indices, coeffs), (W_indices, coeffs)
    # We will build lists of (idx, val) for each matrix
    
    terms = []
    
    # Part 1: M_{2,3,3} for cols 0,1,2 of B
    # Sub-part 1.1: M_{2,3,2} for cols 0,1 of B (result cols 0,1)
    # A_L: a00, a01, a10, a11
    # A_R: a02, a12
    # B_L: b00, b01, b10, b11 (rows 0,1, cols 0,1)
    # B_R: b20, b21 (row 2, cols 0,1)
    
    # Strassen 7 terms
    # M1: (a00+a11)(b00+b11) -> C00, C11
    terms.append(([ (a_idx(0,0), 1), (a_idx(1,1), 1) ], 
                 [ (b_idx(0,0), 1), (b_idx(1,1), 1) ],
                 [ (c_idx(0,0), 1), (c_idx(1,1), 1) ]))
                 
    # M2: (a10+a11)b00 -> C10, C11
    terms.append(([ (a_idx(1,0), 1), (a_idx(1,1), 1) ], 
                 [ (b_idx(0,0), 1) ],
                 [ (c_idx(0,1), -1), (c_idx(1,0), 1) ])) # Wait, C11 has -M2. C10 has +M2.
    # Let's re-verify C11 formula: M1 - M2 + M3 + M6 + M11.
    # C10 formula: M2 + M4 + M10.
    # So for M2: W at C10 is 1, W at C11 is -1.
    
    # M3: a00(b01-b11) -> C01, C11
    terms.append(([ (a_idx(0,0), 1) ], 
                 [ (b_idx(0,1), 1), (b_idx(1,1), -1) ],
                 [ (c_idx(0,1), 1), (c_idx(1,1), 1) ]))
                 
    # M4: a11(b10-b00) -> C00, C10
    terms.append(([ (a_idx(1,1), 1) ], 
                 [ (b_idx(1,0), 1), (b_idx(0,0), -1) ],
                 [ (c_idx(0,0), 1), (c_idx(1,0), 1) ]))
                 
    # M5: (a00+a01)b11 -> C00, C01
    terms.append(([ (a_idx(0,0), 1), (a_idx(0,1), 1) ], 
                 [ (b_idx(1,1), 1) ],
                 [ (c_idx(0,0), -1), (c_idx(0,1), 1) ]))
                 
    # M6: (a10-a00)(b00+b01) -> C11
    terms.append(([ (a_idx(1,0), 1), (a_idx(0,0), -1) ], 
                 [ (b_idx(0,0), 1), (b_idx(0,1), 1) ],
                 [ (c_idx(1,1), 1) ]))
                 
    # M7: (a01-a11)(b10+b11) -> C00
    terms.append(([ (a_idx(0,1), 1), (a_idx(1,1), -1) ], 
                 [ (b_idx(1,0), 1), (b_idx(1,1), 1) ],
                 [ (c_idx(0,0), 1) ]))
                 
    # Outer 4 terms
    # M8: a02 b20 -> C00
    terms.append(([ (a_idx(0,2), 1) ], 
                 [ (b_idx(2,0), 1) ],
                 [ (c_idx(0,0), 1) ]))
                 
    # M9: a02 b21 -> C01
    terms.append(([ (a_idx(0,2), 1) ], 
                 [ (b_idx(2,1), 1) ],
                 [ (c_idx(0,1), 1) ]))
                 
    # M10: a12 b20 -> C10
    terms.append(([ (a_idx(1,2), 1) ], 
                 [ (b_idx(2,0), 1) ],
                 [ (c_idx(1,0), 1) ]))
                 
    # M11: a12 b21 -> C11
    terms.append(([ (a_idx(1,2), 1) ], 
                 [ (b_idx(2,1), 1) ],
                 [ (c_idx(1,1), 1) ]))

    # Sub-part 1.2: M_{2,3,1} for col 2 of B (result col 2)
    # M12: a00 b02 -> C02
    terms.append(([ (a_idx(0,0), 1) ], 
                 [ (b_idx(0,2), 1) ],
                 [ (c_idx(2,0), 1) ]))
                 
    # M13: a01 b12 -> C02
    terms.append(([ (a_idx(0,1), 1) ], 
                 [ (b_idx(1,2), 1) ],
                 [ (c_idx(2,0), 1) ]))
                 
    # M14: a02 b22 -> C02
    terms.append(([ (a_idx(0,2), 1) ], 
                 [ (b_idx(2,2), 1) ],
                 [ (c_idx(2,0), 1) ]))
                 
    # M15: a10 b02 -> C12
    terms.append(([ (a_idx(1,0), 1) ], 
                 [ (b_idx(0,2), 1) ],
                 [ (c_idx(2,1), 1) ]))
                 
    # M16: a11 b12 -> C12
    terms.append(([ (a_idx(1,1), 1) ], 
                 [ (b_idx(1,2), 1) ],
                 [ (c_idx(2,1), 1) ]))
                 
    # M17: a12 b22 -> C12
    terms.append(([ (a_idx(1,2), 1) ], 
                 [ (b_idx(2,2), 1) ],
                 [ (c_idx(2,1), 1) ]))

    # Part 2: M_{2,3,2} for cols 3,4 of B (result cols 3,4)
    # B_L: b03, b04, b13, b14
    # B_R: b23, b24
    
    # M18: (a00+a11)(b03+b14) -> C03, C14
    terms.append(([ (a_idx(0,0), 1), (a_idx(1,1), 1) ], 
                 [ (b_idx(0,3), 1), (b_idx(1,4), 1) ],
                 [ (c_idx(3,0), 1), (c_idx(4,1), 1) ]))
                 
    # M19: (a10+a11)b03 -> C13, C14
    terms.append(([ (a_idx(1,0), 1), (a_idx(1,1), 1) ], 
                 [ (b_idx(0,3), 1) ],
                 [ (c_idx(3,1), 1), (c_idx(4,1), -1) ]))
                 
    # M20: a00(b04-b14) -> C04, C14
    terms.append(([ (a_idx(0,0), 1) ], 
                 [ (b_idx(0,4), 1), (b_idx(1,4), -1) ],
                 [ (c_idx(4,0), 1), (c_idx(4,1), 1) ]))
                 
    # M21: a11(b13-b03) -> C03, C13
    terms.append(([ (a_idx(1,1), 1) ], 
                 [ (b_idx(1,3), 1), (b_idx(0,3), -1) ],
                 [ (c_idx(3,0), 1), (c_idx(3,1), 1) ]))
                 
    # M22: (a00+a01)b14 -> C03, C04
    terms.append(([ (a_idx(0,0), 1), (a_idx(0,1), 1) ], 
                 [ (b_idx(1,4), 1) ],
                 [ (c_idx(3,0), -1), (c_idx(4,0), 1) ]))
                 
    # M23: (a10-a00)(b03+b04) -> C14
    terms.append(([ (a_idx(1,0), 1), (a_idx(0,0), -1) ], 
                 [ (b_idx(0,3), 1), (b_idx(0,4), 1) ],
                 [ (c_idx(4,1), 1) ]))
                 
    # M24: (a01-a11)(b13+b14) -> C03
    terms.append(([ (a_idx(0,1), 1), (a_idx(1,1), -1) ], 
                 [ (b_idx(1,3), 1), (b_idx(1,4), 1) ],
                 [ (c_idx(3,0), 1) ]))
                 
    # Outer
    # M25: a02 b23 -> C03
    terms.append(([ (a_idx(0,2), 1) ], 
                 [ (b_idx(2,3), 1) ],
                 [ (c_idx(3,0), 1) ]))
                 
    # M26: a02 b24 -> C04
    terms.append(([ (a_idx(0,2), 1) ], 
                 [ (b_idx(2,4), 1) ],
                 [ (c_idx(4,0), 1) ]))
                 
    # M27: a12 b23 -> C13
    terms.append(([ (a_idx(1,2), 1) ], 
                 [ (b_idx(2,3), 1) ],
                 [ (c_idx(3,1), 1) ]))
                 
    # M28: a12 b24 -> C14
    terms.append(([ (a_idx(1,2), 1) ], 
                 [ (b_idx(2,4), 1) ],
                 [ (c_idx(4,1), 1) ]))

    # Fill matrices
    for r, (u_list, v_list, w_list) in enumerate(terms):
        for idx, val in u_list:
            U[idx, r] += val
        for idx, val in v_list:
            V[idx, r] += val
        for idx, val in w_list:
            W[idx, r] += val
            
    return {"U": U, "V": V, "W": W}
