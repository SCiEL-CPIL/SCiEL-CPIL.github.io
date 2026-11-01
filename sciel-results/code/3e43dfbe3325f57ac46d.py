import time

DIMENSIONS = [2, 2, 3]

def construct():
    # Rank 11 decomposition using Strassen for the 2x2 block and schoolbook for the last column
    n, m, p = DIMENSIONS
    rank = 11
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Helper to set a term
    def set_term(r, u_idx_vals, v_idx_vals, w_idx_vals):
        for idx, val in u_idx_vals:
            U[idx][r] = val
        for idx, val in v_idx_vals:
            V[idx][r] = val
        for idx, val in w_idx_vals:
            W[idx][r] = val
            
    # Strassen terms for k=0,1
    # M1 = (A00+A11)(B00+B11)
    set_term(0, 
             [(0, 1), (3, 1)], 
             [(0, 1), (4, 1)], 
             [(0, 1), (3, 1)])
             
    # M2 = (A10+A11)B00
    set_term(1, 
             [(2, 1), (3, 1)], 
             [(0, 1)], 
             [(1, 1), (3, -1)])
             
    # M3 = A00(B10-B11)
    set_term(2, 
             [(0, 1)], 
             [(3, 1), (4, -1)], 
             [(2, 1), (3, 1)])
             
    # M4 = A11(B00-B01)
    set_term(3, 
             [(3, 1)], 
             [(0, 1), (1, -1)], 
             [(0, 1), (1, 1)])
             
    # M5 = (A00+A01)B11
    set_term(4, 
             [(0, 1), (1, 1)], 
             [(4, 1)], 
             [(0, -1), (2, 1)])
             
    # M6 = (A10-A00)(B00+B01)
    set_term(5, 
             [(2, 1), (0, -1)], 
             [(0, 1), (1, 1)], 
             [(3, 1)])
             
    # M7 = (A01-A11)(B10+B11)
    set_term(6, 
             [(1, 1), (3, -1)], 
             [(3, 1), (4, 1)], 
             [(0, -1)])
             
    # Schoolbook terms for k=2
    # C02 = A00*B02 + A01*B12
    # C12 = A10*B02 + A11*B12
    
    # T8: A00 B02 -> C02
    set_term(7, 
             [(0, 1)], 
             [(2, 1)], 
             [(4, 1)])
             
    # T9: A01 B12 -> C02
    set_term(8, 
             [(1, 1)], 
             [(5, 1)], 
             [(4, 1)])
             
    # T10: A10 B02 -> C12
    set_term(9, 
             [(2, 1)], 
             [(2, 1)], 
             [(5, 1)])
             
    # T11: A11 B12 -> C12
    set_term(10, 
             [(3, 1)], 
             [(5, 1)], 
             [(5, 1)])
             
    return {"U": U, "V": V, "W": W}
