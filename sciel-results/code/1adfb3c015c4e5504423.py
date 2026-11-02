import numpy as np

def construct():
    n, m, p = 2, 2, 5
    # Rank 18 decomposition
    # 2 blocks of 2x2 (Strassen, rank 7 each) and 1 block of 2x1 (Schoolbook, rank 4)
    rank = 18
    
    U = [[0] * rank for _ in range(n * m)] # 4 x 18
    V = [[0] * rank for _ in range(m * p)] # 10 x 18
    W = [[0] * rank for _ in range(p * n)] # 10 x 18
    
    col = 0
    
    # Helper to add a term
    # u_vals: list of (idx, val) for U
    # v_vals: list of (idx, val) for V
    # w_vals: list of (idx, val) for W
    def add_term(u_vals, v_vals, w_vals):
        nonlocal col
        for idx, val in u_vals:
            U[idx][col] += val
        for idx, val in v_vals:
            V[idx][col] += val
        for idx, val in w_vals:
            W[idx][col] += val
        col += 1
        
    # Block 1: B cols 0, 1 (indices 0, 1 in B, which are V rows 0, 2? No)
    # B is 2x5.
    # V index: j*p + k.
    # j=0, k=0 -> V[0]
    # j=0, k=1 -> V[1]
    # j=1, k=0 -> V[2]
    # j=1, k=1 -> V[3]
    # C is 2x5.
    # W index: k*n + i.
    # k=0, i=0 -> W[0]
    # k=0, i=1 -> W[1]
    # k=1, i=0 -> W[2]
    # k=1, i=1 -> W[3]
    
    # A indices:
    # i=0, j=0 -> U[0] (a)
    # i=0, j=1 -> U[1] (b)
    # i=1, j=0 -> U[2] (c)
    # i=1, j=1 -> U[3] (d)
    
    # Strassen for Block 1 (cols 0, 1)
    # M1 = (a+b)(B00+B11)
    add_term(
        [(0, 1), (1, 1)], # a+b
        [(0, 1), (3, 1)], # B00+B11 (V[0] is j=0,k=0; V[3] is j=1,k=1)
        [(0, 1), (3, -1)] # C00 = M1 - M2 + M3 - M4? 
        # Wait, C00 (i=0,k=0) = M1 - M2 + M3 - M4
        # C01 (i=0,k=1) = M4 + M5
        # C10 (i=1,k=0) = M2 + M6
        # C11 (i=1,k=1) = M1 - M3 + M6 - M7
        # So for M1, W[0] += 1, W[3] += 1
    )
    
    # M2 = b(B00+B11)
    add_term(
        [(1, 1)], # b
        [(0, 1), (3, 1)], # B00+B11
        [(0, -1), (2, 1)] # C00 -= M2, C10 += M2 (W[2] is i=1,k=0)
    )
    
    # M3 = a(B10-B00)
    add_term(
        [(0, 1)], # a
        [(2, 1), (0, -1)], # B10-B00 (V[2] is j=1,k=0; V[0] is j=0,k=0)
        [(0, 1), (3, -1)] # C00 += M3, C11 -= M3
    )
    
    # M4 = (a+b)B11
    add_term(
        [(0, 1), (1, 1)], # a+b
        [(3, 1)], # B11
        [(0, -1), (2, 1)] # C00 -= M4, C01 += M4 (W[2] is i=0,k=1? No, W[2] is k=1,i=0)
        # Check W indices:
        # W[k*n+i]
        # k=0, i=0 -> 0
        # k=0, i=1 -> 1
        # k=1, i=0 -> 2
        # k=1, i=1 -> 3
        # C01 is i=0, k=1 -> W[2]
        # C10 is i=1, k=0 -> W[1]
        # C11 is i=1, k=1 -> W[3]
        # So C00=W[0], C10=W[1], C01=W[2], C11=W[3]
        
        # Recalculate W mapping:
        # C00 (i=0,k=0) -> W[0]
        # C10 (i=1,k=0) -> W[1]
        # C01 (i=0,k=1) -> W[2]
        # C11 (i=1,k=1) -> W[3]
        
        # M1: C00 += 1, C11 += 1 -> W[0]+=1, W[3]+=1
        # M2: C00 -= 1, C10 += 1 -> W[0]-=1, W[1]+=1
        # M3: C00 += 1, C11 -= 1 -> W[0]+=1, W[3]-=1
        # M4: C00 -= 1, C01 += 1 -> W[0]-=1, W[2]+=1
        # M5: C01 += 1 -> W[2]+=1
        # M6: C10 += 1, C11 += 1 -> W[1]+=1, W[3]+=1
        # M7: C11 -= 1 -> W[3]-=1
        
    )
    
    # Correcting M4 term
    # I already added M4 above with wrong W? No, I just defined it.
    # Let's re-do the adds properly in the code.
    
    pass
