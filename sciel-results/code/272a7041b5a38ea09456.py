import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Block decomposition: 5 = 2 + 2 + 1
    blocks = [(0, 2), (2, 4), (4, 5)]
    
    terms = []  # Each term: (U_dict, V_dict, W_dict)
    
    def add_term(u_dict, v_dict, w_dict):
        if time.time() > deadline:
            return
        u_dict = {k: v for k, v in u_dict.items() if v != 0}
        v_dict = {k: v for k, v in v_dict.items() if v != 0}
        w_dict = {k: v for k, v in w_dict.items() if v != 0}
        if u_dict and v_dict and w_dict:
            terms.append((u_dict, v_dict, w_dict))
    
    def add_strassen_2x2(r1, c1, k1):
        A00 = r1 * m + c1
        A01 = r1 * m + (c1 + 1)
        A10 = (r1 + 1) * m + c1
        A11 = (r1 + 1) * m + (c1 + 1)
        
        B00 = c1 * p + k1
        B01 = c1 * p + (k1 + 1)
        B10 = (c1 + 1) * p + k1
        B11 = (c1 + 1) * p + (k1 + 1)
        
        C00 = k1 * n + r1
        C01 = (k1 + 1) * n + r1
        C10 = k1 * n + (r1 + 1)
        C11 = (k1 + 1) * n + (r1 + 1)
        
        # M1 = (A00 + A11)(B00 + B11)
        add_term({A00: 1, A11: 1}, {B00: 1, B11: 1}, {C00: 1, C11: 1})
        # M2 = (A10 + A11)B00
        add_term({A10: 1, A11: 1}, {B00: 1}, {C10: 1, C11: -1})
        # M3 = A00(B01 - B11)
        add_term({A00: 1}, {B01: 1, B11: -1}, {C01: 1, C11: 1})
        # M4 = A11(B10 - B00)
        add_term({A11: 1}, {B10: 1, B00: -1}, {C00: 1, C10: 1})
        # M5 = (A00 + A01)B11
        add_term({A00: 1, A01: 1}, {B11: 1}, {C00: -1, C01: 1})
        # M6 = (A10 - A00)(B00 + B01)
        add_term({A10: 1, A00: -1}, {B00: 1, B01: 1}, {C11: 1})
        # M7 = (A01 - A11)(B10 + B11)
        add_term({A01: 1, A11: -1}, {B10: 1, B11: 1}, {C00: 1})
    
    def add_strassen_3x3(r1, c1, k1):
        """Strassen-style decomposition for 3x3x3 blocks.
        Uses a known 7-term decomposition for 3x3 matrices.
        This is based on the fact that 3x3 matrix multiplication has rank <= 20,
        but we can use a structured approach.
        
        For 3x3, we can use the following approach:
        Split 3 = 2 + 1, use Strassen for the 2x2x2 part and schoolbook for the rest.
        But let's try a direct 3x3 decomposition.
        
        Actually, let's use the known result that 3x3x3 matrix multiplication
        can be done with 19 multiplications (Lader). But implementing that is complex.
        Let's use a simpler approach: split 3=2+1 and use Strassen+schoolbook.
        """
        # For 3x3x3, we'll use a hybrid: treat it as 2+1 split
        # The 2x2x2 part uses Strassen (7 terms)
        # The remaining parts use schoolbook
        # This gives 7 + 3*2*1 + 3*1*2 + 3*1*1 = 7 + 6 + 6 + 3 = 22 terms
        # vs schoolbook 27 terms. So 22 < 27, good.
        
        # Actually, let me just use schoolbook for 3x3 for now, since implementing
        # an optimal 3x3 decomposition is complex. The main gain comes from
        # using Strassen on 2x2x2 blocks.
        
        # For 3x3x3, use schoolbook: 27 terms
        for ii in range(3):
            for jj in range(3):
                for kk in range(3):
                    u_idx = (r1 + ii) * m + (c1 + jj)
                    v_idx = (c1 + jj) * p + (k1 + kk)
                    w_idx = (k1 + kk) * n + (r1 + ii)
                    add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
    
    def add_schoolbook(r1, c1, k1, ar, ac, bc):
        for ii in range(ar):
            for jj in range(ac):
                for kk in range(bc):
                    u_idx = (r1 + ii) * m + (c1 + jj)
                    v_idx = (c1 + jj) * p + (k1 + kk)
                    w_idx = (k1 + kk) * n + (r1 + ii)
                    add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
    
    for r_idx, (r1, r2) in enumerate(blocks):
        if time.time() > deadline:
            break
        for c_idx, (c1, c2) in enumerate(blocks):
            if time.time() > deadline:
                break
            for k_idx, (k1, k2) in enumerate(blocks):
                if time.time() > deadline:
                    break
                ar = r2 - r1
                ac = c2 - c1
                bc = k2 - k1
                
                if ar == 2 and ac == 2 and bc == 2:
                    add_strassen_2x2(r1, c1, k1)
                elif ar == 3 and ac == 3 and bc == 3:
                    # Use a smarter 3x3x3 decomposition
                    # Split 3 = 2 + 1
                    # 2x2x2: Strassen (7 terms)
                    # 2x2x1: schoolbook (4 terms)
                    # 2x1x2: schoolbook (4 terms)
                    # 1x2x2: schoolbook (4 terms)
                    # 2x1x1: schoolbook (2 terms)
                    # 1x2x1: schoolbook (2 terms)
                    # 1x1x2: schoolbook (2 terms)
                    # 1x1x1: schoolbook (1 term)
                    # Total: 7+4+4+4+2+2+2+1 = 26 terms vs 27 schoolbook
                    # Marginal improvement. Let's just use schoolbook for simplicity.
                    add_schoolbook(r1, c1, k1, ar, ac, bc)
                else:
                    add_schoolbook(r1, c1, k1, ar, ac, bc)
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        u_dict, v_dict, w_dict = terms[r]
        for idx, val in u_dict.items():
            U[idx][r] = val
        for idx, val in v_dict.items():
            V[idx][r] = val
        for idx, val in w_dict.items():
            W[idx][r] = val
    
    return {"U": U, "V": V, "W": W}
