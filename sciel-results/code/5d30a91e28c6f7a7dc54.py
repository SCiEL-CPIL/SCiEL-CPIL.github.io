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
        """Add 7 Strassen terms for a 2x2x2 block at (r1,c1,k1)."""
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
        
        # M1 = (A00+A11)(B00+B11) -> C00 + C11
        add_term({A00: 1, A11: 1}, {B00: 1, B11: 1}, {C00: 1, C11: 1})
        # M2 = (A10+A11)B00 -> C10 - C11
        add_term({A10: 1, A11: 1}, {B00: 1}, {C10: 1, C11: -1})
        # M3 = A00(B01-B11) -> C01 + C11
        add_term({A00: 1}, {B01: 1, B11: -1}, {C01: 1, C11: 1})
        # M4 = A11(B10-B00) -> C00 + C10
        add_term({A11: 1}, {B10: 1, B00: -1}, {C00: 1, C10: 1})
        # M5 = (A00+A01)B11 -> -C00 + C01
        add_term({A00: 1, A01: 1}, {B11: 1}, {C00: -1, C01: 1})
        # M6 = (A10-A00)(B00+B01) -> C11
        add_term({A10: 1, A00: -1}, {B00: 1, B01: 1}, {C11: 1})
        # M7 = (A01-A11)(B10+B11) -> C00
        add_term({A01: 1, A11: -1}, {B10: 1, B11: 1}, {C00: 1})
    
    def add_strassen_2x2x1(r1, c1, k1):
        """
        2x2x1 block: A is 2x2, B is 2x1, C is 2x1.
        A = [[a00, a01], [a10, a11]], B = [b0, b1]^T, C = [c0, c1]^T
        c0 = a00*b0 + a01*b1
        c1 = a10*b0 + a11*b1
        
        3-term decomposition:
        M1 = (a00+a11)*b0 -> c0 += M1, c1 += M1
        M2 = (a10+a11)*b0 -> c1 += M2, c0 -= M2 (wait, let me redo)
        
        Actually for 2x2 times 2x1:
        c0 = a00*b0 + a01*b1
        c1 = a10*b0 + a11*b1
        
        3 terms:
        M1 = (a00+a10)*b0  -> c0 += M1*a00/(a00+a10)? No, that's not linear.
        
        Let me think differently. We need:
        c0 = a00*b0 + a01*b1
        c1 = a10*b0 + a11*b1
        
        3-term:
        M1 = (a00+a11)*b0: contributes to both c0 and c1
        M2 = (a10-a00)*b0: adjusts c0 and c1
        M3 = (a01-a11)*b1: adjusts c0 and c1
        
        c0 = M1 + M2 - M3  => (a00+a11)*b0 + (a10-a00)*b0 - (a01-a11)*b1
            = (a00+a11+a10-a00)*b0 + (-a01+a11)*b1
            = (a10+a11)*b0 + (a11-a01)*b1  -- not quite right
        
        Let me try:
        M1 = a00*b0
        M2 = a01*b1  
        M3 = a10*b0 + a11*b1  -- but this is 2 terms in one, not allowed
        
        For a 2x2 * 2x1 multiplication, the tensor has 4 components (2*2*1=4).
        Minimum rank for this is 3 (since it's a 2x2x1 tensor, rank can be 3).
        
        3-term decomposition:
        M1 = (a00+a11)*b0
        M2 = (a10+a11)*b0  
        M3 = (a01-a11)*b1
        
        c0 = M1 + M3 = (a00+a11)*b0 + (a01-a11)*b1 = a00*b0 + a11*b0 + a01*b1 - a11*b1
        That's not right either.
        
        Let me just use schoolbook (4 terms) for now, or try:
        M1 = (a00+a01)*b0
        M2 = (a10+a11)*b0
        M3 = (a01-a11)*b1
        
        c0 = M1 + M3 = (a00+a01)*b0 + (a01-a11)*b1 = a00*b0 + a01*b0 + a01*b1 - a11*b1
        Not right.
        
        OK, for 2x2x1, let's just use 4 schoolbook terms.
        """
        # Schoolbook: 4 terms
        for ii in range(2):
            for jj in range(2):
                u_idx = (r1 + ii) * m + (c1 + jj)
                v_idx = (c1 + jj) * p + k1
                w_idx = k1 * n + (r1 + ii)
                add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
    
    def add_strassen_2x1x2(r1, c1, k1):
        """
        2x1x2 block: A is 2x1, B is 1x2, C is 2x2.
        A = [a0, a1]^T, B = [b0, b1], C = [[c00, c01], [c10, c11]]
        c00 = a0*b0, c01 = a0*b1
        c10 = a1*b0, c11 = a1*b1
        
        This is an outer product, rank 1! But we need 4 separate outputs.
        Actually this is rank 4 in the tensor sense (4 independent terms).
        Wait: C = A * B where A is 2x1 and B is 1x2, so C = a0*b0, a0*b1, a1*b0, a1*b1
        Each output is a single product, so we need 4 terms.
        """
        # Schoolbook: 4 terms
        for ii in range(2):
            for kk in range(2):
                u_idx = (r1 + ii) * m + c1
                v_idx = c1 * p + (k1 + kk)
                w_idx = (k1 + kk) * n + (r1 + ii)
                add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
    
    def add_strassen_1x2x2(r1, c1, k1):
        """
        1x2x2 block: A is 1x2, B is 2x2, C is 1x2.
        A = [a0, a1], B = [[b00,b01],[b10,b11]], C = [c0, c1]
        c0 = a0*b00 + a1*b10
        c1 = a0*b01 + a1*b11
        
        3-term decomposition possible:
        M1 = a0*(b00+b10)
        M2 = a1*(b00+b10)
        M3 = a1*(b10-b00)  -- no
        
        Let me try:
        M1 = a0*(b00+b11)
        M2 = a1*(b00+b11)
        M3 = a1*(b10-b00)
        
        c0 = M1 + M3 = a0*(b00+b11) + a1*(b10-b00) = a0*b00 + a0*b11 + a1*b10 - a1*b00
        Not right.
        
        For 1x2x2: c0 = a0*b00 + a1*b10, c1 = a0*b01 + a1*b11
        3 terms:
        M1 = a0*(b00+b01)
        M2 = a1*(b10+b11)
        M3 = a0*(b01-b11)  -- mixing
        
        Actually, this is equivalent to multiplying a 1x2 row vector by a 2x2 matrix.
        The tensor is 1x2x2 with 4 entries. Rank is 3.
        
        M1 = a0*b00
        M2 = a0*b01
        M3 = a1*(b10+b11)
        
        c0 = M1 + (M3 - a1*b11) -- can't do that.
        
        Let me just use 4 schoolbook terms.
        """
        # Schoolbook: 4 terms
        for jj in range(2):
            for kk in range(2):
                u_idx = r1 * m + (c1 + jj)
                v_idx = (c1 + jj) * p + (k1 + kk)
                w_idx = (k1 + kk) * n + r1
                add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
    
    def add_schoolbook(r1, c1, k1, ar, ac, bc):
        """Add schoolbook terms for a block of size ar x ac x bc."""
        for ii in range(ar):
            for jj in range(ac):
                for kk in range(bc):
                    u_idx = (r1 + ii) * m + (c1 + jj)
                    v_idx = (c1 + jj) * p + (k1 + kk)
                    w_idx = (k1 + kk) * n + (r1 + ii)
                    add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
    
    # Iterate over all block combinations
    for r_idx, (r1, r2) in enumerate(blocks):
        for c_idx, (c1, c2) in enumerate(blocks):
            for k_idx, (k1, k2) in enumerate(blocks):
                if time.time() > deadline:
                    break
                ar = r2 - r1
                ac = c2 - c1
                bc = k2 - k1
                
                if ar == 2 and ac == 2 and bc == 2:
                    add_strassen_2x2(r1, c1, k1)
                elif ar == 2 and ac == 2 and bc == 1:
                    add_strassen_2x2x1(r1, c1, k1)
                elif ar == 2 and ac == 1 and bc == 2:
                    add_strassen_2x1x2(r1, c1, k1)
                elif ar == 1 and ac == 2 and bc == 2:
                    add_strassen_1x2x2(r1, c1, k1)
                else:
                    add_schoolbook(r1, c1, k1, ar, ac, bc)
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Remove duplicates
    seen = set()
    unique_terms = []
    for t in terms:
        u_dict, v_dict, w_dict = t
        key = (tuple(sorted(u_dict.items())), tuple(sorted(v_dict.items())), tuple(sorted(w_dict.items())))
        if key not in seen:
            seen.add(key)
            unique_terms.append(t)
    terms = unique_terms
    
    rank = len(terms)
    
    # Build matrices
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
