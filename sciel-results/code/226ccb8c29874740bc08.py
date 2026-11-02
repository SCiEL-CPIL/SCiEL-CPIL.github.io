"""Matrix multiplication 4,4,9 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Construct a tensor decomposition for 4x4 x 4x9 matrix multiplication.
    Uses Strassen for the 4x4x4 sub-problems (rank 49 each) and schoolbook for 4x4x1 (rank 16).
    Total rank: 49 + 49 + 16 = 114
    Returns {'U', 'V', 'W'} with shapes (16, rank), (36, rank), (36, rank).
    """
    n, m, p = DIMENSIONS  # 4, 4, 9
    nm = n * m  # 16
    mp = m * p  # 36
    pn = p * n  # 36
    
    start_time = time.time()
    deadline = start_time + 100  # 100s deadline to be safe within 120s limit
    
    def make_u():
        return [0.0] * nm
    
    def make_v():
        return [0.0] * mp
    
    def make_w():
        return [0.0] * pn
    
    def u_idx(i, j):
        return i * m + j  # A is n x m = 4 x 4
    
    def v_idx(j, k):
        return j * p + k  # B is m x p = 4 x 9
    
    def w_idx(k, i):
        return k * n + i  # C is p x n = 9 x 4
    
    def strassen_2x2_2x2(a_row_offset, a_col_offset, b_row_offset, b_col_offset, 
                          c_row_offset, c_col_offset, b_col_base, signs):
        """
        Generate 7 Strassen terms for a 2x2 x 2x2 sub-multiplication within the larger matrices.
        
        The 2x2 A block is at A[a_row_offset:a_row_offset+2, a_col_offset:a_col_offset+2]
        The 2x2 B block is at B[b_row_offset:b_row_offset+2, b_col_base:b_col_base+2]
        The 2x2 C block is at C[c_row_offset:c_row_offset+2, c_col_offset:c_col_offset+2]
        
        signs: dict mapping block position to coefficient
          'C00': coefficient for C[c_row_offset, c_col_offset]
          'C01': coefficient for C[c_row_offset, c_col_offset+1]
          'C10': coefficient for C[c_row_offset+1, c_col_offset]
          'C11': coefficient for C[c_row_offset+1, c_col_offset+1]
        """
        terms = []
        
        # M1 = (A00+A11)(B00+B11)
        # Contributes to C00 and C11
        u = make_u()
        v = make_v()
        w = make_w()
        for di in range(2):
            for dj in range(2):
                u[u_idx(a_row_offset+di, a_col_offset+dj)] = 1.0
                u[u_idx(a_row_offset+1+di, a_col_offset+1+dj)] = 1.0
                v[v_idx(b_row_offset+di, b_col_base+dj)] = 1.0
                v[v_idx(b_row_offset+1+di, b_col_base+1+dj)] = 1.0
        if signs.get('C00', 0) != 0:
            w[w_idx(c_col_offset, c_row_offset)] = signs['C00']
        if signs.get('C11', 0) != 0:
            w[w_idx(c_col_offset+1, c_row_offset+1)] = signs['C11']
        terms.append((u, v, w))
        
        # M2 = (A10+A11)B00
        # Contributes to C10
        u = make_u()
        v = make_v()
        w = make_w()
        for di in range(2):
            for dj in range(2):
                u[u_idx(a_row_offset+1+di, a_col_offset+dj)] = 1.0
                u[u_idx(a_row_offset+1+di, a_col_offset+1+dj)] = 1.0
                v[v_idx(b_row_offset+di, b_col_base+dj)] = 1.0
        if signs.get('C10', 0) != 0:
            w[w_idx(c_col_offset, c_row_offset+1)] = signs['C10']
        terms.append((u, v, w))
        
        # M3 = A00(B10-B11)
        # Contributes to C01
        u = make_u()
        v = make_v()
        w = make_w()
        for di in range(2):
            for dj in range(2):
                u[u_idx(a_row_offset+di, a_col_offset+dj)] = 1.0
                v[v_idx(b_row_offset+1+di, b_col_base+dj)] = 1.0
                v[v_idx(b_row_offset+1+di, b_col_base+1+dj)] = -1.0
        if signs.get('C01', 0) != 0:
            w[w_idx(c_col_offset+1, c_row_offset)] = signs['C01']
        terms.append((u, v, w))
        
        # M4 = A11(B01-B00)
        # Contributes to C00 and C10
        u = make_u()
        v = make_v()
        w = make_w()
        for di in range(2):
            for dj in range(2):
                u[u_idx(a_row_offset+1+di, a_col_offset+1+dj)] = 1.0
                v[v_idx(b_row_offset+di, b_col_base+1+dj)] = 1.0
                v[v_idx(b_row_offset+di, b_col_base+dj)] = -1.0
        if signs.get('C00', 0) != 0:
            w[w_idx(c_col_offset, c_row_offset)] = signs['C00']
        if signs.get('C10', 0) != 0:
            w[w_idx(c_col_offset, c_row_offset+1)] = signs['C10']
        terms.append((u, v, w))
        
        # M5 = (A00+A01)B11
        # Contributes to C01 and -C00
        u = make_u()
        v = make_v()
        w = make_w()
        for di in range(2):
            for dj in range(2):
                u[u_idx(a_row_offset+di, a_col_offset+dj)] = 1.0
                u[u_idx(a_row_offset+di, a_col_offset+1+dj)] = 1.0
                v[v_idx(b_row_offset+1+di, b_col_base+1+dj)] = 1.0
        if signs.get('C01', 0) != 0:
            w[w_idx(c_col_offset+1, c_row_offset)] = signs['C01']
        if signs.get('C00', 0) != 0:
            w[w_idx(c_col_offset, c_row_offset)] = -signs['C00']
        terms.append((u, v, w))
        
        # M6 = (A10-A00)(B00+B01)
        # Contributes to C11
        u = make_u()
        v = make_v()
        w = make_w()
        for di in range(2):
            for dj in range(2):
                u[u_idx(a_row_offset+1+di, a_col_offset+dj)] = 1.0
                u[u_idx(a_row_offset+di, a_col_offset+dj)] = -1.0
                v[v_idx(b_row_offset+di, b_col_base+dj)] = 1.0
                v[v_idx(b_row_offset+di, b_col_base+1+dj)] = 1.0
        if signs.get('C11', 0) != 0:
            w[w_idx(c_col_offset+1, c_row_offset+1)] = signs['C11']
        terms.append((u, v, w))
        
        # M7 = (A01-A11)(B10+B11)
        # Contributes to C00 and C11
        u = make_u()
        v = make_v()
        w = make_w()
        for di in range(2):
            for dj in range(2):
                u[u_idx(a_row_offset+di, a_col_offset+1+dj)] = 1.0
                u[u_idx(a_row_offset+1+di, a_col_offset+1+dj)] = -1.0
                v[v_idx(b_row_offset+1+di, b_col_base+dj)] = 1.0
                v[v_idx(b_row_offset+1+di, b_col_base+1+dj)] = 1.0
        if signs.get('C00', 0) != 0:
            w[w_idx(c_col_offset, c_row_offset)] = signs['C00']
        if signs.get('C11', 0) != 0:
            w[w_idx(c_col_offset+1, c_row_offset+1)] = signs['C11']
        terms.append((u, v, w))
        
        return terms
    
    def strassen_4x4_4x4(b_col_base):
        """
        Generate 49 Strassen terms for 4x4 x 4x4 multiplication.
        B columns are at b_col_base to b_col_base+3.
        """
        terms = []
        
        # Block structure:
        # A = [[A00, A01], [A10, A11]] each 2x2
        # B = [[B00, B01], [B10, B11]] each 2x2 (columns b_col_base to b_col_base+3)
        # C = [[C00, C01], [C10, C11]] each 2x2
        
        # Strassen at block level:
        # S1 = (A00+A11)(B00+B11) -> C00 + C11
        # S2 = (A10+A11)B00 -> C10
        # S3 = A00(B10-B11) -> C01
        # S4 = A11(B01-B00) -> C00 + C10
        # S5 = (A00+A01)B11 -> C01 - C00
        # S6 = (A10-A00)(B00+B01) -> C11
        # S7 = (A01-A11)(B10+B11) -> C00 + C11
        
        # Each Si is a 2x2 x 2x2 multiplication, decomposed with 7 Strassen terms.
        
        # S1: (A00+A11)(B00+B11)
        # A factor: A00 at (0,0), A11 at (2,2)
        # B factor: B00 at (0,b_col_base), B11 at (2,b_col_base+2)
        # C: C00 at (0,0), C11 at (2,2)
        terms.extend(strassen_2x2_2x2(0, 0, 0, b_col_base, 0, 0, b_col_base, 
                                       {'C00': 1, 'C11': 1}))
        
        # S2: (A10+A11)B00
        # A factor: A10 at (2,0), A11 at (2,2)
        # B factor: B00 at (0,b_col_base)
        # C: C10 at (2,0)
        terms.extend(strassen_2x2_2x2(2, 0, 0, b_col_base, 2, 0, b_col_base,
                                       {'C10': 1}))
        
        # S3: A00(B10-B11)
        # A factor: A00 at (0,0)
        # B factor: B10 at (2,b_col_base), B11 at (2,b_col_base+2)
        # C: C01 at (0,2)
        terms.extend(strassen_2x2_2x2(0, 0, 2, b_col_base, 0, 2, b_col_base,
                                       {'C01': 1}))
        
        # S4: A11(B01-B00)
        # A factor: A11 at (2,2)
        # B factor: B01 at (0,b_col_base+2), B00 at (0,b_col_base)
        # C: C00 at (0,0), C10 at (2,0)
        terms.extend(strassen_2x2_2x2(2, 2, 0, b_col_base, 0, 0, b_col_base,
                                       {'C00': 1, 'C10': 1}))
        
        # S5: (A00+A01)B11
        # A factor: A00 at (0,0), A01 at (0,2)
        # B factor: B11 at (2,b_col_base+2)
        # C: C01 at (0,2), -C00 at (0,0)
        terms.extend(strassen_2x2_2x2(0, 0, 2, b_col_base, 0, 2, b_col_base,
                                       {'C01': 1, 'C00': -1}))
        
        # S6: (A10-A00)(B00+B01)
        # A factor: A10 at (2,0), A00 at (0,0)
        # B factor: B00 at (0,b_col_base), B01 at (0,b_col_base+2)
        # C: C11 at (2,2)
        terms.extend(strassen_2x2_2x2(2, 0, 0, b_col_base, 2, 2, b_col_base,
                                       {'C11': 1}))
        
        # S7: (A01-A11)(B10+B11)
        # A factor: A01 at (0,2), A11 at (2,2)
        # B factor: B10 at (2,b_col_base), B11 at (2,b_col_base+2)
        # C: C00 at (0,0), C11 at (2,2)
        terms.extend(strassen_2x2_2x2(0, 2, 2, b_col_base, 0, 0, b_col_base,
                                       {'C00': 1, 'C11': 1}))
        
        return terms
    
    # Build all terms
    terms = []
    
    # Part 1: 4x4 x 4x4 for columns 0-3 (rank 49)
    terms.extend(strassen_4x4_4x4(0))
    
    # Check deadline
    if time.time() > deadline:
        # Fallback to schoolbook
        pass
    
    # Part 2: 4x4 x 4x4 for columns 4-7 (rank 49)
    terms.extend(strassen_4x4_4x4(4))
    
    # Check deadline
    if time.time() > deadline:
        pass
    
    # Part 3: 4x4 x 4x1 for column 8 (schoolbook, rank 16)
    for i in range(4):
        for j in range(4):
            u = make_u()
            v = make_v()
            w = make_w()
            u[u_idx(i, j)] = 1.0
            v[v_idx(j, 8)] = 1.0
            w[w_idx(8, i)] = 1.0
            terms.append((u, v, w))
    
    rank = len(terms)
    
    # Build U, V, W matrices
    U = [[0.0] * rank for _ in range(nm)]
    V = [[0.0] * rank for _ in range(mp)]
    W = [[0.0] * rank for _ in range(pn)]
    
    for r, (u, v, w) in enumerate(terms):
        for i in range(nm):
            if u[i] != 0:
                U[i][r] = u[i]
        for j in range(mp):
            if v[j] != 0:
                V[j][r] = v[j]
        for k in range(pn):
            if w[k] != 0:
                W[k][r] = w[k]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
