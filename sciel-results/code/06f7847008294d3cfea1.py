import time

DIMENSIONS = [2, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    n, m, p = DIMENSIONS
    # Target rank is 36
    # U: (n*m, rank) = (8, 36)
    # V: (m*p, rank) = (20, 36)
    # W: (p*n, rank) = (10, 36)
    
    U = [[0] * 36 for _ in range(n * m)]
    V = [[0] * 36 for _ in range(m * p)]
    W = [[0] * 36 for _ in range(p * n)]
    
    # Helper to add a rank-1 term
    # u_idx: index in U (i*m + j)
    # v_idx: index in V (j*p + k)
    # w_idx: index in W (k*n + i)
    # coeff: coefficient for this term
    def add_term(col, u_idx, v_idx, w_idx, coeff):
        U[u_idx][col] += coeff
        V[v_idx][col] += coeff
        W[w_idx][col] += coeff

    # Strassen 2x2x2 decomposition
    # n=2, m=2, p=2
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}
    # U idx: i*2+j
    # V idx: j*2+k
    # W idx: k*2+i
    
    def add_strassen_222(base_u, base_v, base_w, col_offset):
        # base_u: offset for i in U index (i*m + j)
        # base_v: offset for j in V index (j*p + k)
        # base_w: offset for k in W index (k*n + i)
        
        # M1 = (A00+A11)(B00+B11) -> C00, C11
        # u: (1,0,1,0) -> indices 0, 2
        # v: (1,0,0,1) -> indices 0, 3
        # w: (1,0,0,1) -> indices 0, 3
        add_term(col_offset + 0, base_u + 0, base_v + 0, base_w + 0, 1)
        add_term(col_offset + 0, base_u + 2, base_v + 0, base_w + 0, 1)
        add_term(col_offset + 0, base_u + 0, base_v + 3, base_w + 0, 1)
        add_term(col_offset + 0, base_u + 2, base_v + 3, base_w + 0, 1)
        add_term(col_offset + 0, base_u + 0, base_v + 0, base_w + 3, 1)
        add_term(col_offset + 0, base_u + 2, base_v + 0, base_w + 3, 1)
        add_term(col_offset + 0, base_u + 0, base_v + 3, base_w + 3, 1)
        add_term(col_offset + 0, base_u + 2, base_v + 3, base_w + 3, 1)
        
        # M2 = (C11+A11)B00 -> C10
        # u: (0,0,1,1) -> indices 2, 3
        # v: (1,0,0,0) -> index 0
        # w: (0,0,1,0) -> index 2 (k=1, i=0)
        add_term(col_offset + 1, base_u + 2, base_v + 0, base_w + 2, 1)
        add_term(col_offset + 1, base_u + 3, base_v + 0, base_w + 2, 1)
        
        # M3 = A00(B01-B11) -> C01
        # u: (1,0,0,0) -> index 0
        # v: (0,1,0,-1) -> indices 1, 3
        # w: (0,1,0,0) -> index 1 (k=0, i=1)
        add_term(col_offset + 2, base_u + 0, base_v + 1, base_w + 1, 1)
        add_term(col_offset + 2, base_u + 0, base_v + 3, base_w + 1, -1)
        
        # M4 = A11(B10-B00) -> C10, C11
        # u: (0,0,0,1) -> index 3
        # v: (0,0,1,-1) -> indices 2, 0
        # w: (0,0,1,1) -> indices 2, 3
        add_term(col_offset + 3, base_u + 3, base_v + 2, base_w + 2, 1)
        add_term(col_offset + 3, base_u + 3, base_v + 0, base_w + 2, -1)
        add_term(col_offset + 3, base_u + 3, base_v + 2, base_w + 3, 1)
        add_term(col_offset + 3, base_u + 3, base_v + 0, base_w + 3, -1)
        
        # M5 = (A00+A01)B11 -> C00, C01
        # u: (1,1,0,0) -> indices 0, 1
        # v: (0,0,0,1) -> index 3
        # w: (1,1,0,0) -> indices 0, 1
        add_term(col_offset + 4, base_u + 0, base_v + 3, base_w + 0, 1)
        add_term(col_offset + 4, base_u + 1, base_v + 3, base_w + 0, 1)
        add_term(col_offset + 4, base_u + 0, base_v + 3, base_w + 1, 1)
        add_term(col_offset + 4, base_u + 1, base_v + 3, base_w + 1, 1)
        
        # M6 = (A10-A00)(B00+B01) -> C11
        # u: (0,-1,1,0) -> indices 1, 2
        # v: (1,1,0,0) -> indices 0, 1
        # w: (0,0,0,1) -> index 3
        add_term(col_offset + 5, base_u + 1, base_v + 0, base_w + 3, -1)
        add_term(col_offset + 5, base_u + 2, base_v + 0, base_w + 3, 1)
        add_term(col_offset + 5, base_u + 1, base_v + 1, base_w + 3, -1)
        add_term(col_offset + 5, base_u + 2, base_v + 1, base_w + 3, 1)
        
        # M7 = (A01-A11)(B10+B11) -> C00, C11
        # u: (0,1,0,-1) -> indices 1, 3
        # v: (0,0,1,1) -> indices 2, 3
        # w: (1,0,0,1) -> indices 0, 3
        add_term(col_offset + 6, base_u + 1, base_v + 2, base_w + 0, 1)
        add_term(col_offset + 6, base_u + 3, base_v + 2, base_w + 0, -1)
        add_term(col_offset + 6, base_u + 1, base_v + 3, base_w + 0, 1)
        add_term(col_offset + 6, base_u + 3, base_v + 3, base_w + 0, -1)
        add_term(col_offset + 6, base_u + 1, base_v + 2, base_w + 3, 1)
        add_term(col_offset + 6, base_u + 3, base_v + 2, base_w + 3, -1)
        add_term(col_offset + 6, base_u + 1, base_v + 3, base_w + 3, 1)
        add_term(col_offset + 6, base_u + 3, base_v + 3, base_w + 3, -1)

    # Schoolbook 2x2x1 decomposition
    # n=2, m=2, p=1
    # Indices: i in {0,1}, j in {0,1}, k=0
    # U idx: i*2+j
    # V idx: j*1+0 = j
    # W idx: 0*2+i = i
    
    def add_schoolbook_221(base_u, base_v, base_w, col_offset):
        # Term 1: A00 B0 C0
        add_term(col_offset + 0, base_u + 0, base_v + 0, base_w + 0, 1)
        # Term 2: A01 B1 C0
        add_term(col_offset + 1, base_u + 1, base_v + 1, base_w + 0, 1)
        # Term 3: A10 B0 C1
        add_term(col_offset + 2, base_u + 2, base_v + 0, base_w + 1, 1)
        # Term 4: A11 B1 C1
        add_term(col_offset + 3, base_u + 3, base_v + 1, base_w + 1, 1)

    # 2x2x5 decomposition
    # n=2, m=2, p=5
    # k in {0,1,2,3,4}
    # Split k into {0,1}, {2,3}, {4}
    
    def add_225(base_u, base_v, base_w, col_offset):
        # base_u: offset for i in U (i*2+j)
        # base_v: offset for j in V (j*5+k)
        # base_w: offset for k in W (k*2+i)
        
        # Part 1: k in {0,1} -> Strassen 2x2x2
        # Local j in {0,1}, local k in {0,1}
        # Global j = base_j + local_j
        # Global k = base_k + local_k
        
        # We need to map local indices to global offsets.
        # For V: index is j*p + k.
        # Local V index for Strassen is j*2 + k.
        # We need to add terms to global V at j*5 + k.
        
        # Instead of generic mapping, we will explicitly call add_strassen_222
        # but we need to adjust the V and W indices.
        
        # Let's create a wrapper for Strassen that allows custom index mapping for V and W.
        # Or just implement the terms directly here.
        
        # Strassen M1: (A00+A11)(B00+B11)
        # B00 is V[0*5+0], B11 is V[1*5+1]
        # C00 is W[0*2+0], C11 is W[1*2+1]
        
        # M1
        add_term(col_offset + 0, base_u + 0, base_v + 0*5 + 0, base_w + 0*2 + 0, 1)
        add_term(col_offset + 0, base_u + 2, base_v + 0*5 + 0, base_w + 0*2 + 0, 1)
        add_term(col_offset + 0, base_u + 0, base_v + 1*5 + 1, base_w + 0*2 + 0, 1)
        add_term(col_offset + 0, base_u + 2, base_v + 1*5 + 1, base_w + 0*2 + 0, 1)
        add_term(col_offset + 0, base_u + 0, base_v + 0*5 + 0, base_w + 1*2 + 1, 1)
        add_term(col_offset + 0, base_u + 2, base_v + 0*5 + 0, base_w + 1*2 + 1, 1)
        add_term(col_offset + 0, base_u + 0, base_v + 1*5 + 1, base_w + 1*2 + 1, 1)
        add_term(col_offset + 0, base_u + 2, base_v + 1*5 + 1, base_w + 1*2 + 1, 1)
        
        # M2: (C11+A11)B00 -> C10
        # B00 is V[0]
        # C10 is W[1*2+0]
        add_term(col_offset + 1, base_u + 2, base_v + 0*5 + 0, base_w + 1*2 + 0, 1)
        add_term(col_offset + 1, base_u + 3, base_v + 0*5 + 0, base_w + 1*2 + 0, 1)
        
        # M3: A00(B01-B11) -> C01
        # B01 is V[1], B11 is V[6]
        # C01 is W[0*2+1]
        add_term(col_offset + 2, base_u + 0, base_v + 0*5 + 1, base_w + 0*2 + 1, 1)
        add_term(col_offset + 2, base_u + 0, base_v + 1*5 + 1, base_w + 0*2 + 1, -1)
        
        # M4: A11(B10-B00) -> C10, C11
        # B10 is V[5], B00 is V[0]
        # C10 is W[2], C11 is W[3]
        add_term(col_offset + 3, base_u + 3, base_v + 1*5 + 0, base_w + 1*2 + 0, 1)
        add_term(col_offset + 3, base_u + 3, base_v + 0*5 + 0, base_w + 1*2 + 0, -1)
        add_term(col_offset + 3, base_u + 3, base_v + 1*5 + 0, base_w + 1*2 + 1, 1)
        add_term(col_offset + 3, base_u + 3, base_v + 0*5 + 0, base_w + 1*2 + 1, -1)
        
        # M5: (A00+A01)B11 -> C00, C01
        # B11 is V[6]
        # C00 is W[0], C01 is W[1]
        add_term(col_offset + 4, base_u + 0, base_v + 1*5 + 1, base_w + 0*2 + 0, 1)
        add_term(col_offset + 4, base_u + 1, base_v + 1*5 + 1, base_w + 0*2 + 0, 1)
        add_term(col_offset + 4, base_u + 0, base_v + 1*5 + 1, base_w + 0*2 + 1, 1)
        add_term(col_offset + 4, base_u + 1, base_v + 1*5 + 1, base_w + 0*2 + 1, 1)
        
        # M6: (A10-A00)(B00+B01) -> C11
        # B00 is V[0], B01 is V[1]
        # C11 is W[3]
        add_term(col_offset + 5, base_u + 1, base_v + 0*5 + 0, base_w + 1*2 + 1, -1)
        add_term(col_offset + 5, base_u + 2, base_v + 0*5 + 0, base_w + 1*2 + 1, 1)
        add_term(col_offset + 5, base_u + 1, base_v + 0*5 + 1, base_w + 1*2 + 1, -1)
        add_term(col_offset + 5, base_u + 2, base_v + 0*5 + 1, base_w + 1*2 + 1, 1)
        
        # M7: (A01-A11)(B10+B11) -> C00, C11
        # B10 is V[5], B11 is V[6]
        # C00 is W[0], C11 is W[3]
        add_term(col_offset + 6, base_u + 1, base_v + 1*5 + 0, base_w + 0*2 + 0, 1)
        add_term(col_offset + 6, base_u + 3, base_v + 1*5 + 0, base_w + 0*2 + 0, -1)
        add_term(col_offset + 6, base_u + 1, base_v + 1*5 + 1, base_w + 0*2 + 0, 1)
        add_term(col_offset + 6, base_u + 3, base_v + 1*5 + 1, base_w + 0*2 + 0, -1)
        add_term(col_offset + 6, base_u + 1, base_v + 1*5 + 0, base_w + 1*2 + 1, 1)
        add_term(col_offset + 6, base_u + 3, base_v + 1*5 + 0, base_w + 1*2 + 1, -1)
        add_term(col_offset + 6, base_u + 1, base_v + 1*5 + 1, base_w + 1*2 + 1, 1)
        add_term(col_offset + 6, base_u + 3, base_v + 1*5 + 1, base_w + 1*2 + 1, -1)
        
        # Part 2: k in {2,3} -> Strassen 2x2x2 shifted
        # Local k 0,1 maps to Global k 2,3
        # V indices: j*5 + (2+local_k)
        # W indices: (2+local_k)*2 + i
        
        # M1
        add_term(col_offset + 7, base_u + 0, base_v + 0*5 + 2, base_w + 2*2 + 0, 1)
        add_term(col_offset + 7, base_u + 2, base_v + 0*5 + 2, base_w + 2*2 + 0, 1)
        add_term(col_offset + 7, base_u + 0, base_v + 1*5 + 3, base_w + 2*2 + 0, 1)
        add_term(col_offset + 7, base_u + 2, base_v + 1*5 + 3, base_w + 2*2 + 0, 1)
        add_term(col_offset + 7, base_u + 0, base_v + 0*5 + 2, base_w + 3*2 + 1, 1)
        add_term(col_offset + 7, base_u + 2, base_v + 0*5 + 2, base_w + 3*2 + 1, 1)
        add_term(col_offset + 7, base_u + 0, base_v + 1*5 + 3, base_w + 3*2 + 1, 1)
        add_term(col_offset + 7, base_u + 2, base_v + 1*5 + 3, base_w + 3*2 + 1, 1)
        
        # M2
        add_term(col_offset + 8, base_u + 2, base_v + 0*5 + 2, base_w + 2*2 + 0, 1)
        add_term(col_offset + 8, base_u + 3, base_v + 0*5 + 2, base_w + 2*2 + 0, 1)
        
        # M3
        add_term(col_offset + 9, base_u + 0, base_v + 0*5 + 3, base_w + 2*2 + 1, 1)
        add_term(col_offset + 9, base_u + 0, base_v + 1*5 + 3, base_w + 2*2 + 1, -1)
        
        # M4
        add_term(col_offset + 10, base_u + 3, base_v + 1*5 + 2, base_w + 2*2 + 0, 1)
        add_term(col_offset + 10, base_u + 3, base_v + 0*5 + 2, base_w + 2*2 + 0, -1)
        add_term(col_offset + 10, base_u + 3, base_v + 1*5 + 2, base_w + 3*2 + 1, 1)
        add_term(col_offset + 10, base_u + 3, base_v + 0*5 + 2, base_w + 3*2 + 1, -1)
        
        # M5
        add_term(col_offset + 11, base_u + 0, base_v + 1*5 + 3, base_w + 2*2 + 0, 1)
        add_term(col_offset + 11, base_u + 1, base_v + 1*5 + 3, base_w + 2*2 + 0, 1)
        add_term(col_offset + 11, base_u + 0, base_v + 1*5 + 3, base_w + 2*2 + 1, 1)
        add_term(col_offset + 11, base_u + 1, base_v + 1*5 + 3, base_w + 2*2 + 1, 1)
        
        # M6
        add_term(col_offset + 12, base_u + 1, base_v + 0*5 + 2, base_w + 3*2 + 1, -1)
        add_term(col_offset + 12, base_u + 2, base_v + 0*5 + 2, base_w + 3*2 + 1, 1)
        add_term(col_offset + 12, base_u + 1, base_v + 0*5 + 3, base_w + 3*2 + 1, -1)
        add_term(col_offset + 12, base_u + 2, base_v + 0*5 + 3, base_w + 3*2 + 1, 1)
        
        # M7
        add_term(col_offset + 13, base_u + 1, base_v + 1*5 + 2, base_w + 2*2 + 0, 1)
        add_term(col_offset + 13, base_u + 3, base_v + 1*5 + 2, base_w + 2*2 + 0, -1)
        add_term(col_offset + 13, base_u + 1, base_v + 1*5 + 3, base_w + 2*2 + 0, 1)
        add_term(col_offset + 13, base_u + 3, base_v + 1*5 + 3, base_w + 2*2 + 0, -1)
        add_term(col_offset + 13, base_u + 1, base_v + 1*5 + 2, base_w + 3*2 + 1, 1)
        add_term(col_offset + 13, base_u + 3, base_v + 1*5 + 2, base_w + 3*2 + 1, -1)
        add_term(col_offset + 13, base_u + 1, base_v + 1*5 + 3, base_w + 3*2 + 1, 1)
        add_term(col_offset + 13, base_u + 3, base_v + 1*5 + 3, base_w + 3*2 + 1, -1)
        
        # Part 3: k in {4} -> Schoolbook 2x2x1 shifted
        # Local k 0 maps to Global k 4
        # V indices: j*5 + 4
        # W indices: 4*2 + i
        
        add_term(col_offset + 14, base_u + 0, base_v + 0*5 + 4, base_w + 4*2 + 0, 1)
        add_term(col_offset + 15, base_u + 1, base_v + 1*5 + 4, base_w + 4*2 + 0, 1)
        add_term(col_offset + 16, base_u + 2, base_v + 0*5 + 4, base_w + 4*2 + 1, 1)
        add_term(col_offset + 17, base_u + 3, base_v + 1*5 + 4, base_w + 4*2 + 1, 1)

    # Main 2x4x5 construction
    # Split j into {0,1} and {2,3}
    
    # Block 1: j in {0,1}
    # base_u: 0 (i*4 + j) -> i*4 + 0/1
    # base_v: 0 (j*5 + k) -> 0/1 * 5 + k
    # base_w: 0 (k*2 + i)
    add_225(0, 0, 0, 0)
    
    # Block 2: j in {2,3}
    # base_u: 2 (i*4 + 2/3)
    # base_v: 10 (2/3 * 5 + k) -> 10/15 + k
    # base_w: 0 (k*2 + i)
    add_225(2, 10, 0, 18)
    
    return {"U": U, "V": V, "W": W}
