import time

def construct():
    # Constants
    N = 4
    M = 4
    P = 9
    
    # U: (16, R), V: (36, R), W: (36, R)
    # Indices: U[i*M + j, r], V[j*P + k, r], W[k*N + i, r]
    
    terms = []
    
    # Helper to add a term with given U, V, W vectors
    def _add(u, v, w):
        terms.append((u, v, w))
    
    # Strassen 2x2 definitions
    # Each term is (A_mask, B_mask, C_mask)
    # Masks are lists of (row, col, coeff)
    s2x2 = [
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
    ]
    
    def _gen_4x4(b_off, c_off):
        """Generate 49 terms for 4x4 * 4x4 block."""
        block_terms = []
        for o_a, o_b, o_c in s2x2:
            for i_a, i_b, i_c in s2x2:
                u = [0.0] * 16
                v = [0.0] * 36
                w = [0.0] * 36
                
                # Expand outer and inner
                for (ar, ac, ca) in o_a:
                    for (br, bc, cb) in i_a:
                        # A index: (2*ar + br, 2*ac + bc)
                        idx_u = (2*ar + br)*4 + (2*ac + bc)
                        u[idx_u] += ca * cb
                        
                for (ar, ac, ca) in o_b:
                    for (br, bc, cb) in i_b:
                        # B index: (2*ar + br, b_off + 2*ac + bc)
                        idx_v = (2*ar + br)*9 + (b_off + 2*ac + bc)
                        v[idx_v] += ca * cb
                        
                for (ar, ac, ca) in o_c:
                    for (br, bc, cb) in i_c:
                        # C index: (c_off + 2*ac + bc, 2*ar + br)
                        # W index: k*N + i -> (col)*4 + row
                        idx_w = (c_off + 2*ac + bc)*4 + (2*ar + br)
                        w[idx_w] += ca * cb
                        
                block_terms.append((u, v, w))
        return block_terms

    start = time.time()
    # Deadline guard
    deadline = start + 100
    
    if time.time() > deadline:
        # Fallback: Schoolbook 144
        for i in range(4):
            for j in range(4):
                for k in range(9):
                    u = [0.0]*16
                    v = [0.0]*36
                    w = [0.0]*36
                    u[i*4+j] = 1.0
                    v[j*9+k] = 1.0
                    w[k*4+i] = 1.0
                    _add(u, v, w)
    else:
        # Block 1: cols 0-3 of B, cols 0-3 of C
        t1 = _gen_4x4(0, 0)
        # Block 2: cols 4-7 of B, cols 4-7 of C
        t2 = _gen_4x4(4, 4)
        
        for t in t1:
            _add(t[0], t[1], t[2])
        for t in t2:
            _add(t[0], t[1], t[2])
            
        # Block 3: col 8 of B, col 8 of C (4x1 mult)
        # Schoolbook for 4x4 * 4x1
        for i in range(4):
            for j in range(4):
                u = [0.0]*16
                v = [0.0]*36
                w = [0.0]*36
                u[i*4+j] = 1.0
                v[j*9+8] = 1.0
                w[8*4+i] = 1.0
                _add(u, v, w)

    R = len(terms)
    U = [[terms[r][0][i] for r in range(R)] for i in range(16)]
    V = [[terms[r][1][i] for r in range(R)] for i in range(36)]
    W = [[terms[r][2][i] for r in range(R)] for i in range(36)]
    
    return {"U": U, "V": V, "W": W}
