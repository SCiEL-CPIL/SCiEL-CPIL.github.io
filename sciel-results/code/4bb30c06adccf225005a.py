# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 8
    rank = 44
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    current_rank = 0
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal current_rank
        r = current_rank
        current_rank += 1
        for idx, val in u_coeffs.items():
            U[idx][r] = val
        for idx, val in v_coeffs.items():
            V[idx][r] = val
        for idx, val in w_coeffs.items():
            W[idx][r] = val

    # Strassen 2x2x2 helper
    # A indices: (0,0), (0,1), (1,0), (1,1) -> U rows
    # B indices: (0,0), (0,1), (1,0), (1,1) -> V rows
    # C indices: (0,0), (0,1), (1,0), (1,1) -> W rows (offset by k*n+i)
    def strassen_2x2x2(u_offset_i, u_offset_j, v_offset_j, v_offset_k, w_offset_k, w_offset_i):
        # u_offset_i: row offset for A (i index)
        # u_offset_j: col offset for A (j index)
        # v_offset_j: row offset for B (j index)
        # v_offset_k: col offset for B (k index)
        # w_offset_k: row offset for C (k index)
        # w_offset_i: col offset for C (i index)
        
        # P1 = (A00 + A11)(B00 + B11)
        u = { (u_offset_i*3 + u_offset_j): 1.0, (u_offset_i+1)*3 + u_offset_j+1: 1.0 }
        v = { (v_offset_j*8 + v_offset_k): 1.0, (v_offset_j+1)*8 + v_offset_k+1: 1.0 }
        w = { (w_offset_k*2 + w_offset_i): 1.0, (w_offset_k*2 + w_offset_i+1): 1.0, 
              (w_offset_k+1)*2 + w_offset_i: 1.0, (w_offset_k+1)*2 + w_offset_i+1: 1.0 }
        # Wait, W indices are k*n + i. n=2.
        # C00: k=0, i=0 -> 0
        # C01: k=1, i=0 -> 2
        # C10: k=0, i=1 -> 1
        # C11: k=1, i=1 -> 3
        # My w dict above is wrong.
        
        # Let's define W indices properly.
        # W row = k * n + i = k * 2 + i.
        
        # P1 contributes to all 4 outputs.
        # C00: +1, C01: +1, C10: +1, C11: -1
        w = { 0: 1.0, 2: 1.0, 1: 1.0, 3: -1.0 }
        add_term(u, v, w)
        
        # P2 = (A10 + A11)B00
        u = { (u_offset_i+1)*3 + u_offset_j: 1.0, (u_offset_i+1)*3 + u_offset_j+1: 1.0 }
        v = { (v_offset_j*8 + v_offset_k): 1.0 }
        # C10: +1, C11: -1
        w = { 1: 1.0, 3: -1.0 }
        add_term(u, v, w)
        
        # P3 = A00(B01 - B11)
        u = { (u_offset_i*3 + u_offset_j): 1.0 }
        v = { (v_offset_j*8 + v_offset_k+1): 1.0, (v_offset_j+1)*8 + v_offset_k+1: -1.0 }
        # C01: +1, C11: +1
        w = { 2: 1.0, 3: 1.0 }
        add_term(u, v, w)
        
        # P4 = A11(B10 - B00)
        u = { (u_offset_i+1)*3 + u_offset_j+1: 1.0 }
        v = { (v_offset_j+1)*8 + v_offset_k: 1.0, (v_offset_j*8 + v_offset_k): -1.0 }
        # C00: +1, C10: +1
        w = { 0: 1.0, 1: 1.0 }
        add_term(u, v, w)
        
        # P5 = (A00 + A01)B11
        u = { (u_offset_i*3 + u_offset_j): 1.0, (u_offset_i*3 + u_offset_j+1): 1.0 }
        v = { (v_offset_j+1)*8 + v_offset_k+1: 1.0 }
        # C00: -1, C01: +1
        w = { 0: -1.0, 2: 1.0 }
        add_term(u, v, w)
        
        # P6 = (A10 - A00)(B00 + B01)
        u = { (u_offset_i+1)*3 + u_offset_j: 1.0, (u_offset_i*3 + u_offset_j): -1.0 }
        v = { (v_offset_j*8 + v_offset_k): 1.0, (v_offset_j*8 + v_offset_k+1): 1.0 }
        # C11: +1
        w = { 3: 1.0 }
        add_term(u, v, w)
        
        # P7 = (A01 - A11)(B10 + B11)
        u = { (u_offset_i*3 + u_offset_j+1): 1.0, (u_offset_i+1)*3 + u_offset_j+1: -1.0 }
        v = { (v_offset_j+1)*8 + v_offset_k: 1.0, (v_offset_j+1)*8 + v_offset_k+1: 1.0 }
        # C00: +1
        w = { 0: 1.0 }
        add_term(u, v, w)

    # 1. A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    strassen_2x2x2(0, 0, 0, 0, 0, 0)
    
    # 2. A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    strassen_2x2x2(0, 0, 0, 2, 2, 0)
    
    # 3. A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6]
    strassen_2x2x2(0, 0, 0, 4, 4, 0)
    
    # 4. A[0:2, 0:2] * B[0:2, 6:8] -> C[0:2, 6:8]
    strassen_2x2x2(0, 0, 0, 6, 6, 0)
    
    # 5. A[0:2, 2] * B[2, 0:8] -> C[0:2, 0:8]
    # A col 2, B row 2.
    # This is 2x1 * 1x8.
    # We can just do 16 trivial multiplications?
    # Or split into 4 blocks of 1x2?
    # Trivial is fine for now, 16 mults.
    for i in range(2):
        for k in range(8):
            u = { i * 3 + 2: 1.0 }
            v = { 2 * 8 + k: 1.0 }
            w = { k * 2 + i: 1.0 }
            add_term(u, v, w)
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
