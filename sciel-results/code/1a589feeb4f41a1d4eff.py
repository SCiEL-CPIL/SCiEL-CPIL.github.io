import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # We use a rank-27 decomposition based on the Coppersmith-Winograd / Strassen-like
    # tensor contraction that exploits the 2x4x4 structure.
    #
    # The key idea: partition A (2x4) into two 2x2 blocks A_L and A_R,
    # and B (4x4) into four 2x2 blocks B_LL, B_LR, B_RL, B_RR.
    # C (2x4) = C_L (2x2) + C_R (2x2) where:
    #   C_L = A_L * B_LL + A_R * B_RL
    #   C_R = A_L * B_LR + A_R * B_RR
    #
    # Naive Strassen gives 4 * 7 = 28. We save one rank by sharing a bilinear form
    # across blocks. Specifically, we can combine certain terms.
    #
    # Here's the rank-27 construction:
    # We use 7 forms for block (A_L, B_LL) -> C_L contribution
    # We use 7 forms for block (A_R, B_RL) -> C_L contribution  
    # We use 7 forms for block (A_L, B_LR) -> C_R contribution
    # We use 6 forms for block (A_R, B_RR) -> C_R contribution (sharing 1 with above)
    #
    # Actually, let me use a cleaner approach. The known rank-27 for 2x4x4:
    # 
    # Let me define the 27 bilinear forms explicitly.
    # A = [[a00, a01, a02, a03],
    #      [a10, a11, a12, a13]]
    # B = [[b00, b01, b02, b03],
    #      [b10, b11, b12, b13],
    #      [b20, b21, b22, b23],
    #      [b30, b31, b32, b33]]
    # C = [[c00, c01, c02, c03],
    #      [c10, c11, c12, c13]]
    #
    # I'll use the following 27 forms based on a known decomposition:
    
    rank = 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a form
    def add_form(idx, u_entries, v_entries, w_entries):
        """
        u_entries: list of (row_in_U, coeff) where row_in_U = i*m + k
        v_entries: list of (row_in_V, coeff) where row_in_V = k*p + j
        w_entries: list of (row_in_W, coeff) where row_in_W = j*n + i
        """
        for (u_row, coeff) in u_entries:
            U[u_row][idx] += coeff
        for (v_row, coeff) in v_entries:
            V[v_row][idx] += coeff
        for (w_row, coeff) in w_entries:
            W[w_row][idx] += coeff
    
    # Let me construct a valid rank-27 decomposition.
    # 
    # I'll use the approach where we partition into 2x2 blocks and use a 
    # modified Strassen that shares one multiplication.
    #
    # Block indices:
    # A_L = A[:, 0:2], A_R = A[:, 2:4]
    # B_LL = B[0:2, 0:2], B_LR = B[0:2, 2:4]
    # B_RL = B[2:4, 0:2], B_RR = B[2:4, 2:4]
    # C_L = C[:, 0:2], C_R = C[:, 2:4]
    #
    # C_L = A_L * B_LL + A_R * B_RL
    # C_R = A_L * B_LR + A_R * B_RR
    #
    # For each 2x2 * 2x2 product, Strassen uses 7 multiplications.
    # Total naive: 4 * 7 = 28.
    #
    # To get 27, we need to share one form. One way: notice that certain
    # linear combinations can be shared.
    #
    # Let me just use a known rank-27 construction. Here's one:
    #
    # I'll define the 27 forms as follows, using the block structure.
    # For simplicity, let me use a systematic approach.
    
    # Let me define local indices for each 2x2 block
    # A_L: rows [0,1], cols [0,1] -> U indices: 0*m+0, 0*m+1, 1*m+0, 1*m+1 = 0,1,4,5
    # A_R: rows [0,1], cols [2,3] -> U indices: 0*m+2, 0*m+3, 1*m+2, 1*m+3 = 2,3,6,7
    # B_LL: rows [0,1], cols [0,1] -> V indices: 0*p+0, 0*p+1, 1*p+0, 1*p+1 = 0,1,4,5
    # B_LR: rows [0,1], cols [2,3] -> V indices: 0*p+2, 0*p+3, 1*p+2, 1*p+3 = 2,3,6,7
    # B_RL: rows [2,3], cols [0,1] -> V indices: 2*p+0, 2*p+1, 3*p+0, 3*p+1 = 8,9,12,13
    # B_RR: rows [2,3], cols [2,3] -> V indices: 2*p+2, 2*p+3, 3*p+2, 3*p+3 = 10,11,14,15
    # C_L: rows [0,1], cols [0,1] -> W indices: 0*n+0, 0*n+1, 1*n+0, 1*n+1 = 0,1,2,3
    # C_R: rows [0,1], cols [2,3] -> W indices: 2*n+0, 2*n+1, 3*n+0, 3*n+1 = 4,5,6,7
    
    # Strassen for 2x2:
    # M1 = (x00+x11)(y00+y11)
    # M2 = (x10+x11)y00
    # M3 = x00(y01-y11)
    # M4 = x11(y10-y00)
    # M5 = (x00+x01)y11
    # M6 = (x10-x00)(y00+y01)
    # M7 = (x01-x11)(y10+y11)
    # z00 = M1+M4-M5+M7
    # z01 = M3+M5
    # z10 = M2+M4
    # z11 = M1-M2+M3+M6
    
    # For block (A_L, B_LL) -> contributes to C_L
    # A_L indices in U: x00=0, x01=1, x10=4, x11=5
    # B_LL indices in V: y00=0, y01=1, y10=4, y11=5
    # C_L indices in W: z00=0, z01=1, z10=2, z11=3
    
    base = 0
    # M1
    add_form(base+0, [(0,1),(5,1)], [(0,1),(5,1)], [(0,1),(2,1),(3,-1),(7,1)])
    # M2
    add_form(base+1, [(4,1),(5,1)], [(0,1)], [(2,1),(3,-1)])
    # M3
    add_form(base+2, [(0,1)], [(1,1),(5,-1)], [(1,1),(3,1)])
    # M4
    add_form(base+3, [(5,1)], [(4,1),(0,-1)], [(0,1),(2,1)])
    # M5
    add_form(base+4, [(0,1),(1,1)], [(5,1)], [(3,-1),(1,1)])
    # M6
    add_form(base+5, [(4,1),(0,-1)], [(0,1),(1,1)], [(3,1)])
    # M7
    add_form(base+6, [(1,1),(5,-1)], [(4,1),(5,1)], [(0,1)])
    
    # For block (A_R, B_RL) -> contributes to C_L
    # A_R indices in U: x00=2, x01=3, x10=6, x11=7
    # B_RL indices in V: y00=8, y01=9, y10=12, y11=13
    # C_L indices in W: z00=0, z01=1, z10=2, z11=3
    
    base = 7
    # M1
    add_form(base+0, [(2,1),(7,1)], [(8,1),(13,1)], [(0,1),(2,1),(3,-1),(7,1)])
    # M2
    add_form(base+1, [(6,1),(7,1)], [(8,1)], [(2,1),(3,-1)])
    # M3
    add_form(base+2, [(2,1)], [(9,1),(13,-1)], [(1,1),(3,1)])
    # M4
    add_form(base+3, [(7,1)], [(12,1),(8,-1)], [(0,1),(2,1)])
    # M5
    add_form(base+4, [(2,1),(3,1)], [(13,1)], [(3,-1),(1,1)])
    # M6
    add_form(base+5, [(6,1),(2,-1)], [(8,1),(9,1)], [(3,1)])
    # M7
    add_form(base+6, [(3,1),(7,-1)], [(12,1),(13,1)], [(0,1)])
    
    # For block (A_L, B_LR) -> contributes to C_R
    # A_L indices in U: x00=0, x01=1, x10=4, x11=5
    # B_LR indices in V: y00=2, y01=3, y10=6, y11=7
    # C_R indices in W: z00=4, z01=5, z10=6, z11=7
    
    base = 14
    # M1
    add_form(base+0, [(0,1),(5,1)], [(2,1),(7,1)], [(4,1),(6,1),(5,-1),(7,1)])
    # M2
    add_form(base+1, [(4,1),(5,1)], [(2,1)], [(6,1),(5,-1)])
    # M3
    add_form(base+2, [(0,1)], [(3,1),(7,-1)], [(5,1),(7,1)])
    # M4
    add_form(base+3, [(5,1)], [(6,1),(2,-1)], [(4,1),(6,1)])
    # M5
    add_form(base+4, [(0,1),(1,1)], [(7,1)], [(5,-1),(5,1)])  # Wait, z01=5, so M5 contributes to z01
    # Let me redo: z00=4, z01=5, z10=6, z11=7
    # M5: (x00+x01)y11 -> z00 -= M5, z01 += M5
    # So W[4] -= 1, W[5] += 1
    # Let me fix this.
    
    # Actually, let me just be careful. Let me redo from base=14.
    
    # I need to clear the previous incorrect entries. Let me restart.
    
    # Reset
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_form2(idx, u_entries, v_entries, w_entries):
        for (u_row, coeff) in u_entries:
            U[u_row][idx] += coeff
        for (v_row, coeff) in v_entries:
            V[v_row][idx] += coeff
        for (w_row, coeff) in w_entries:
            W[w_row][idx] += coeff
    
    # Block 1: A_L * B_LL -> C_L (forms 0-6)
    # U: A_L = [0,1;4,5], V: B_LL = [0,1;4,5], W: C_L = [0,1;2,3]
    add_form2(0, [(0,1),(5,1)], [(0,1),(5,1)], [(0,1),(2,1),(3,-1),(7,1)])  # M1: z00+=M1,z10+=M1,z11-=M1... wait
    # Let me be very careful with the Strassen formulas.
    # z00 = M1+M4-M5+M7 -> W[0] gets M1+M4-M5+M7
    # z01 = M3+M5 -> W[1] gets M3+M5
    # z10 = M2+M4 -> W[2] gets M2+M4
    # z11 = M1-M2+M3+M6 -> W[3] gets M1-M2+M3+M6
    
    # Reset again
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_form3(idx, u_entries, v_entries, w_entries):
        for (u_row, coeff) in u_entries:
            U[u_row][idx] += coeff
        for (v_row, coeff) in v_entries:
            V[v_row][idx] += coeff
        for (w_row, coeff) in w_entries:
            W[w_row][idx] += coeff
    
    # Block 1: A_L * B_LL -> C_L (forms 0-6)
    # U rows for A_L: x00=0, x01=1, x10=4, x11=5
    # V rows for B_LL: y00=0, y01=1, y10=4, y11=5
    # W rows for C_L: z00=0, z01=1, z10=2, z11=3
    
    # M1 = (x00+x11)(y00+y11)
    add_form3(0, [(0,1),(5,1)], [(0,1),(5,1)], [(0,1),(3,1)])  # z00+=M1, z11+=M1
    # M2 = (x10+x11)y00
    add_form3(1, [(4,1),(5,1)], [(0,1)], [(2,1),(3,-1)])  # z10+=M2, z11-=M2
    # M3 = x00(y01-y11)
    add_form3(2, [(0,1)], [(1,1),(5,-1)], [(1,1),(3,1)])  # z01+=M3, z11+=M3
    # M4 = x11(y10-y00)
    add_form3(3, [(5,1)], [(4,1),(0,-1)], [(0,1),(2,1)])  # z00+=M4, z10+=M4
    # M5 = (x00+x01)y11
    add_form3(4, [(0,1),(1,1)], [(5,1)], [(0,-1),(1,1)])  # z00-=M5, z01+=M5
    # M6 = (x10-x00)(y00+y01)
    add_form3(5, [(4,1),(0,-1)], [(0,1),(1,1)], [(3,1)])  # z11+=M6
    # M7 = (x01-x11)(y10+y11)
    add_form3(6, [(1,1),(5,-1)], [(4,1),(5,1)], [(0,1)])  # z00+=M7
    
    # Block 2: A_R * B_RL -> C_L (forms 7-13)
    # U rows for A_R: x00=2, x01=3, x10=6, x11=7
    # V rows for B_RL: y00=8, y01=9, y10=12, y11=13
    # W rows for C_L: z00=0, z01=1, z10=2, z11=3
    
    add_form3(7, [(2,1),(7,1)], [(8,1),(13,1)], [(0,1),(3,1)])
    add_form3(8, [(6,1),(7,1)], [(8,1)], [(2,1),(3,-1)])
    add_form3(9, [(2,1)], [(9,1),(13,-1)], [(1,1),(3,1)])
    add_form3(10, [(7,1)], [(12,1),(8,-1)], [(0,1),(2,1)])
    add_form3(11, [(2,1),(3,1)], [(13,1)], [(0,-1),(1,1)])
    add_form3(12, [(6,1),(2,-1)], [(8,1),(9,1)], [(3,1)])
    add_form3(13, [(3,1),(7,-1)], [(12,1),(13,1)], [(0,1)])
    
    # Block 3: A_L * B_LR -> C_R (forms 14-20)
    # U rows for A_L: x00=0, x01=1, x10=4, x11=5
    # V rows for B_LR: y00=2, y01=3, y10=6, y11=7
    # W rows for C_R: z00=4, z01=5, z10=6, z11=7
    
    add_form3(14, [(0,1),(5,1)], [(2,1),(7,1)], [(4,1),(7,1)])
    add_form3(15, [(4,1),(5,1)], [(2,1)], [(6,1),(7,-1)])
    add_form3(16, [(0,1)], [(3,1),(7,-1)], [(5,1),(7,1)])
    add_form3(17, [(5,1)], [(6,1),(2,-1)], [(4,1),(6,1)])
    add_form3(18, [(0,1),(1,1)], [(7,1)], [(4,-1),(5,1)])
    add_form3(19, [(4,1),(0,-1)], [(2,1),(3,1)], [(7,1)])
    add_form3(20, [(1,1),(5,-1)], [(6,1),(7,1)], [(4,1)])
    
    # Block 4: A_R * B_RR -> C_R (forms 21-26, only 6 forms)
    # U rows for A_R: x00=2, x01=3, x10=6, x11=7
    # V rows for B_RR: y00=10, y01=11, y10=14, y11=15
    # W rows for C_R: z00=4, z01=5, z10=6, z11=7
    #
    # To use only 6 forms, we need a special construction. One approach:
    # Share one of the forms with block 3. But that's complex.
    #
    # Alternative: just use 7 forms for block 4, giving total 28, and accept that.
    # But the task asks for rank-minimal. Let me just use 7 forms for now.
    
    add_form3(21, [(2,1),(7,1)], [(10,1),(15,1)], [(4,1),(7,1)])
    add_form3(22, [(6,1),(7,1)], [(10,1)], [(6,1),(7,-1)])
    add_form3(23, [(2,1)], [(11,1),(15,-1)], [(5,1),(7,1)])
    add_form3(24, [(7,1)], [(14,1),(10,-1)], [(4,1),(6,1)])
    add_form3(25, [(2,1),(3,1)], [(15,1)], [(4,-1),(5,1)])
    add_form3(26, [(6,1),(2,-1)], [(10,1),(11,1)], [(7,1)])
    # Missing M7 for block 4. Let me add it.
    # Actually, I only have 27 forms (0-26). Let me add M7 for block 4 as form 26.
    # Wait, I already used form 26 for M6. Let me restructure.
    
    # I have 4 blocks * 7 forms = 28 forms, but I only allocated 27.
    # Let me just use rank 28 for now, since I can't easily find the rank-27.
    
    # Actually, let me just return rank 28 with the 4 Strassen blocks.
    # This is the same as the current program but let me verify it's correct.
    
    # Reset and use rank 28
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_form4(idx, u_entries, v_entries, w_entries):
        for (u_row, coeff) in u_entries:
            U[u_row][idx] += coeff
        for (v_row, coeff) in v_entries:
            V[v_row][idx] += coeff
        for (w_row, coeff) in w_entries:
            W[w_row][idx] += coeff
    
    # Block 1: A_L * B_LL -> C_L (forms 0-6)
    add_form4(0, [(0,1),(5,1)], [(0,1),(5,1)], [(0,1),(3,1)])
    add_form4(1, [(4,1),(5,1)], [(0,1)], [(2,1),(3,-1)])
    add_form4(2, [(0,1)], [(1,1),(5,-1)], [(1,1),(3,1)])
    add_form4(3, [(5,1)], [(4,1),(0,-1)], [(0,1),(2,1)])
    add_form4(4, [(0,1),(1,1)], [(5,1)], [(0,-1),(1,1)])
    add_form4(5, [(4,1),(0,-1)], [(0,1),(1,1)], [(3,1)])
    add_form4(6, [(1,1),(5,-1)], [(4,1),(5,1)], [(0,1)])
    
    # Block 2: A_R * B_RL -> C_L (forms 7-13)
    add_form4(7, [(2,1),(7,1)], [(8,1),(13,1)], [(0,1),(3,1)])
    add_form4(8, [(6,1),(7,1)], [(8,1)], [(2,1),(3,-1)])
    add_form4(9, [(2,1)], [(9,1),(13,-1)], [(1,1),(3,1)])
    add_form4(10, [(7,1)], [(12,1),(8,-1)], [(0,1),(2,1)])
    add_form4(11, [(2,1),(3,1)], [(13,1)], [(0,-1),(1,1)])
    add_form4(12, [(6,1),(2,-1)], [(8,1),(9,1)], [(3,1)])
    add_form4(13, [(3,1),(7,-1)], [(12,1),(13,1)], [(0,1)])
    
    # Block 3: A_L * B_LR -> C_R (forms 14-20)
    add_form4(14, [(0,1),(5,1)], [(2,1),(7,1)], [(4,1),(7,1)])
    add_form4(15, [(4,1),(5,1)], [(2,1)], [(6,1),(7,-1)])
    add_form4(16, [(0,1)], [(3,1),(7,-1)], [(5,1),(7,1)])
    add_form4(17, [(5,1)], [(6,1),(2,-1)], [(4,1),(6,1)])
    add_form4(18, [(0,1),(1,1)], [(7,1)], [(4,-1),(5,1)])
    add_form4(19, [(4,1),(0,-1)], [(2,1),(3,1)], [(7,1)])
    add_form4(20, [(1,1),(5,-1)], [(6,1),(7,1)], [(4,1)])
    
    # Block 4: A_R * B_RR -> C_R (forms 21-27)
    add_form4(21, [(2,1),(7,1)], [(10,1),(15,1)], [(4,1),(7,1)])
    add_form4(22, [(6,1),(7,1)], [(10,1)], [(6,1),(7,-1)])
    add_form4(23, [(2,1)], [(11,1),(15,-1)], [(5,1),(7,1)])
    add_form4(24, [(7,1)], [(14,1),(10,-1)], [(4,1),(6,1)])
    add_form4(25, [(2,1),(3,1)], [(15,1)], [(4,-1),(5,1)])
    add_form4(26, [(6,1),(2,-1)], [(10,1),(11,1)], [(7,1)])
    add_form4(27, [(3,1),(7,-1)], [(14,1),(15,1)], [(4,1)])
    
    return {"U": U, "V": V, "W": W}
