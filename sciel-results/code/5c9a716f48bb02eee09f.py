"""Rewritten program using hybrid Strassen + schoolbook decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    start_time = time.time()
    deadline = start_time + 130.0

    # Precompute index arrays for verification
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    u_idx = i_idx * m + j_idx  # (112,)
    v_idx = j_idx * p + k_idx  # (112,)
    w_idx = k_idx * n + i_idx  # (112,)

    def verify_fast(Ua, Va, Wa):
        """Fast verification. Returns max error."""
        u_vals = Ua[u_idx, :]
        v_vals = Va[v_idx, :]
        w_vals = Wa[w_idx, :]
        prod = u_vals * v_vals * w_vals
        result = prod.sum(axis=1)
        return np.max(np.abs(result - 1.0))

    def strassen_2x2():
        """
        Strassen decomposition for 2x2 * 2x2 -> 2x2.
        Returns U (4,7), V (4,7), W (4,7).
        """
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = A*B
        # C = [[p1+p3-p5-p7, p5+p7], [p2+p4, p1-p2+p4+p6]]
        # p1 = (a+d)(e+h)
        # p2 = (c+d)e
        # p3 = a(f-h)
        # p4 = d(g-e)
        # p5 = (a+b)h
        # p6 = (c-a)(e+f)
        # p7 = (b-d)(g+h)
        #
        # U indices: 0=a, 1=b, 2=c, 3=d
        # V indices: 0=e, 1=f, 2=g, 3=h
        # W indices: 0=C00, 1=C01, 2=C10, 3=C11
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # p1 = (a+d)(e+h): U: a=1,d=1; V: e=1,h=1; W: C00+=1, C11+=1
        U[0,0] = 1; U[3,0] = 1
        V[0,0] = 1; V[3,0] = 1
        W[0,0] = 1; W[3,0] = 1
        
        # p2 = (c+d)e: U: c=1,d=1; V: e=1; W: C10+=1
        U[2,1] = 1; U[3,1] = 1
        V[0,1] = 1
        W[2,1] = 1
        
        # p3 = a(f-h): U: a=1; V: f=1,h=-1; W: C00+=1
        U[0,2] = 1
        V[1,2] = 1; V[3,2] = -1
        W[0,2] = 1
        
        # p4 = d(g-e): U: d=1; V: g=1,e=-1; W: C10+=1
        U[3,3] = 1
        V[2,3] = 1; V[0,3] = -1
        W[2,3] = 1
        
        # p5 = (a+b)h: U: a=1,b=1; V: h=1; W: C01+=1
        U[0,4] = 1; U[1,4] = 1
        V[3,4] = 1
        W[1,4] = 1
        
        # p6 = (c-a)(e+f): U: c=1,a=-1; V: e=1,f=1; W: C11+=1
        U[2,5] = 1; U[0,5] = -1
        V[0,5] = 1; V[1,5] = 1
        W[3,5] = 1
        
        # p7 = (b-d)(g+h): U: b=1,d=-1; V: g=1,h=1; W: C00-=1, C01+=1
        U[1,6] = 1; U[3,6] = -1
        V[2,6] = 1; V[3,6] = 1
        W[0,6] = -1; W[1,6] = 1
        
        return U, V, W

    def strassen_4x4():
        """
        Strassen decomposition for 4x4 * 4x4 -> 4x4 using recursive Strassen.
        Returns U (16,49), V (16,49), W (16,49).
        """
        # Partition 4x4 into 2x2 blocks:
        # A = [[A11, A12], [A21, A22]]
        # B = [[B11, B12], [B21, B22]]
        # C = A*B via Strassen on blocks
        #
        # Block indices in 4x4:
        # A11: (0,0),(0,1),(1,0),(1,1) -> indices 0,1,4,5
        # A12: (0,2),(0,3),(1,2),(1,3) -> indices 2,3,6,7
        # A21: (2,0),(2,1),(3,0),(3,1) -> indices 8,9,12,13
        # A22: (2,2),(2,3),(3,2),(3,3) -> indices 10,11,14,15
        #
        # Similarly for B and C.
        
        U_2, V_2, W_2 = strassen_2x2()
        
        # For each of the 7 Strassen products at the block level,
        # we need to compute products of 2x2 blocks.
        # Each block product uses 7 scalar products (from strassen_2x2).
        # Total: 7 * 7 = 49 products.
        
        U = np.zeros((16, 49))
        V = np.zeros((16, 49))
        W = np.zeros((16, 49))
        
        # Block index mapping: block (bi, bj) in 4x4 corresponds to
        # rows bi*2 to bi*2+1, cols bj*2 to bj*2+1
        # Element (r, c) in 4x4 has index r*4+c
        # Element (dr, dc) in 2x2 block has global index (bi*2+dr)*4 + (bj*2+dc)
        
        # Strassen at block level:
        # P1 = (A11+A22)(B11+B22)
        # P2 = (A21+A22)B11
        # P3 = A11(B12-B22)
        # P4 = A22(B21-B11)
        # P5 = (A11+A12)B22
        # P6 = (A21-A11)(B11+B12)
        # P7 = (A12-A22)(B21+B22)
        #
        # C11 = P1 + P3 - P5 - P7
        # C12 = P5 + P7
        # C21 = P2 + P4
        # C22 = P1 - P2 + P4 + P6
        
        # For each block-level product Pi, it's a product of two 2x2 matrices.
        # Each such product can be decomposed using strassen_2x2 (7 scalar products).
        
        # Define block index offsets:
        # Block (bi, bj): top-left element at row bi*2, col bj*2
        # Element (dr, dc) in block: global index (bi*2+dr)*4 + (bj*2+dc)
        
        def block_offset(bi, bj):
            """Return 4x4 array of global indices for 2x2 block (bi,bj)."""
            return np.array([[
                (bi*2+0)*4 + (bj*2+0),
                (bi*2+0)*4 + (bj*2+1),
                (bi*2+1)*4 + (bj*2+0),
                (bi*2+1)*4 + (bj*2+1)
            ]])
        
        # Block offsets for A, B, C (same structure)
        A_blocks = {
            (0,0): [0,1,4,5],
            (0,1): [2,3,6,7],
            (1,0): [8,9,12,13],
            (1,1): [10,11,14,15]
        }
        B_blocks = {
            (0,0): [0,1,4,5],
            (0,1): [2,3,6,7],
            (1,0): [8,9,12,13],
            (1,1): [10,11,14,15]
        }
        C_blocks = {
            (0,0): [0,1,4,5],
            (0,1): [2,3,6,7],
            (1,0): [8,9,12,13],
            (1,1): [10,11,14,15]
        }
        
        # Strassen block-level operations:
        # Each operation is: sum of coefficients * (block_i * block_j)
        # where block_i is a 2x2 block of A, block_j is a 2x2 block of B
        
        # P1 = (A11+A22) * (B11+B22)
        # P2 = (A21+A22) * B11
        # P3 = A11 * (B12-B22)
        # P4 = A22 * (B21-B11)
        # P5 = (A11+A12) * B22
        # P6 = (A21-A11) * (B11+B12)
        # P7 = (A12-A22) * (B21+B22)
        
        # For each Pi, we have a linear combination of A blocks times a linear combination of B blocks.
        # Then we decompose that 2x2 * 2x2 product using strassen_2x2.
        
        # Let's define the block-level Strassen more carefully.
        # For P1 = (A11+A22)(B11+B22):
        #   A part: A11 + A22 (coefficient 1 for each)
        #   B part: B11 + B22 (coefficient 1 for each)
        #   So we need to compute (A11+A22) * (B11+B22)
        #   This is a 2x2 * 2x2 product where the "A matrix" is A11+A22 and "B matrix" is B11+B22.
        
        # To decompose this, for each scalar product in strassen_2x2:
        #   U_entry = coeff_a * A_block_a + coeff_a2 * A_block_a2
        #   V_entry = coeff_b * B_block_b + coeff_b2 * B_block_b2
        #   W_entry = coeff_c * C_block_c
        
        # Let me build this systematically.
        
        # Block-level Strassen products:
        # (A_part, B_part, C_part) where A_part is dict {block_idx: coeff}, etc.
        
        block_strassen = [
            # P1 = (A11+A22)(B11+B22)
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            # P2 = (A21+A22)B11
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}),
            # P3 = A11(B12-B22)
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,0): 1}),
            # P4 = A22(B21-B11)
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(1,0): 1}),
            # P5 = (A11+A12)B22
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1}),
            # P6 = (A21-A11)(B11+B12)
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            # P7 = (A12-A22)(B21+B22)
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): -1, (0,1): 1}),
        ]
        
        # C block contributions:
        # C11 = P1 + P3 - P5 - P7
        # C12 = P5 + P7
        # C21 = P2 + P4
        # C22 = P1 - P2 + P4 + P6
        c_contrib = [
            {(0,0): 1, (1,1): 1},  # P1 contributes to C11 and C22
            {(1,0): 1},             # P2 contributes to C21
            {(0,0): 1},             # P3 contributes to C11
            {(1,0): 1},             # P4 contributes to C21
            {(0,1): 1},             # P5 contributes to C12
            {(1,1): 1},             # P6 contributes to C22
            {(0,0): -1, (0,1): 1},  # P7 contributes to C00(-1) and C01(+1)
        ]
        
        # Wait, I need to be more careful. Let me redo this.
        # The block-level Strassen computes:
        # C11 = P1 + P3 - P5 - P7
        # C12 = P5 + P7
        # C21 = P2 + P4
        # C22 = P1 - P2 + P4 + P6
        #
        # Where Pi are 2x2 matrices (block products).
        # So each Pi contributes to specific C blocks with specific coefficients.
        
        # Let me redefine:
        # For each i in 0..6 (the 7 Strassen products):
        #   A_coeffs[i]: dict mapping A block index to coefficient
        #   B_coeffs[i]: dict mapping B block index to coefficient
        #   C_contrib[i]: dict mapping C block index to coefficient
        
        A_coeffs = [
            {(0,0): 1, (1,1): 1},  # P1
            {(1,0): 1, (1,1): 1},  # P2
            {(0,0): 1},             # P3
            {(1,1): 1},             # P4
            {(0,0): 1, (0,1): 1},  # P5
            {(1,0): 1, (0,0): -1}, # P6
            {(0,1): 1, (1,1): -1}, # P7
        ]
        B_coeffs = [
            {(0,0): 1, (1,1): 1},  # P1
            {(0,0): 1},             # P2
            {(0,1): 1, (1,1): -1}, # P3
            {(1,0): 1, (0,0): -1}, # P4
            {(1,1): 1},             # P5
            {(0,0): 1, (0,1): 1},  # P6
            {(1,0): 1, (1,1): 1},  # P7
        ]
        C_contrib = [
            {(0,0): 1, (1,1): 1},  # P1 -> C11 + C22
            {(1,0): 1},             # P2 -> C21
            {(0,0): 1},             # P3 -> C11
            {(1,0): 1},             # P4 -> C21
            {(0,1): 1},             # P5 -> C12
            {(1,1): 1},             # P6 -> C22
            {(0,0): -1, (0,1): 1}, # P7 -> -C11 + C12
        ]
        
        # Now, for each Strassen product i, we decompose the 2x2*2x2 product
        # using strassen_2x2. The "A matrix" in strassen_2x2 is the linear combination
        # defined by A_coeffs[i], and similarly for B.
        
        col = 0
        for i in range(7):
            # For each of the 7 scalar products in strassen_2x2:
            for s in range(7):
                # U entries: for each A block with nonzero coeff in A_coeffs[i],
                # add coeff * U_2[elem, s]
                for (bi, bj), coeff_a in A_coeffs[i].items():
                    for dr in range(2):
                        for dc in range(2):
                            global_idx = A_blocks[(bi,bj)][dr*2+dc]
                            U[global_idx, col] += coeff_a * U_2[dr*2+dc, s]
                
                # V entries: similarly for B
                for (bi, bj), coeff_b in B_coeffs[i].items():
                    for dr in range(2):
                        for dc in range(2):
                            global_idx = B_blocks[(bi,bj)][dr*2+dc]
                            V[global_idx, col] += coeff_b * V_2[dr*2+dc, s]
                
                # W entries: for each C block with nonzero coeff in C_contrib[i],
                # add coeff * W_2[elem, s]
                for (bi, bj), coeff_c in C_contrib[i].items():
                    for dr in range(2):
                        for dc in range(2):
                            global_idx = C_blocks[(bi,bj)][dr*2+dc]
                            W[global_idx, col] += coeff_c * W_2[dr*2+dc, s]
                
                col += 1
        
        return U, V, W

    def schoolbook_4x4x3():
        """
        Schoolbook decomposition for 4x4 * 4x3 -> 4x3.
        Returns U (16,48), V (24,48), W (24,48).
        """
        # This is for the 4x4 * 4x3 part.
        # But we need to integrate it into the full 4x4 * 4x7 problem.
        # Actually, let me just do the full schoolbook for 4x4*4x7 and then
        # replace the 4x4*4x4 part with Strassen.
        
        # For now, let me just return schoolbook for the full problem.
        pass

    def make_hybrid_decomposition():
        """
        Create hybrid decomposition: Strassen for 4x4x4 part, schoolbook for 4x4x3 part.
        Total rank: 49 + 48 = 97.
        """
        U_str, V_str, W_str = strassen_4x4()
        
        # Schoolbook for 4x4 * 4x3:
        # A is 4x4, B is 4x3, C is 4x3
        # Rank = 4*4*3 = 48
        U_school = np.zeros((16, 48))
        V_school = np.zeros((12, 48))  # 4*3 = 12
        W_school = np.zeros((12, 48))  # 3*4 = 12
        
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    U_school[i*4+j, col] = 1
                    V_school[j*3+k, col] = 1
                    W_school[k*4+i, col] = 1
                    col += 1
        
        # Now combine: the full problem is 4x4 * 4x7.
        # The Strassen part handles columns 0-3 of B (the 4x4x4 part).
        # The schoolbook part handles columns 3-6 of B (the 4x4x3 part).
        #
        # But wait, the Strassen 4x4x4 assumes B is 4x4, i.e., B has 4 columns.
        # In the full problem, B has 7 columns. So I need to map the Strassen
        # decomposition to the first 4 columns of B, and the schoolbook to the last 3.
        
        # For the Strassen part:
        # U_str has shape (16, 49) - same for A
        # V_str has shape (16, 49) - but this is for B being 4x4, so V_str indices are 0..15
        #   In the full problem, B is 4x7, so V indices are 0..27.
        #   The first 4 columns of B correspond to V indices 0..15 (j*7+k for j=0..3, k=0..3)
        #   Wait, V index is j*p+k = j*7+k. For the first 4 columns (k=0..3):
        #     j=0: k=0,1,2,3 -> indices 0,1,2,3
        #     j=1: k=0,1,2,3 -> indices 7,8,9,10
        #     j=2: k=0,1,2,3 -> indices 14,15,16,17
        #     j=3: k=0,1,2,3 -> indices 21,22,23,24
        #   So V_str indices 0..15 (which are j*4+k for j=0..3, k=0..3) need to be remapped
        #   to j*7+k for the full problem.
        #
        # Similarly, W_str has shape (16, 49) for C being 4x4, so W indices are 0..15.
        #   In the full problem, C is 4x7, so W indices are 0..27.
        #   C index is k*n+i = k*4+i. For the first 4 columns (k=0..3):
        #     k=0: i=0..3 -> indices 0,1,2,3
        #     k=1: i=0..3 -> indices 4,5,6,7
        #     k=2: i=0..3 -> indices 8,9,10,11
        #     k=3: i=0..3 -> indices 12,13,14,15
        #   So W_str indices 0..15 map directly to the same indices in the full problem.
        
        # Remap V_str to full B indices:
        V_str_full = np.zeros((28, 49))
        for j in range(4):
            for k in range(4):
                old_idx = j*4+k
                new_idx = j*7+k
                V_str_full[new_idx, :] = V_str[old_idx, :]
        
        # W_str maps directly:
        W_str_full = np.zeros((28, 49))
        W_str_full[:16, :] = W_str
        
        # For the schoolbook part (last 3 columns of B, k=4,5,6):
        # U_school has shape (16, 48) - same for A
        # V_school has shape (12, 48) - for B being 4x3, V indices are j*3+k for j=0..3, k=0..2
        #   In the full problem, these correspond to j*7+(k+4) for j=0..3, k=0..2
        #   j=0: k=0,1,2 -> indices 4,5,6
        #   j=1: k=0,1,2 -> indices 11,12,13
        #   j=2: k=0,1,2 -> indices 18,19,20
        #   j=3: k=0,1,2 -> indices 25,26,27
        V_school_full = np.zeros((28, 48))
        for j in range(4):
            for k in range(3):
                old_idx = j*3+k
                new_idx = j*7+(k+4)
                V_school_full[new_idx, :] = V_school[old_idx, :]
        
        # W_school has shape (12, 48) - for C being 4x3, W indices are k*4+i for k=0..2, i=0..3
        #   In the full problem, these correspond to (k+4)*4+i for k=0..2, i=0..3
        #   k=0: i=0..3 -> indices 16,17,18,19
        #   k=1: i=0..3 -> indices 20,21,22,23
        #   k=2: i=0..3 -> indices 24,25,26,27
        W_school_full = np.zeros((28, 48))
        for k in range(3):
            for i in range(4):
                old_idx = k*4+i
                new_idx = (k+4)*4+i
                W_school_full[new_idx, :] = W_school[old_idx, :]
        
        # Combine:
        U_full = np.hstack([U_str, U_school])
        V_full = np.hstack([V_str_full, V_school_full])
        W_full = np.hstack([W_str_full, W_school_full])
        
        return U_full, V_full, W_full

    # Generate the decomposition
    U, V, W = make_hybrid_decomposition()
    
    # Verify
    err = verify_fast(U, V, W)
    if err < 1e-8:
        # Round to half-integers
        U_round = np.round(U * 2) / 2
        V_round = np.round(V * 2) / 2
        W_round = np.round(W * 2) / 2
        err_round = verify_fast(U_round, V_round, W_round)
        if err_round < 1e-9:
            return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
    
    # Fallback to schoolbook
    rank = n * m * p
    U_s = [[0] * rank for _ in range(n * m)]
    V_s = [[0] * rank for _ in range(m * p)]
    W_s = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i * m + j][col] = 1
                V_s[j * p + k][col] = 1
                W_s[k * n + i][col] = 1
                col += 1
    return {"U": U_s, "V": V_s, "W": W_s}
# EVOLVE-BLOCK-END
