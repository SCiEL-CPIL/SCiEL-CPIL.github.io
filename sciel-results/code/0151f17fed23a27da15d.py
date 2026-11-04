import time
import sys

# EVOLVE-BLOCK-START

def construct():
    n, m, p = 4, 4, 7
    
    # Strategy:
    # 1. Decompose B (4x7) into B1 (4x4) and B2 (4x3).
    # 2. Compute C1 = A * B1 using Strassen's algorithm for 4x4x4 (rank 49).
    # 3. Compute C2 = A * B2 using the trivial (schoolbook) method (rank 48).
    # 4. Total rank = 49 + 48 = 97.
    
    rank = 49 + 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 2x2x2 Strassen UVW coefficients
    # Indices for 2x2: 00, 01, 10, 11 -> 0, 1, 2, 3
    strassen_2x2_U = [
        [1, 0, 1, 0, 1, -1, 0],  # A00
        [0, 1, 0, 0, 1, 0, 1],   # A01
        [0, 1, 0, 0, 0, 1, 0],   # A10
        [1, 1, 0, 1, 0, 0, -1]   # A11
    ]
    strassen_2x2_V = [
        [1, 1, 0, 0, 0, 1, 1],   # B00
        [0, 0, 1, 0, 0, 1, 0],   # B01
        [0, 0, 0, 1, 0, 0, 1],   # B10
        [1, 0, -1, 0, 1, 0, 1]   # B11
    ]
    strassen_2x2_W = [
        [1, 0, 0, 1, 0, 0, 1],   # C00
        [0, 0, 1, 0, 1, 0, -1],  # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, 0, 0, 0, -1, 1, -1]  # C11
    ]
    
    # The 4x4 Strassen algorithm computes 7 intermediate 2x2 products M1..M7
    # from linear combinations of 2x2 blocks of A and B.
    # Then the 4 blocks of C are:
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Define the 7 outer Strassen steps
    # Each step: X = sum coeff_a * A_block, Y = sum coeff_b * B_block
    # Then compute 7 inner Strassen products on X*Y (each a 2x2 product)
    # The result M is a 2x2 matrix that gets added to specific C blocks.
    
    outer_steps = [
        # step 0: M1
        {
            'a': {(0,0): 1, (1,1): 1},
            'b': {(0,0): 1, (1,1): 1},
            'c_contrib': {(0,0): 1, (1,1): 1}  # M1 adds to C00 (+1) and C11 (+1)
        },
        # step 1: M2
        {
            'a': {(1,0): 1, (1,1): 1},
            'b': {(0,0): 1},
            'c_contrib': {(1,0): 1, (1,1): -1}  # M2 adds to C10 (+1) and C11 (-1)
        },
        # step 2: M3
        {
            'a': {(0,0): 1},
            'b': {(0,1): 1, (1,1): -1},
            'c_contrib': {(0,1): 1, (1,1): 1}  # M3 adds to C01 (+1) and C11 (+1)
        },
        # step 3: M4
        {
            'a': {(1,1): 1},
            'b': {(1,0): 1, (0,0): -1},
            'c_contrib': {(0,0): 1, (1,0): 1}  # M4 adds to C00 (+1) and C10 (+1)
        },
        # step 4: M5
        {
            'a': {(0,0): 1, (0,1): 1},
            'b': {(1,1): 1},
            'c_contrib': {(0,0): -1, (0,1): 1}  # M5 adds to C00 (-1) and C01 (+1)
        },
        # step 5: M6
        {
            'a': {(1,0): 1, (0,0): -1},
            'b': {(0,0): 1, (0,1): 1},
            'c_contrib': {(1,1): 1}  # M6 adds to C11 (+1)
        },
        # step 6: M7
        {
            'a': {(0,1): 1, (1,1): -1},
            'b': {(1,0): 1, (1,1): 1},
            'c_contrib': {(0,0): -1}  # M7 adds to C00 (-1)
        }
    ]
    
    t = 0
    for step_idx, step in enumerate(outer_steps):
        a_blocks = step['a']
        b_blocks = step['b']
        c_contrib = step['c_contrib']
        
        # For each of the 7 inner Strassen terms
        for inner_idx in range(7):
            # Build the global U, V, W vectors for this rank-1 tensor
            
            # U: coefficients for A elements
            # X[lr, lc] = sum_{(br, bc) in a_blocks} a_blocks[(br,bc)] * A[br*2+lr, bc*2+lc]
            # The inner Strassen term inner_idx uses X elements with coefficients strassen_2x2_U[local_a][inner_idx]
            # So the coefficient for A[br*2+lr, bc*2+lc] is:
            #   sum over (br,bc) in a_blocks: a_blocks[(br,bc)] * strassen_2x2_U[lr*2+lc][inner_idx]
            #   BUT only if (lr,lc) is the local position in the 2x2 block
            # Actually, for each local position (lr, lc) in the 2x2 X matrix,
            # X[lr, lc] = sum_{(br, bc) in a_blocks} a_blocks[(br,bc)] * A[br*2+lr, bc*2+lc]
            # The coefficient of A[br*2+lr, bc*2+lc] in the inner term is:
            #   a_blocks[(br,bc)] * strassen_2x2_U[lr*2+lc][inner_idx]
            
            # V: coefficients for B elements
            # Y[lr, lc] = sum_{(br, bc) in b_blocks} b_blocks[(br,bc)] * B[br*2+lr, bc*2+lc]
            # Coefficient of B[br*2+lr, bc*2+lc] is:
            #   b_blocks[(br,bc)] * strassen_2x2_V[lr*2+lc][inner_idx]
            
            # W: coefficients for C elements
            # The inner Strassen result gives a 2x2 matrix M where:
            # M[lr, lc] = sum_t ... (the inner Strassen decomposition)
            # The coefficient for M[lr, lc] in the inner term is strassen_2x2_W[lr*2+lc][inner_idx]
            # M is added to C blocks according to c_contrib.
            # If M is added to C block (cr, cc) with coefficient c_coeff,
            # then M[lr, lc] contributes to C[cr*2+lr, cc*2+lc] with coefficient c_coeff * strassen_2x2_W[lr*2+lc][inner_idx]
            
            # Now build the actual vectors
            u_vec = [0] * (n * m)  # index: i*m + j
            v_vec = [0] * (m * p)  # index: j*p + k
            w_vec = [0] * (p * n)  # index: k*n + i
            
            # Fill U
            for lr in range(2):
                for lc in range(2):
                    local_a_idx = lr * 2 + lc
                    coeff_inner = strassen_2x2_U[local_a_idx][inner_idx]
                    if coeff_inner == 0:
                        continue
                    for (br, bc), coeff_a in a_blocks.items():
                        if coeff_a == 0:
                            continue
                        gi = br * 2 + lr
                        gj = bc * 2 + lc
                        global_a_idx = gi * m + gj
                        u_vec[global_a_idx] += coeff_a * coeff_inner
            
            # Fill V
            for lr in range(2):
                for lc in range(2):
                    local_b_idx = lr * 2 + lc
                    coeff_inner = strassen_2x2_V[local_b_idx][inner_idx]
                    if coeff_inner == 0:
                        continue
                    for (br, bc), coeff_b in b_blocks.items():
                        if coeff_b == 0:
                            continue
                        gi = br * 2 + lr
                        gj = bc * 2 + lc
                        # B1 is the first 4 columns of B (4x7)
                        # B index: row=gi, col=gj -> global index gi*p + gj
                        global_b_idx = gi * p + gj
                        v_vec[global_b_idx] += coeff_b * coeff_inner
            
            # Fill W
            for lr in range(2):
                for lc in range(2):
                    local_c_idx = lr * 2 + lc
                    coeff_inner = strassen_2x2_W[local_c_idx][inner_idx]
                    if coeff_inner == 0:
                        continue
                    for (cr, cc), coeff_c in c_contrib.items():
                        if coeff_c == 0:
                            continue
                        gi = cr * 2 + lr
                        gj = cc * 2 + lc
                        # C index: C[gi, gj] -> global index gj*n + gi
                        global_c_idx = gj * n + gi
                        w_vec[global_c_idx] += coeff_c * coeff_inner
            
            # Store this rank-1 tensor
            for idx in range(n * m):
                U[idx][t] = u_vec[idx]
            for idx in range(m * p):
                V[idx][t] = v_vec[idx]
            for idx in range(p * n):
                W[idx][t] = w_vec[idx]
            t += 1
    
    # Part 2: Trivial decomposition for A (4x4) * B2 (4x3) -> C2 (4x3)
    # B2 corresponds to columns 4-6 of B (4x7)
    # C2 corresponds to columns 4-6 of C (4x7)
    for i in range(n):
        for j in range(m):
            for k in range(4, p):  # k = 4,5,6
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][t] = 1
                V[v_idx][t] = 1
                W[w_idx][t] = 1
                t += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

