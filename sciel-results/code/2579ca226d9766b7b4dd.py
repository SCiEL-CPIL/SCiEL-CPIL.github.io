import time
import sys

# EVOLVE-BLOCK-START

def construct():
    n, m, p = 4, 4, 7
    
    # Strategy:
    # 1. Split B (4x7) into B1 (4x4) and B2 (4x3).
    # 2. Compute C1 = A * B1 using Strassen's algorithm for 4x4x4 (rank 49).
    # 3. Compute C2 = A * B2 using the trivial method (rank 4*4*3 = 48).
    # 4. Total rank = 49 + 48 = 97.
    
    rank = 97
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    t = 0
    
    # --- Part 1: 4x4x4 Strassen (Rank 49) ---
    # The 4x4 Strassen algorithm splits into 2x2 blocks and uses 7 recursive 2x2 multiplications.
    # Each 2x2 multiplication uses Strassen's 7-term algorithm.
    # Total: 7 * 7 = 49 rank-1 terms.
    
    # Strassen 2x2x2 UVW coefficients
    # For 2x2 matrices, indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    S2_U = [
        [1, 0, 1, 0, 1, -1, 0],  # A00
        [0, 1, 0, 0, 1, 0, 1],   # A01
        [0, 1, 0, 0, 0, 1, 0],   # A10
        [1, 1, 0, 1, 0, 0, -1]   # A11
    ]
    S2_V = [
        [1, 1, 0, 0, 0, 1, 1],   # B00
        [0, 0, 1, 0, 0, 1, 0],   # B01
        [0, 0, 0, 1, 0, 0, 1],   # B10
        [1, 0, -1, 0, 1, 0, 1]   # B11
    ]
    S2_W = [
        [1, 0, 0, 1, 0, 0, 1],   # C00
        [0, 0, 1, 0, 1, 0, -1],  # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, 0, 0, 0, -1, 1, -1]  # C11
    ]
    
    # Outer Strassen steps for 4x4x4
    # Each step defines linear combinations of 2x2 blocks of A and B
    outer_steps = [
        # M1 = (A00 + A11)(B00 + B11)
        {'a': {(0,0): 1, (1,1): 1}, 'b': {(0,0): 1, (1,1): 1}},
        # M2 = (A10 + A11)(B00)
        {'a': {(1,0): 1, (1,1): 1}, 'b': {(0,0): 1}},
        # M3 = (A00)(B01 + B11)
        {'a': {(0,0): 1}, 'b': {(0,1): 1, (1,1): -1}},
        # M4 = (A11)(B10 + B00)
        {'a': {(1,1): 1}, 'b': {(1,0): 1, (0,0): -1}},
        # M5 = (A00 + A01)(B11)
        {'a': {(0,0): 1, (0,1): 1}, 'b': {(1,1): 1}},
        # M6 = (A10 - A00)(B00 + B01)
        {'a': {(1,0): 1, (0,0): -1}, 'b': {(0,0): 1, (0,1): 1}},
        # M7 = (A01 - A11)(B10 + B11)
        {'a': {(0,1): 1, (1,1): -1}, 'b': {(1,0): 1, (1,1): 1}}
    ]
    
    # For each outer step and inner Strassen term, create a rank-1 tensor
    for step_idx in range(7):
        a_blocks = outer_steps[step_idx]['a']
        b_blocks = outer_steps[step_idx]['b']
        
        for inner_idx in range(7):
            u_vec = [0] * (n * m)
            v_vec = [0] * (m * p)
            w_vec = [0] * (p * n)
            
            # Fill U vector (coefficients for A elements)
            for local_a in range(4):
                coeff_u = S2_U[local_a][inner_idx]
                if coeff_u == 0:
                    continue
                lr = local_a // 2
                lc = local_a % 2
                for (br, bc), coeff_a in a_blocks.items():
                    gi = br * 2 + lr
                    gj = bc * 2 + lc
                    u_vec[gi * m + gj] += coeff_a * coeff_u
            
            # Fill V vector (coefficients for B elements)
            for local_b in range(4):
                coeff_v = S2_V[local_b][inner_idx]
                if coeff_v == 0:
                    continue
                lr = local_b // 2
                lc = local_b % 2
                for (br, bc), coeff_b in b_blocks.items():
                    gi = br * 2 + lr
                    gj = bc * 2 + lc
                    v_vec[gi * p + gj] += coeff_b * coeff_v
            
            # Fill W vector (coefficients for C elements)
            # The result of each outer step contributes to specific C blocks
            # C00 = M1 + M4 - M5 - M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # For each local C element in the 2x2 result of this outer step,
            # we need to determine which global C block it contributes to and with what weight.
            # The inner Strassen W gives the contribution to the 2x2 matrix M_step.
            # Then M_step is added to the appropriate C blocks with the outer weights.
            
            # Outer weights for each C block from each M step:
            # M1: C00(+1), C11(+1)
            # M2: C10(+1), C11(-1)
            # M3: C01(+1), C11(+1)
            # M4: C00(+1), C10(+1)
            # M5: C00(-1), C01(+1)
            # M6: C11(+1)
            # M7: C00(-1)
            
            outer_w_weights = {
                0: {(0,0): 1, (1,1): 1},     # M1
                1: {(1,0): 1, (1,1): -1},    # M2
                2: {(0,1): 1, (1,1): 1},     # M3
                3: {(0,0): 1, (1,0): 1},     # M4
                4: {(0,0): -1, (0,1): 1},    # M5
                5: {(1,1): 1},               # M6
                6: {(0,0): -1}               # M7
            }
            
            w_weights = outer_w_weights[step_idx]
            
            for local_c in range(4):
                coeff_w_inner = S2_W[local_c][inner_idx]
                if coeff_w_inner == 0:
                    continue
                lr = local_c // 2
                lc = local_c % 2
                for (br, bc), coeff_w_outer in w_weights.items():
                    gi = br * 2 + lr
                    gj = bc * 2 + lc
                    w_vec[gj * n + gi] += coeff_w_outer * coeff_w_inner
            
            # Add this rank-1 term
            for i in range(n * m):
                if u_vec[i] != 0:
                    U[i][t] = u_vec[i]
            for j in range(m * p):
                if v_vec[j] != 0:
                    V[j][t] = v_vec[j]
            for k in range(p * n):
                if w_vec[k] != 0:
                    W[k][t] = w_vec[k]
            t += 1
    
    # --- Part 2: 4x4x3 Trivial (Rank 48) ---
    for i in range(n):
        for j in range(m):
            for k in range(4, p):  # columns 4,5,6 of B
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][t] = 1
                V[v_idx][t] = 1
                W[w_idx][t] = 1
                t += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

