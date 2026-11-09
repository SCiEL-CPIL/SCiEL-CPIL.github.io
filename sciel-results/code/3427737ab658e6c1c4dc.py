import numpy as np
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 decomposition using recursive Strassen
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to create a linear form (list of 16 floats)
    def zero_lf():
        return [0.0] * 16

    def add_lf(a, b, sign=1):
        return [x + sign * y for x, y in zip(a, b)]

    def sub_lf(a, b):
        return [x - y for x, y in zip(a, b)]

    # Define LFs for A entries
    # A is 4x4. Blocks are 2x2.
    # Block (br, bc) has local indices (lr, lc)
    # Flat index = br * 8 + bc * 2 + lc + lr * 2
    # Wait, let's verify the index mapping for 4x4 matrix flattened row-major.
    # Row i, Col j -> i * 4 + j
    # Block (br, bc) covers rows 2*br, 2*br+1 and cols 2*bc, 2*bc+1
    # Local (lr, lc) corresponds to global row 2*br + lr, col 2*bc + lc
    # Flat index = (2*br + lr) * 4 + (2*bc + lc)
    # = 8*br + 4*lr + 2*bc + lc
    # = 8*br + 2*bc + 4*lr + lc
    
    # Let's check block (0,0):
    # (0,0) -> 0
    # (0,1) -> 1
    # (1,0) -> 4
    # (1,1) -> 5
    # Matches previous analysis.
    
    def get_a_lf(br, bc, lr, lc):
        lf = zero_lf()
        idx = 8 * br + 2 * bc + 4 * lr + lc
        lf[idx] = 1.0
        return lf

    def get_b_lf(br, bc, lr, lc):
        lf = zero_lf()
        idx = 8 * br + 2 * bc + 4 * lr + lc
        lf[idx] = 1.0
        return lf

    # Pre-compute LFs for all A and B entries
    A_entries = {}
    B_entries = {}
    for br in range(2):
        for bc in range(2):
            for lr in range(2):
                for lc in range(2):
                    A_entries[(br, bc, lr, lc)] = get_a_lf(br, bc, lr, lc)
                    B_entries[(br, bc, lr, lc)] = get_b_lf(br, bc, lr, lc)

    # Helper to get a 2x2 matrix of LFs for a block combination
    # block_lf(br, bc) returns [[LF(0,0), LF(0,1)], [LF(1,0), LF(1,1)]]
    def get_block_lf(entries, br, bc):
        return [
            [entries[(br, bc, 0, 0)], entries[(br, bc, 0, 1)]],
            [entries[(br, bc, 1, 0)], entries[(br, bc, 1, 1)]]
        ]

    # Helper to add two 2x2 matrices of LFs
    def add_block(X, Y, sign=1):
        return [
            [add_lf(X[i][j], Y[i][j], sign) for j in range(2)]
            for i in range(2)
        ]

    # Helper to subtract two 2x2 matrices of LFs
    def sub_block(X, Y):
        return [
            [sub_lf(X[i][j], Y[i][j]) for j in range(2)]
            for i in range(2)
        ]

    # Define Outer Strassen Steps
    # Each step k defines X_k, Y_k (2x2 matrices of LFs) and D_k (coeffs for C blocks)
    # C blocks are indexed by (br, bc)
    
    A00 = get_block_lf(A_entries, 0, 0)
    A01 = get_block_lf(A_entries, 0, 1)
    A10 = get_block_lf(A_entries, 1, 0)
    A11 = get_block_lf(A_entries, 1, 1)
    
    B00 = get_block_lf(B_entries, 0, 0)
    B01 = get_block_lf(B_entries, 0, 1)
    B10 = get_block_lf(B_entries, 1, 0)
    B11 = get_block_lf(B_entries, 1, 1)

    outer_steps = [
        {
            "X": add_block(A00, A11),
            "Y": add_block(B00, B11),
            "D": {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1} # C00+=M1, C01+=M1, C10+=M1, C11+=M1
        },
        {
            "X": add_block(A10, A11),
            "Y": B00,
            "D": {(1,0): 1, (1,1): 1} # C10+=M2, C11+=M2 (Wait, C11 = M1 - M2 ... so -1)
            # Correction:
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
        },
        {
            "X": A00,
            "Y": sub_block(B01, B11),
            "D": {(0,1): 1, (1,1): 1} # C01+=M3, C11+=M3
        },
        {
            "X": A11,
            "Y": sub_block(B10, B00),
            "D": {(0,0): 1, (1,0): 1} # C00+=M4, C10+=M4
        },
        {
            "X": add_block(A00, A01),
            "Y": B11,
            "D": {(0,0): -1, (0,1): 1} # C00-=M5, C01+=M5
        },
        {
            "X": sub_block(A10, A00),
            "Y": add_block(B00, B01),
            "D": {(1,1): 1} # C11+=M6
        },
        {
            "X": sub_block(A01, A11),
            "Y": add_block(B10, B11),
            "D": {(0,0): 1} # C00+=M7
        }
    ]
    
    # Correct D mappings:
    outer_steps[0]["D"] = {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}
    outer_steps[1]["D"] = {(1,0): 1, (1,1): -1}
    outer_steps[2]["D"] = {(0,1): 1, (1,1): 1}
    outer_steps[3]["D"] = {(0,0): 1, (1,0): 1}
    outer_steps[4]["D"] = {(0,0): -1, (0,1): 1}
    outer_steps[5]["D"] = {(1,1): 1}
    outer_steps[6]["D"] = {(0,0): 1}

    # Inner Strassen Coefficients for 2x2
    # S1 = (X00+X11)(Y00+Y11)
    # S2 = (X10+X11)Y00
    # S3 = X00(Y01-Y11)
    # S4 = X11(Y10-Y00)
    # S5 = (X00+X01)Y11
    # S6 = (X10-X00)(Y00+Y01)
    # S7 = (X01-X11)(Y10+Y11)
    
    # Z00 = S1 + S4 - S5 + S7
    # Z01 = S3 + S5
    # Z10 = S2 + S4
    # Z11 = S1 - S2 + S3 + S6
    
    # For each s, we need U_s, V_s and the coefficients for Z00, Z01, Z10, Z11
    inner_steps = [
        { # S1
            "U_func": lambda X: add_lf(X[0][0], X[1][1]),
            "V_func": lambda Y: add_lf(Y[0][0], Y[1][1]),
            "Z_coeffs": {0: 1, 1: 0, 2: 0, 3: 1} # Z00:1, Z01:0, Z10:0, Z11:1
        },
        { # S2
            "U_func": lambda X: add_lf(X[1][0], X[1][1]),
            "V_func": lambda Y: Y[0][0],
            "Z_coeffs": {0: 0, 1: 0, 2: 1, 3: -1} # Z10:1, Z11:-1
        },
        { # S3
            "U_func": lambda X: X[0][0],
            "V_func": lambda Y: sub_lf(Y[0][1], Y[1][1]),
            "Z_coeffs": {0: 0, 1: 1, 2: 0, 3: 1} # Z01:1, Z11:1
        },
        { # S4
            "U_func": lambda X: X[1][1],
            "V_func": lambda Y: sub_lf(Y[1][0], Y[0][0]),
            "Z_coeffs": {0: 1, 1: 0, 2: 1, 3: 0} # Z00:1, Z10:1
        },
        { # S5
            "U_func": lambda X: add_lf(X[0][0], X[0][1]),
            "V_func": lambda Y: Y[1][1],
            "Z_coeffs": {0: -1, 1: 1, 2: 0, 3: 0} # Z00:-1, Z01:1
        },
        { # S6
            "U_func": lambda X: sub_lf(X[1][0], X[0][0]),
            "V_func": lambda Y: add_lf(Y[0][0], Y[0][1]),
            "Z_coeffs": {0: 0, 1: 0, 2: 0, 3: 1} # Z11:1
        },
        { # S7
            "U_func": lambda X: sub_lf(X[0][1], X[1][1]),
            "V_func": lambda Y: add_lf(Y[1][0], Y[1][1]),
            "Z_coeffs": {0: 1, 1: 0, 2: 0, 3: 0} # Z00:1
        }
    ]

    col = 0
    for k in range(7):
        X_k = outer_steps[k]["X"]
        Y_k = outer_steps[k]["Y"]
        D_k = outer_steps[k]["D"]
        
        for s in range(7):
            U_s = inner_steps[s]["U_func"](X_k)
            V_s = inner_steps[s]["V_func"](Y_k)
            Z_coeffs = inner_steps[s]["Z_coeffs"]
            
            # Construct W for this term
            W_s = zero_lf()
            # Z_coeffs keys: 0->Z00, 1->Z01, 2->Z10, 3->Z11
            # Z00 is block (0,0), Z01 is block (0,1), Z10 is block (1,0), Z11 is block (1,1)
            block_map = {0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)}
            
            for z_idx, coeff in Z_coeffs.items():
                if coeff == 0:
                    continue
                br, bc = block_map[z_idx]
                if (br, bc) in D_k:
                    outer_coeff = D_k[(br, bc)]
                    total_coeff = coeff * outer_coeff
                    if total_coeff != 0:
                        # Add total_coeff to W_s for all 4 entries of block (br, bc)
                        for lr in range(2):
                            for lc in range(2):
                                idx = 8 * br + 2 * bc + 4 * lr + lc
                                W_s[idx] += total_coeff
            
            # Add term to U, V, W
            for i in range(16):
                if U_s[i] != 0:
                    U[i][col] += U_s[i]
                if V_s[i] != 0:
                    V[i][col] += V_s[i]
                if W_s[i] != 0:
                    W[i][col] += W_s[i]
            
            col += 1

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
