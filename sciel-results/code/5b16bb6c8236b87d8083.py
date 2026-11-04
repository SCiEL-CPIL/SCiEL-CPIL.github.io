import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Global dimensions
    dim_u = n * m  # 16
    dim_v = m * p  # 28
    dim_w = p * n  # 28
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u, v, w):
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)

    # Schoolbook 2x2x2 (Rank 8)
    # U (4x8), V (4x8), W (4x8)
    def get_schoolbook_2x2():
        U = np.zeros((4, 8))
        V = np.zeros((4, 8))
        W = np.zeros((4, 8))
        t = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U[i*2+j, t] = 1
                    V[j*2+k, t] = 1
                    W[k*2+i, t] = 1
                    t += 1
        return U, V, W

    # Schoolbook 2x2x1 (Rank 4)
    # U (4x4), V (2x4), W (2x4)
    def get_schoolbook_2x2x1():
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        t = 0
        for i in range(2):
            for j in range(2):
                U[i*2+j, t] = 1
                V[j, t] = 1
                W[i, t] = 1
                t += 1
        return U, V, W

    # Part 1: 4x4x4 (Rank 56 with Schoolbook 2x2x2)
    # 7 Strassen steps.
    
    # Strassen Coefficients
    # A blocks: 00, 01, 10, 11
    # B blocks: 00, 01, 10, 11
    
    # Step 0: M1 = (A00+A11)(B00+B11)
    # Step 1: M2 = (A10+A11)B00
    # Step 2: M3 = A00(B01-B11)
    # Step 3: M4 = A11(B10-B00)
    # Step 4: M5 = (A00+A01)B11
    # Step 5: M6 = (A10-A00)(B00+B01)
    # Step 6: M7 = (A01-A11)(B10+B11)
    
    # C contributions:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    steps = [
        ({'00': 1, '11': 1}, {'00': 1, '11': 1}), # M1
        ({'10': 1, '11': 1}, {'00': 1}),           # M2
        ({'00': 1}, {'01': 1, '11': -1}),          # M3
        ({'11': 1}, {'10': 1, '00': -1}),          # M4
        ({'00': 1, '01': 1}, {'11': 1}),           # M5
        ({'10': 1, '00': -1}, {'00': 1, '01': 1}), # M6
        ({'01': 1, '11': -1}, {'10': 1, '11': 1})  # M7
    ]
    
    c_contribs = [
        {'00': 1, '11': 1},      # M1 -> C00, C11
        {'10': 1, '11': -1},     # M2 -> C10, C11
        {'01': 1, '11': 1},      # M3 -> C01, C11
        {'00': 1, '10': 1},      # M4 -> C00, C10
        {'01': 1, '00': -1},     # M5 -> C01, C00
        {'11': 1},               # M6 -> C11
        {'00': 1, '11': -1}      # M7 -> C00, C11
    ]

    U_s22, V_s22, W_s22 = get_schoolbook_2x2() # Shapes 4x8
    
    for s in range(7):
        alpha = steps[s][0]
        beta = steps[s][1]
        gamma = c_contribs[s]
        
        for t in range(8): # 8 terms in schoolbook 2x2x2
            u_local = U_s22[:, t]
            v_local = V_s22[:, t]
            w_local = W_s22[:, t]
            
            u_global = np.zeros(16)
            v_global = np.zeros(28)
            w_global = np.zeros(28)
            
            # Map A (4x4)
            # u_local index: i*2+j where i is row, j is col of 2x2 block
            for block, coeff in alpha.items():
                if coeff == 0: continue
                r, c = int(block[0]), int(block[1])
                for i in range(2):
                    for j in range(2):
                        u_global[(r*2+i)*4 + (c*2+j)] = coeff * u_local[i*2+j]
                        
            # Map B (4x4)
            # v_local index: j*2+k where j is row, k is col of 2x2 block
            for block, coeff in beta.items():
                if coeff == 0: continue
                r, c = int(block[0]), int(block[1])
                for i in range(2):
                    for j in range(2):
                        v_global[(r*2+i)*7 + (c*2+j)] = coeff * v_local[i*2+j]
                        
            # Map C (4x4)
            # w_local index: k*2+i where k is col, i is row of 2x2 block
            # Global C index: (col)*4 + row
            for block, coeff in gamma.items():
                if coeff == 0: continue
                r, c = int(block[0]), int(block[1])
                for i in range(2):
                    for k in range(2):
                        w_global[(c*2+k)*4 + (r*2+i)] = coeff * w_local[k*2+i]
                        
            add_term(u_global, v_global, w_global)

    # Part 2: 4x4x2 (Cols 4-5)
    # 4 terms, each 2x2x2.
    # A blocks (4x4) multiplied by B blocks (4x2)
    # B is 4x2. Cols 4, 5.
    # We can view B as two 2x1 vectors? No, B is 4x2.
    # Let's stick to 2x2x2 blocks.
    # A is 4x4. B is 4x2.
    # We can split A into 2x2 blocks.
    # B into 2x1 blocks? 
    # If we use 2x2x2, we need B to be 2x2.
    # B has 4 rows, 2 cols.
    # Top 2 rows of B (rows 0-1) form a 2x2 block (cols 4-5).
    # Bottom 2 rows of B (rows 2-3) form a 2x2 block (cols 4-5).
    
    # Terms:
    # 1. A00 (2x2) * B_top (2x2) -> C_top (2x2)
    # 2. A01 (2x2) * B_bot (2x2) -> C_top (2x2)
    # 3. A10 (2x2) * B_top (2x2) -> C_bot (2x2)
    # 4. A11 (2x2) * B_bot (2x2) -> C_bot (2x2)
    
    terms_p2 = [
        ({'00': 1}, {'00': 1}, {'00': 1}), # A00 Btop -> Ctop
        ({'01': 1}, {'11': 1}, {'00': 1}), # A01 Bbot -> Ctop
        ({'10': 1}, {'00': 1}, {'10': 1}), # A10 Btop -> Cbot
        ({'11': 1}, {'11': 1}, {'10': 1})  # A11 Bbot -> Cbot
    ]
    
    for alpha, beta, gamma in terms_p2:
        for t in range(8):
            u_local = U_s22[:, t]
            v_local = V_s22[:, t]
            w_local = W_s22[:, t]
            
            u_global = np.zeros(16)
            v_global = np.zeros(28)
            w_global = np.zeros(28)
            
            for block, coeff in alpha.items():
                r, c = int(block[0]), int(block[1])
                for i in range(2):
                    for j in range(2):
                        u_global[(r*2+i)*4 + (c*2+j)] = coeff * u_local[i*2+j]
                        
            # Map B (4x2)
            # Local B block is 2x2.
            # Global B block for '00' is rows 0-1, cols 4-5.
            # Global B block for '11' is rows 2-3, cols 4-5.
            for block, coeff in beta.items():
                r, c = int(block[0]), int(block[1])
                # c is always 0 in local 4x2 representation of block?
                # Wait, B is 4x2.
                # Block '00' corresponds to rows 0-1. Cols 0-1 of the 4x2 matrix?
                # Yes, the 4x2 matrix has cols 0,1 (which are global 4,5).
                # So local col 0 -> global 4, local col 1 -> global 5.
                for i in range(2):
                    for j in range(2):
                        v_global[(r*2+i)*7 + (4+j)] = coeff * v_local[i*2+j]
                        
            # Map C (4x2)
            # Local C block is 2x2.
            # Global C block for '00' is rows 0-1, cols 4-5.
            # Global C block for '10' is rows 2-3, cols 4-5.
            # C is 7x4. W[k*n+i, r] -> W[col*4+row, r].
            # C cols 4,5.
            for block, coeff in gamma.items():
                r, c = int(block[0]), int(block[1])
                # r is block row (0 or 1). c is block col (0 or 1).
                # Global C row: r*2 + i
                # Global C col: 4 + c + k
                for i in range(2):
                    for k in range(2):
                        w_global[(4+c+k)*4 + (r*2+i)] = coeff * w_local[k*2+i]
                        
            add_term(u_global, v_global, w_global)

    # Part 3: 4x4x1 (Col 6)
    # 4 terms, each 2x2x1.
    # A (4x4) * B (4x1) -> C (4x1)
    # B is 4x1. Col 6.
    # Split B into top 2x1 and bot 2x1.
    # Terms:
    # 1. A00 (2x2) * B_top (2x1) -> C_top (2x1)
    # 2. A01 (2x2) * B_bot (2x1) -> C_top (2x1)
    # 3. A10 (2x2) * B_top (2x1) -> C_bot (2x1)
    # 4. A11 (2x2) * B_bot (2x1) -> C_bot (2x1)
    
    U_s221, V_s221, W_s221 = get_schoolbook_2x2x1() # 4x4, 2x4, 2x4
    
    terms_p3 = [
        ({'00': 1}, {'00': 1}, {'00': 1}), # A00 Btop -> Ctop
        ({'01': 1}, {'11': 1}, {'00': 1}), # A01 Bbot -> Ctop
        ({'10': 1}, {'00': 1}, {'10': 1}), # A10 Btop -> Cbot
        ({'11': 1}, {'11': 1}, {'10': 1})  # A11 Bbot -> Cbot
    ]
    
    for alpha, beta, gamma in terms_p3:
        for t in range(4):
            u_local = U_s221[:, t]
            v_local = V_s221[:, t]
            w_local = W_s221[:, t]
            
            u_global = np.zeros(16)
            v_global = np.zeros(28)
            w_global = np.zeros(28)
            
            for block, coeff in alpha.items():
                r, c = int(block[0]), int(block[1])
                for i in range(2):
                    for j in range(2):
                        u_global[(r*2+i)*4 + (c*2+j)] = coeff * u_local[i*2+j]
                        
            # Map B (4x1)
            # Local B is 2x1.
            # Global B is 4x1. Col 6.
            # Block '00' is rows 0-1. Block '11' is rows 2-3.
            for block, coeff in beta.items():
                r, c = int(block[0]), int(block[1])
                for i in range(2):
                    v_global[(r*2+i)*7 + 6] = coeff * v_local[i]
                    
            # Map C (4x1)
            # Local C is 2x1.
            # Global C is 4x1. Col 6.
            # Block '00' is rows 0-1. Block '10' is rows 2-3.
            for block, coeff in gamma.items():
                r, c = int(block[0]), int(block[1])
                for i in range(2):
                    w_global[6*4 + (r*2+i)] = coeff * w_local[i]
                    
            add_term(u_global, v_global, w_global)

    rank = len(U_list)
    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
