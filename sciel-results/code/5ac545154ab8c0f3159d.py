import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add coefficients to U, V, W
    # A_index: 0..15 (row-major)
    # B_index: 0..15
    # C_index: 0..15 (row-major? No, task says W[k*n+i] which is col-major for C if C is n x p?
    # Task: W[k*n+i, r]. k is col (0..p-1), i is row (0..n-1).
    # So C_index = k*n + i.
    
    def add_U(A_idx, val):
        U[A_idx][col] += val
        
    def add_V(B_idx, val):
        V[B_idx][col] += val
        
    def add_W(C_idx, val):
        W[C_idx][col] += val

    # Define block indices for A and B
    # A blocks: A11 (0,0), A12 (0,1), A21 (1,0), A22 (1,1) in block coords
    # Scalar index for A: br*2 + dr, bc*2 + dc -> (br*2+dr)*4 + (bc*2+dc)
    
    # Linear combinations for P_i (coeffs for A)
    # P_i is 2x2. P_i[br][bc] is a linear comb of A scalars.
    # We represent P_i as a dict of (br, bc) -> list of 16 coeffs?
    # Or just compute the scalar coeffs for the 4 entries of P_i directly.
    
    # Let's define the 7 top-level products
    # i: 0..6 corresponding to T_1..T_7
    # For each i, we need:
    # 1. The 4 linear forms for P_i (each is a vector of 16 coeffs for A)
    # 2. The 4 linear forms for Q_i (each is a vector of 16 coeffs for B)
    # 3. The contribution to C for each of the 7 base Strassen products of P_i*Q_i
    
    # Base Strassen formulas for 2x2:
    # Inputs P = [[p11, p12], [p21, p22]], Q = [[q11, q12], [q21, q22]]
    # m1 = (p11+p22)(q11+q22)
    # m2 = (p21+p22)q11
    # m3 = p11(q12-q22)
    # m4 = p22(q21-q11)
    # m5 = (p11+p12)q22
    # m6 = (p21-p11)(q11+q12)
    # m7 = (p12-p22)(q21+q22)
    
    # Result T = [[t11, t12], [t21, t22]]
    # t11 = m1 + m4 - m5 + m7
    # t12 = m3 + m5
    # t21 = m2 + m4
    # t22 = m1 - m2 + m3 + m6
    
    # Coeffs for T entries in terms of m1..m7
    S_base = {
        (0,0): [1, 0, 0, 1, -1, 0, 1], # t11
        (0,1): [0, 0, 1, 0, 1, 0, 0],  # t12
        (1,0): [0, 1, 0, 1, 0, 0, 0],  # t21
        (1,1): [1, -1, 1, 0, 0, 1, 0]  # t22
    }
    
    # Top-level assembly
    # C11 = T1 + T4 - T5 + T7
    # C12 = T3 + T5
    # C21 = T2 + T4
    # C22 = T1 - T2 + T3 + T6
    
    # Map C block (br, bc) to global C indices
    # C is 4x4. Block (0,0) is rows 0-1, cols 0-1.
    # C index for global (r, c) is c*n + r (per task W[k*n+i])
    
    # Define the linear forms for P_i and Q_i
    # We need to express P_i[br][bc] as sum of A scalars.
    # A scalar A[r][c] is at index r*4+c.
    
    def get_A_coeffs(br, bc, sign=1):
        # Returns a list of 16 coeffs for A, where A[br*2+dr, bc*2+dc] has coeff sign
        coeffs = [0]*16
        for dr in range(2):
            for dc in range(2):
                r = br*2 + dr
                c = bc*2 + dc
                coeffs[r*4+c] = sign
        return coeffs

    def get_B_coeffs(br, bc, sign=1):
        coeffs = [0]*16
        for dr in range(2):
            for dc in range(2):
                r = br*2 + dr
                c = bc*2 + dc
                coeffs[r*4+c] = sign
        return coeffs

    def add_vec_to_coeffs(base, vec, sign=1):
        for i in range(16):
            base[i] += sign * vec[i]

    # Define the 7 products
    # i=0: T1 = (A11+A22)(B11+B22)
    # i=1: T2 = (A21+A22)B11
    # i=2: T3 = A11(B12-B22)
    # i=3: T4 = A22(B21-B11)
    # i=4: T5 = (A11+A12)B22
    # i=5: T6 = (A21-A11)(B11+B12)
    # i=6: T7 = (A12-A22)(B21+B22)
    
    # P_i is 2x2 matrix of linear forms.
    # Q_i is 2x2 matrix of linear forms.
    
    products = []
    
    # T1
    P1 = [[get_A_coeffs(0,0,1), get_A_coeffs(0,0,0)], # This is wrong. P1 is the matrix (A11+A22).
           [get_A_coeffs(1,1,1), get_A_coeffs(1,1,0)]]
    # Wait, P1 is a 2x2 matrix.
    # P1 = A11 + A22.
    # A11 is top-left block. A22 is bottom-right block.
    # So P1[0][0] = A11[0][0] + A22[0][0]
    # P1[0][1] = A11[0][1] + A22[0][1]
    # etc.
    
    def make_P_from_blocks(blocks):
        # blocks is a list of (br, bc, sign)
        # P is 2x2.
        P = [[[0]*16 for _ in range(2)] for _ in range(2)]
        for br, bc, sign in blocks:
            vec = get_A_coeffs(br, bc, sign)
            for dr in range(2):
                for dc in range(2):
                    # P[dr][dc] gets contribution from A[br*2+dr, bc*2+dc]
                    # Wait, P is a 2x2 matrix of linear forms.
                    # The linear form at P[dr][dc] is the sum of coeffs.
                    # The scalar A[br*2+dr, bc*2+dc] contributes to P[dr][dc].
                    add_vec_to_coeffs(P[dr][dc], [0]*16, 0) # placeholder
                    # Actually, get_A_coeffs returns coeffs for the whole 4x4 A.
                    # We just need to add the vector to P[dr][dc].
                    for k in range(16):
                        P[dr][dc][k] += vec[k]
        return P

    def make_Q_from_blocks(blocks):
        Q = [[[0]*16 for _ in range(2)] for _ in range(2)]
        for br, bc, sign in blocks:
            vec = get_B_coeffs(br, bc, sign)
            for dr in range(2):
                for dc in range(2):
                    for k in range(16):
                        Q[dr][dc][k] += vec[k]
        return Q

    # T1: P = A11+A22, Q = B11+B22
    P = make_P_from_blocks([(0,0,1), (1,1,1)])
    Q = make_Q_from_blocks([(0,0,1), (1,1,1)])
    products.append((P, Q, [(0,0,1), (0,1,0), (1,0,0), (1,1,0)])) # C contributions: C11 += T1
    
    # T2: P = A21+A22, Q = B11
    P = make_P_from_blocks([(1,0,1), (1,1,1)])
    Q = make_Q_from_blocks([(0,0,1)])
    products.append((P, Q, [(1,0,1), (1,1,0)])) # C21 += T2
    
    # T3: P = A11, Q = B12-B22
    P = make_P_from_blocks([(0,0,1)])
    Q = make_Q_from_blocks([(0,1,1), (1,1,-1)])
    products.append((P, Q, [(0,1,1), (1,1,0)])) # C12 += T3, C22 += T3
    
    # T4: P = A22, Q = B21-B11
    P = make_P_from_blocks([(1,1,1)])
    Q = make_Q_from_blocks([(1,0,1), (0,0,-1)])
    products.append((P, Q, [(0,0,1), (1,0,1)])) # C11 += T4, C21 += T4
    
    # T5: P = A11+A12, Q = B22
    P = make_P_from_blocks([(0,0,1), (0,1,1)])
    Q = make_Q_from_blocks([(1,1,1)])
    products.append((P, Q, [(0,0,-1), (0,1,1)])) # C11 -= T5, C12 += T5
    
    # T6: P = A21-A11, Q = B11+B12
    P = make_P_from_blocks([(1,0,1), (0,0,-1)])
    Q = make_Q_from_blocks([(0,0,1), (0,1,1)])
    products.append((P, Q, [(1,1,1)])) # C22 += T6
    
    # T7: P = A12-A22, Q = B21+B22
    P = make_P_from_blocks([(0,1,1), (1,1,-1)])
    Q = make_Q_from_blocks([(1,0,1), (1,1,1)])
    products.append((P, Q, [(0,0,1)])) # C11 += T7
    
    # Now iterate through products and fill U, V, W
    for i, (P, Q, c_contribs) in enumerate(products):
        # P and Q are 2x2 matrices of 16-dim vectors
        # We need to form the 7 base Strassen linear forms for P and Q
        
        # p11 = P[0][0], p12 = P[0][1], p21 = P[1][0], p22 = P[1][1]
        p11, p12, p21, p22 = P[0][0], P[0][1], P[1][0], P[1][1]
        q11, q12, q21, q22 = Q[0][0], Q[0][1], Q[1][0], Q[1][1]
        
        # Base Strassen forms
        # m1 = (p11+p22)(q11+q22)
        # m2 = (p21+p22)q11
        # m3 = p11(q12-q22)
        # m4 = p22(q21-q11)
        # m5 = (p11+p12)q22
        # m6 = (p21-p11)(q11+q12)
        # m7 = (p12-p22)(q21+q22)
        
        # Helper to add vector to U/V
        def add_U_vec(vec, col):
            for idx in range(16):
                if vec[idx] != 0:
                    U[idx][col] += vec[idx]
                    
        def add_V_vec(vec, col):
            for idx in range(16):
                if vec[idx] != 0:
                    V[idx][col] += vec[idx]
        
        # m1
        col = i*7 + 0
        u_vec = [p11[j] + p22[j] for j in range(16)]
        v_vec = [q11[j] + q22[j] for j in range(16)]
        add_U_vec(u_vec, col)
        add_V_vec(v_vec, col)
        
        # m2
        col = i*7 + 1
        u_vec = [p21[j] + p22[j] for j in range(16)]
        v_vec = q11[:]
        add_U_vec(u_vec, col)
        add_V_vec(v_vec, col)
        
        # m3
        col = i*7 + 2
        u_vec = p11[:]
        v_vec = [q12[j] - q22[j] for j in range(16)]
        add_U_vec(u_vec, col)
        add_V_vec(v_vec, col)
        
        # m4
        col = i*7 + 3
        u_vec = p22[:]
        v_vec = [q21[j] - q11[j] for j in range(16)]
        add_U_vec(u_vec, col)
        add_V_vec(v_vec, col)
        
        # m5
        col = i*7 + 4
        u_vec = [p11[j] + p12[j] for j in range(16)]
        v_vec = q22[:]
        add_U_vec(u_vec, col)
        add_V_vec(v_vec, col)
        
        # m6
        col = i*7 + 5
        u_vec = [p21[j] - p11[j] for j in range(16)]
        v_vec = [q11[j] + q12[j] for j in range(16)]
        add_U_vec(u_vec, col)
        add_V_vec(v_vec, col)
        
        # m7
        col = i*7 + 6
        u_vec = [p12[j] - p22[j] for j in range(16)]
        v_vec = [q21[j] + q22[j] for j in range(16)]
        add_U_vec(u_vec, col)
        add_V_vec(v_vec, col)
        
        # Now update W for this product i
        # c_contribs is a list of (c_br, c_bc, sign)
        # For each contribution, we add S_base[local_r, local_c][v] * sign to W[C_idx, col]
        
        for c_br, c_bc, sign in c_contribs:
            for lr in range(2):
                for lc in range(2):
                    # Global C row = c_br*2 + lr, col = c_bc*2 + lc
                    r = c_br*2 + lr
                    c = c_bc*2 + lc
                    # W index: c*n + r
                    w_idx = c*n + r
                    
                    # S_base[lr, lc] gives coeffs for m1..m7
                    s_vec = S_base[(lr, lc)]
                    for v in range(7):
                        if s_vec[v] != 0:
                            col = i*7 + v
                            W[w_idx][col] += sign * s_vec[v]
                            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
