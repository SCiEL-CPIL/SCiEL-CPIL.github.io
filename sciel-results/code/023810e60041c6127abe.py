"""Program for matrix multiplication 4x4x5 tensor decomposition."""
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 4x4 matrix by a 4x5 matrix.
    Uses Strassen's algorithm for the 4x4x4 part (rank 49) and schoolbook for the 4x4x1 part (rank 16).
    Total rank: 65.
    """
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # We'll construct U, V, W as lists of lists.
    rank = 65
    U = [[0] * rank for _ in range(n * m)]  # 16 x 65
    V = [[0] * rank for _ in range(m * p)]  # 20 x 65
    W = [[0] * rank for _ in range(p * n)]  # 20 x 65
    
    # Helper to add a term
    def add_term(u_vec, v_vec, w_vec, col):
        for i in range(n * m):
            U[i][col] += u_vec[i]
        for i in range(m * p):
            V[i][col] += v_vec[i]
        for i in range(p * n):
            W[i][col] += w_vec[i]

    # --- Part 1: 4x4x4 using Strassen (Rank 49) ---
    
    # Define the 7 Strassen linear forms for 2x2 block multiplication.
    # P_k and Q_k are 2x2 matrices of linear forms in A and B respectively.
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # Each block is 2x2.
    
    # We represent a linear form in A (4x4) as a 4x4 matrix of coefficients.
    # A_flat index: i*4 + j.
    
    # Let's define the block coefficients.
    # A blocks:
    # A00: (0,0), (0,1), (1,0), (1,1) -> flat: 0,1,4,5
    # A01: (0,2), (0,3), (1,2), (1,3) -> flat: 2,3,6,7
    # A10: (2,0), (2,1), (3,0), (3,1) -> flat: 8,9,12,13
    # A11: (2,2), (2,3), (3,2), (3,3) -> flat: 10,11,14,15
    
    # B blocks (4x5, but we only use first 4 cols for 4x4x4 part):
    # B00: (0,0), (0,1), (1,0), (1,1) -> flat: 0,1,4,5 (B is 4x5, flat j*5+k)
    # B01: (0,2), (0,3), (1,2), (1,3) -> flat: 2,3,6,7
    # B10: (2,0), (2,1), (3,0), (3,1) -> flat: 10,11,14,15
    # B11: (2,2), (2,3), (3,2), (3,3) -> flat: 12,13,16,17
    
    # Wait, B flat index is j*p + k = j*5 + k.
    # B00 (j=0,1; k=0,1): 0,1, 5,6
    # B01 (j=0,1; k=2,3): 2,3, 7,8
    # B10 (j=2,3; k=0,1): 10,11, 15,16
    # B11 (j=2,3; k=2,3): 12,13, 17,18
    
    # C blocks (4x4, first 4 cols):
    # C00: i=0,1; k=0,1 -> W index k*4+i: 0,1, 4,5
    # C01: i=0,1; k=2,3 -> W index k*4+i: 8,9, 12,13
    # C10: i=2,3; k=0,1 -> W index k*4+i: 2,3, 6,7
    # C11: i=2,3; k=2,3 -> W index k*4+i: 10,11, 14,15

    # Strassen formulas for block matrices:
    # M1 = (A11 - A00)(B10 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B00 - B11)
    # M4 = A11(B11 - B00)
    # M5 = (A00 + A11)(B00 + B11)
    # M6 = (A10 - A00)(B01 + B00)
    # M7 = (A01 + A11)(B10 - B11)
    
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M6
    # C10 = M2 + M5
    # C11 = M1 - M3 + M4 + M6
    
    # We need to decompose each Mk = Pk * Qk into 7 scalar terms.
    # Pk is a 2x2 matrix of linear forms in A.
    # Qk is a 2x2 matrix of linear forms in B.
    
    # Define the 7 inner Strassen terms for 2x2 matrices X, Y.
    # X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # Z = X Y
    # Inner terms (U_inner, V_inner, W_inner) for Z entries.
    # Z00 = m1 - m4 + m5 + m7
    # Z01 = m3 + m6
    # Z10 = m2 + m5
    # Z11 = m1 - m3 + m4 + m6
    
    # Inner rank-1 components (coeffs for x00, x01, x10, x11 and y00, y01, y10, y11):
    # m1: u=[-1,0,0,1], v=[0,0,1,1], w=[1,0,0,1] (w for z00, z10, z01, z11? No, w is for Z entries)
    # Let's define W_inner for Z entries (z00, z10, z01, z11) corresponding to indices 0, 1, 2, 3 in a 2x2 block context.
    # But we need to map these to the global W indices.
    
    # Let's just generate the scalar linear forms for Pk and Qk.
    
    # Define block indices for A (4x4)
    # A00: rows 0-1, cols 0-1
    # A01: rows 0-1, cols 2-3
    # A10: rows 2-3, cols 0-1
    # A11: rows 2-3, cols 2-3
    
    def get_A_block_coeffs(block):
        # Returns 2x2 matrix of 16-dim vectors (linear forms in A)
        coeffs = [[None]*2 for _ in range(2)]
        for r in range(2):
            for c in range(2):
                vec = [0]*16
                if block == 'A00':
                    vec[r*4 + c] = 1
                elif block == 'A01':
                    vec[r*4 + (c+2)] = 1
                elif block == 'A10':
                    vec[(r+2)*4 + c] = 1
                elif block == 'A11':
                    vec[(r+2)*4 + (c+2)] = 1
                coeffs[r][c] = vec
        return coeffs

    def get_B_block_coeffs(block, p_dim=5):
        # Returns 2x2 matrix of 20-dim vectors (linear forms in B)
        # B is 4 x p_dim. We only use first 4 cols for 4x4x4 part.
        coeffs = [[None]*2 for _ in range(2)]
        for r in range(2):
            for c in range(2):
                vec = [0]*(4*p_dim)
                if block == 'B00':
                    vec[r*p_dim + c] = 1
                elif block == 'B01':
                    vec[r*p_dim + (c+2)] = 1
                elif block == 'B10':
                    vec[(r+2)*p_dim + c] = 1
                elif block == 'B11':
                    vec[(r+2)*p_dim + (c+2)] = 1
                coeffs[r][c] = vec
        return coeffs

    def add_linear_combination(vecs, coeffs, target_vec):
        for v, c in zip(vecs, coeffs):
            for i in range(len(v)):
                target_vec[i] += c * v[i]

    # Define the 7 block multiplications
    # Each is (P_block_A, P_block_B, C_block_contrib)
    # P_block_A is a 2x2 matrix of A-blocks (with signs)
    # P_block_B is a 2x2 matrix of B-blocks (with signs)
    # C_block_contrib is a 2x2 matrix of C-blocks (with signs)
    
    # M1 = (A11 - A00)(B10 + B11)
    # P1 = [[A11, A11], [A11, A11]] - [[A00, A00], [A00, A00]]? No.
    # A11 - A00 is a 2x2 matrix.
    # P1 = A11 - A00.
    # Q1 = B10 + B11.
    # C00 += M1, C11 += M1.
    
    # We will iterate over the 7 block multiplications.
    # For each, we construct the 2x2 matrices P and Q of linear forms.
    # Then we apply the inner Strassen decomposition.
    
    # Inner Strassen terms for Z = X Y:
    # We define the 7 terms (u_coeffs, v_coeffs, w_coeffs) where u, v are 4-vectors (x00..x11, y00..y11)
    # and w is a 4-vector (z00, z10, z01, z11) -- wait, order in W is k*n+i.
    # For 2x2 block C_block, indices are:
    # z00 (i=0,k=0), z10 (i=1,k=0), z01 (i=0,k=1), z11 (i=1,k=1)
    # Global W index for C_block (row offset R, col offset K) in 4x4 C:
    # c_{i,k} -> W[k*4+i].
    # Block C00 (R=0,K=0): c_{i,k} -> W[(k)*4 + i]. i in 0..1, k in 0..1.
    # Block C10 (R=2,K=0): c_{i,k} -> W[(k)*4 + (i+2)].
    # Block C01 (R=0,K=2): c_{i,k} -> W[(k+2)*4 + i].
    # Block C11 (R=2,K=2): c_{i,k} -> W[(k+2)*4 + (i+2)].
    
    # Inner W vector order for a 2x2 block: [z00, z10, z01, z11] corresponds to local i=0,k=0; i=1,k=0; i=0,k=1; i=1,k=1.
    # This matches the order [0, 1, 2, 3] for local indices.
    
    inner_terms = [
        # (u, v, w)
        ([-1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 1]), # m1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]), # m2
        ([1, 0, 0, 0], [1, 0, 0, -1], [0, 0, 1, -1]), # m3
        ([0, 0, 0, 1], [-1, 0, 0, 1], [-1, 0, 0, 1]), # m4
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 0, 0]), # m5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 1, 1]), # m6
        ([0, 1, 0, 1], [0, 0, 1, -1], [1, 0, 0, -1]), # m7
    ]
    
    # Block definitions: (P_blocks, Q_blocks, C_blocks, signs)
    # P_blocks: list of (block_name, sign) for 2x2 matrix
    # Q_blocks: list of (block_name, sign) for 2x2 matrix
    # C_blocks: list of (block_name, sign) for 2x2 matrix
    
    block_defs = [
        # M1
        (
            [('A11', 1), ('A00', -1)], # P = A11 - A00 (scalar mult)
            [('B10', 1), ('B11', 1)], # Q = B10 + B11
            [('C00', 1), ('C11', 1)]  # C00 += M1, C11 += M1
        ),
        # M2
        (
            [('A10', 1), ('A11', 1)], # P = A10 + A11
            [('B00', 1)], # Q = B00
            [('C10', 1)] # C10 += M2
        ),
        # M3
        (
            [('A00', 1)], # P = A00
            [('B00', 1), ('B11', -1)], # Q = B00 - B11
            [('C01', 1), ('C11', -1)] # C01 += M3, C11 -= M3
        ),
        # M4
        (
            [('A11', 1)], # P = A11
            [('B11', 1), ('B00', -1)], # Q = B11 - B00
            [('C00', -1), ('C11', 1)] # C00 -= M4, C11 += M4
        ),
        # M5
        (
            [('A00', 1), ('A11', 1)], # P = A00 + A11
            [('B00', 1), ('B11', 1)], # Q = B00 + B11
            [('C00', 1), ('C10', 1)] # C00 += M5, C10 += M5
        ),
        # M6
        (
            [('A10', 1), ('A00', -1)], # P = A10 - A00
            [('B01', 1), ('B00', 1)], # Q = B01 + B00
            [('C01', 1), ('C11', 1)] # C01 += M6, C11 += M6
        ),
        # M7
        (
            [('A01', 1), ('A11', 1)], # P = A01 + A11
            [('B10', 1), ('B11', -1)], # Q = B10 - B11
            [('C00', 1), ('C11', -1)] # C00 += M7, C11 -= M7
        ),
    ]
    
    col = 0
    for P_list, Q_list, C_list in block_defs:
        # Construct 2x2 matrices of linear forms for P and Q
        P_mat = [[None]*2 for _ in range(2)]
        Q_mat = [[None]*2 for _ in range(2)]
        
        # Initialize P_mat and Q_mat with zeros
        for r in range(2):
            for c in range(2):
                P_mat[r][c] = [0]*16
                Q_mat[r][c] = [0]*20
                
        for block_name, sign in P_list:
            coeffs = get_A_block_coeffs(block_name)
            for r in range(2):
                for c in range(2):
                    for i in range(16):
                        P_mat[r][c][i] += sign * coeffs[r][c][i]
                        
        for block_name, sign in Q_list:
            coeffs = get_B_block_coeffs(block_name)
            for r in range(2):
                for c in range(2):
                    for i in range(20):
                        Q_mat[r][c][i] += sign * coeffs[r][c][i]

        # Now apply inner Strassen
        for u_in, v_in, w_in in inner_terms:
            # u_in is for x00, x01, x10, x11
            # x00 is P_mat[0][0], x01 is P_mat[0][1], etc.
            u_global = [0]*16
            v_global = [0]*20
            w_global = [0]*20 # W size is 20 (5*4)
            
            # Construct u_global
            for i in range(4):
                # u_in[i] corresponds to P_mat[r][c] where i = r*2+c?
                # x00 (0), x01 (1), x10 (2), x11 (3)
                r = i // 2
                c = i % 2
                for k in range(16):
                    u_global[k] += u_in[i] * P_mat[r][c][k]
                    
            # Construct v_global
            for i in range(4):
                r = i // 2
                c = i % 2
                for k in range(20):
                    v_global[k] += v_in[i] * Q_mat[r][c][k]
                    
            # Construct w_global
            # w_in is for z00, z10, z01, z11 (local block indices)
            # We need to add these to the correct global W positions with the signs from C_list
            # C_list tells which global C blocks are affected and with what sign
            # But wait, C_list applies to the whole M_k.
            # The inner w_in distributes the contribution to the 4 entries of the 2x2 block M_k.
            # We need to map the local z entries to global C entries.
            
            # Determine the global C block offset for this M_k?
            # No, C_list contains multiple blocks.
            # Example M1: C00 and C11.
            # The result M1 is a 2x2 matrix.
            # M1 contributes to C00 (with sign 1) and C11 (with sign 1).
            # This means the 2x2 matrix M1 is added to C00 and also to C11?
            # No, Strassen formulas are:
            # C00 = M1 - M4 + M5 + M7
            # C11 = M1 - M3 + M4 + M6
            # So M1 contributes to C00 and C11.
            # The 2x2 matrix M1 is added to the C00 block and the C11 block.
            
            # So for each local z entry (z00, z10, z01, z11), it contributes to:
            # - The corresponding entry in the first C block in C_list
            # - The corresponding entry in the second C block in C_list
            # etc.
            
            for c_block_name, c_sign in C_list:
                # Map local z indices to global W indices
                # c_block_name is like 'C00', 'C10', etc.
                # Determine row offset R and col offset K
                R = int(c_block_name[1]) * 2
                K = int(c_block_name[2]) * 2
                
                # Local z00 (i=0, k=0) -> global i=R, k=K -> W[K*4 + R]
                # Local z10 (i=1, k=0) -> global i=R+1, k=K -> W[K*4 + R+1]
                # Local z01 (i=0, k=1) -> global i=R, k=K+1 -> W[(K+1)*4 + R]
                # Local z11 (i=1, k=1) -> global i=R+1, k=K+1 -> W[(K+1)*4 + R+1]
                
                w_indices = [
                    K*4 + R,
                    K*4 + R + 1,
                    (K+1)*4 + R,
                    (K+1)*4 + R + 1
                ]
                
                for i in range(4):
                    w_global[w_indices[i]] += c_sign * w_in[i]
            
            add_term(u_global, v_global, w_global, col)
            col += 1

    # --- Part 2: 4x4x1 using Schoolbook (Rank 16) ---
    # A is 4x4, b is 4x1 (column 4 of B)
    # c_i = sum_j a_ij b_j
    
    for i in range(4):
        for j in range(4):
            u_vec = [0]*16
            v_vec = [0]*20
            w_vec = [0]*20
            
            u_vec[i*4 + j] = 1
            v_vec[j*5 + 4] = 1 # B[j, 4]
            w_vec[4*4 + i] = 1 # C[i, 4] -> W[4*4 + i]
            
            add_term(u_vec, v_vec, w_vec, col)
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
