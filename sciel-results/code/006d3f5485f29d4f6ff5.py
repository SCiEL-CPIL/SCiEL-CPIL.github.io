import numpy as np
import time

def construct():
    n, m, p = 4, 4, 6
    start_time = time.time()
    DEADLINE = 120.0
    
    # We'll build a decomposition using recursive Strassen.
    # For the 4x4x4 part: use Strassen (7 recursive 2x2x2 calls)
    # For the 4x4x2 part: use schoolbook (32 terms)
    # Total: 7 + 32 = 39? No, Strassen for 4x4x4 needs 7 terms where each term is a 2x2x2 multiplication.
    # Each 2x2x2 multiplication uses 7 scalar multiplications.
    # So 4x4x4 via Strassen = 7 * 7 = 49? No, that's not right either.
    # 
    # Actually, Strassen's algorithm for 4x4x4:
    # Split into four 2x2 blocks. Compute 7 products of 2x2 matrices.
    # Each 2x2x2 product uses 7 scalar multiplications (Strassen) or 4 (schoolbook).
    # If we use schoolbook for the 2x2x2: 7 * 4 = 28
    # If we use Strassen for the 2x2x2: 7 * 7 = 49
    #
    # Wait, the base case 2x2x2 has rank 7 with Strassen.
    # So 4x4x4 with one level of Strassen: 7 terms, each is a 2x2x2 multiplication with rank 7.
    # Total rank = 7 * 7 = 49.
    #
    # But we can do better. For 4x4x4, the known best is 49 (using Strassen).
    # For 4x4x2, the schoolbook is 32. Can we do better?
    # 4x4x2: Split A into two 4x2 blocks, multiply each by 4x2 block of B... 
    # Actually, 4x4x2 means A is 4x4, B is 4x2, C is 4x2.
    # We can split A into two 2x4 blocks and B into two 2x2 blocks.
    # C_top = A_top * B_top (2x4 * 4x2 = 2x2)
    # C_bot = A_bot * B_bot (2x4 * 4x2 = 2x2)
    # Each 2x4x2 multiplication: split into 2x2x2 blocks.
    # 2x4x2: A is 2x4, B is 4x2. Split A into [A11 A12], B into [B11; B21]
    # C = A11*B11 + A12*B21. Each is 2x2x2.
    # Using Strassen for each: 2 * 7 = 14 terms.
    # Total for 4x4x2: 14 terms.
    #
    # So total: 49 + 14 = 63? That's better than 96!
    #
    # Let me implement this carefully.

    # We'll track the rank as we build
    U = []  # Will be list of lists, shape (n*m, rank)
    V = []  # shape (m*p, rank)
    W = []  # shape (p*n, rank)
    
    def add_term(u_vec, v_vec, w_vec):
        """Add a single rank-1 term. u_vec, v_vec, w_vec are dicts mapping index to coeff."""
        max_u = n * m
        max_v = m * p
        max_w = p * n
        for i in range(len(U)):
            U[i].append(u_vec.get(i, 0))
            V[i].append(v_vec.get(i, 0))
            W[i].append(w_vec.get(i, 0))
    
    # Initialize U, V, W
    for i in range(n * m):
        U.append([0])
        V.append([0])
        W.append([0])
    # Actually, let me use a different approach: collect terms first
    
    terms = []  # List of (u_dict, v_dict, w_dict)
    
    def add_term(u_dict, v_dict, w_dict):
        terms.append((u_dict, v_dict, w_dict))
    
    # Helper to get index in U: A[i,j] -> i*m + j
    def u_idx(i, j):
        return i * m + j
    
    # Helper to get index in V: B[j,k] -> j*p + k
    def v_idx(j, k):
        return j * p + k
    
    # Helper to get index in W: C[i,k] -> k*n + i (trace convention W[k*n+i])
    def w_idx(k, i):
        return k * n + i
    
    # Strassen 2x2x2: decompose A(2x2) * B(2x2) = C(2x2)
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6 + M7
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    def strassen_2x2x2(a_offsets, b_offsets, c_offsets, sign=1):
        """
        a_offsets: dict with keys 'A00','A01','A10','A11' mapping to (i,j) or list of (i,j,coeff)
        b_offsets: dict with keys 'B00','B01','B10','B11' mapping to (j,k) or list
        c_offsets: dict with keys 'C00','C01','C10','C11' mapping to (i,k) or list
        """
        # For simplicity, assume each block is a single element with coeff 1
        # We'll handle the linear combinations in the calling code
        
        pass
    
    # Let me use a more direct approach. I'll define the 7 Strassen multipliers
    # and for each, build the U, V, W vectors.
    
    def add_strassen_block(i0, j0, k0):
        """
        Add Strassen decomposition for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2] = C[i0:i0+2, k0:k0+2]
        """
        # M1 = (A00+A11)(B00+B11)
        u1 = {u_idx(i0, j0): 1, u_idx(i0+1, j0+1): 1}
        v1 = {v_idx(j0, k0): 1, v_idx(j0+1, k0+1): 1}
        w1 = {w_idx(k0, i0): 1, w_idx(k0+1, i0+1): 1}
        add_term(u1, v1, w1)
        
        # M2 = (A10+A11)B00
        u2 = {u_idx(i0+1, j0): 1, u_idx(i0+1, j0+1): 1}
        v2 = {v_idx(j0, k0): 1}
        w2 = {w_idx(k0+1, i0): 1}  # C10 += M2
        add_term(u2, v2, w2)
        
        # M3 = A00(B01-B11)
        u3 = {u_idx(i0, j0): 1}
        v3 = {v_idx(j0, k0+1): 1, v_idx(j0+1, k0+1): -1}
        w3 = {w_idx(k0+1, i0): 1}  # C01 += M3
        add_term(u3, v3, w3)
        
        # M4 = A11(B10-B00)
        u4 = {u_idx(i0+1, j0+1): 1}
        v4 = {v_idx(j0+1, k0): 1, v_idx(j0, k0): -1}
        w4 = {w_idx(k0, i0): -1, w_idx(k0+1, i0): 1}  # C00 -= M4, C10 += M4
        add_term(u4, v4, w4)
        
        # M5 = (A00+A01)B11
        u5 = {u_idx(i0, j0): 1, u_idx(i0, j0+1): 1}
        v5 = {v_idx(j0+1, k0+1): 1}
        w5 = {w_idx(k0, i0): 1, w_idx(k0+1, i0+1): 1}  # C00 += M5, C11 += M5
        add_term(u5, v5, w5)
        
        # M6 = (A10-A00)(B00+B01)
        u6 = {u_idx(i0+1, j0): 1, u_idx(i0, j0): -1}
        v6 = {v_idx(j0, k0): 1, v_idx(j0, k0+1): 1}
        w6 = {w_idx(k0+1, i0+1): 1}  # C11 += M6
        add_term(u6, v6, w6)
        
        # M7 = (A01-A11)(B10+B11)
        u7 = {u_idx(i0, j0+1): 1, u_idx(i0+1, j0+1): -1}
        v7 = {v_idx(j0+1, k0): 1, v_idx(j0+1, k0+1): 1}
        w7 = {w_idx(k0, i0): -1, w_idx(k0+1, i0+1): 1}  # C00 -= M7, C11 += M7
        add_term(u7, v7, w7)
    
    # For 4x4x4: We need to compute A(4x4) * B(4x4) = C(4x4)
    # Split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where A11 is top-left 2x2, etc.
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    #
    # Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M7
    # C21 = M2 + M4
    # C22 = M1 - M3 + M6 + M7
    #
    # Each M is a 2x2x2 multiplication. We use Strassen for each.
    
    # A block offsets: A11 at (0,0), A12 at (0,2), A21 at (2,0), A22 at (2,2)
    # B block offsets: B11 at (0,0), B12 at (0,2), B21 at (2,0), B22 at (2,2)
    # C block offsets: C11 at (0,0), C12 at (0,2), C21 at (2,0), C22 at (2,2)
    
    # M1 = (A11+A22)(B11+B22)
    # This is a sum of 4 products: A11*B11, A11*B22, A22*B11, A22*B22
    # We need to add Strassen for each, but with appropriate signs and C offsets.
    #
    # Actually, it's cleaner to think of it as:
    # M1 = (A11+A22)(B11+B22) is itself a 2x2x2 multiplication where
    # the "effective A" is A11+A22 and "effective B" is B11+B22.
    # But A11+A22 is not a 2x2 matrix in the same position - it's the sum of two 2x2 blocks.
    #
    # The correct way: M1 contributes to C11 and C22.
    # M1 is computed as a 2x2x2 Strassen, but the A and B inputs are linear combinations.
    #
    # Let me define a helper that adds a Strassen decomposition where the A and B blocks
    # are given as linear combinations of global A and B elements.
    
    def add_strassen_linear(a_combo, b_combo, c_combos, sign=1):
        """
        a_combo: list of (i, j, coeff) for A elements
        b_combo: list of (j, k, coeff) for B elements
        c_combos: list of (i, k, coeff) for C elements
        sign: overall sign multiplier
        """
        # Build the 2x2x2 Strassen with linear combinations
        # This is complex. Let me use a different approach.
        
        pass
    
    # Given the complexity, let me just use the schoolbook for the 4x4x4 part
    # and Strassen for the 4x4x2 part, or vice versa.
    #
    # Actually, let me try a different decomposition:
    # 4x4x6 = 4x4x4 + 4x4x2
    # For 4x4x4: use Strassen (rank 49)
    # For 4x4x2: use schoolbook (rank 32)
    # Total: 81
    #
    # Or:
    # For 4x4x4: use schoolbook (rank 64)
    # For 4x4x2: use Strassen (rank 14)
    # Total: 78
    #
    # Let me implement 4x4x2 with Strassen and 4x4x4 with schoolbook.
    #
    # 4x4x2: A(4x4) * B(4x2) = C(4x2)
    # Split A into A_top (2x4) and A_bot (2x4)
    # Split B into B_top (2x2) and B_bot (2x2)
    # C_top = A_top * B_top (2x4 * 4x2)
    # C_bot = A_bot * B_bot (2x4 * 4x2)
    #
    # For 2x4x2: A(2x4) * B(4x2) = C(2x2)
    # Split A into [A1 A2] where A1, A2 are 2x2
    # Split B into [B1; B2] where B1, B2 are 2x2
    # C = A1*B1 + A2*B2
    # Each is 2x2x2. Use Strassen for each: 7 + 7 = 14 terms.
    
    # Let me implement this.
    
    def add_2x4x2_strassen(i0, j0, k0):
        """
        Add Strassen decomposition for A[i0:i0+2, j0:j0+4] * B[j0:j0+4, k0:k0+2] = C[i0:i0+2, k0:k0+2]
        """
        # C = A1*B1 + A2*B2
        # A1 = A[i0:i0+2, j0:j0+2], A2 = A[i0:i0+2, j0+2:j0+4]
        # B1 = B[j0:j0+2, k0:k0+2], B2 = B[j0+2:j0+4, k0:k0+2]
        
        # Strassen for A1*B1 (2x2x2)
        add_strassen_block(i0, j0, k0)
        # Strassen for A2*B2 (2x2x2)
        add_strassen_block(i0, j0+2, k0)
    
    # For 4x4x2:
    # C_top = A[0:2, :] * B[0:2, 0:2]  (2x4 * 4x2)
    # C_bot = A[2:4, :] * B[2:4, 0:2]  (2x4 * 4x2)
    
    # Add 2x4x2 for top part
    add_2x4x2_strassen(0, 0, 0)
    # Add 2x4x2 for bottom part
    add_2x4x2_strassen(2, 0, 0)
    
    # Now add schoolbook for 4x4x4
    for i in range(4):
        for j in range(4):
            for k in range(4):
                u = {u_idx(i, j): 1}
                v = {v_idx(j, k): 1}
                w = {w_idx(k, i): 1}
                add_term(u, v, w)
    
    # Build U, V, W
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (u_dict, v_dict, w_dict) in enumerate(terms):
        for idx, coeff in u_dict.items():
            U[idx][col] = coeff
        for idx, coeff in v_dict.items():
            V[idx][col] = coeff
        for idx, coeff in w_dict.items():
            W[idx][col] = coeff
    
    return {"U": U, "V": V, "W": W}
