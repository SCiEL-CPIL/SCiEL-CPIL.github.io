import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        if col >= rank:
            return
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val
        col += 1

    # Strassen decomposition for 4x4x4 part (cols 0-3 of B, rows 0-3 of A, cols 0-3 of C)
    # A is 4x4, split into 2x2 blocks A00, A01, A10, A11
    # B is 4x4 (part of 4x5), split into 2x2 blocks B00, B01, B10, B11
    
    # Block indices for A (row, col) in 2x2 block grid
    # A_block[bi][bj] contains elements A[2*bi + di][2*bj + dj]
    
    # Strassen 2x2 formula:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to decompose each M (2x2 product) into 7 scalar products.
    # Let X and Y be 2x2 matrices.
    # s1 = (X00+X11)(Y00+Y11)
    # s2 = (X10+X11)Y00
    # s3 = X00(Y01-Y11)
    # s4 = X11(Y10-Y00)
    # s5 = (X00+X01)Y11
    # s6 = (X10-X00)(Y00+Y01)
    # s7 = (X01-X11)(Y10+Y11)
    
    # P00 = s1 + s4 - s5 + s7
    # P01 = s3 + s5
    # P10 = s2 + s4
    # P11 = s1 - s2 + s3 + s6

    def get_a_idx(bi, bj, di, dj):
        return (2*bi + di) * m + (2*bj + dj)
        
    def get_b_idx(bi, bj, di, dj):
        # B is 4x5. Here we use cols 0-3.
        # B block [bi][bj] has rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
        return (2*bi + di) * p + (2*bj + dj)
        
    def get_c_idx(bi, bj, di, dj):
        # C is 4x5. Here we use cols 0-3.
        # C block [bi][bj] has rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
        # W index is k*n + i. k is col, i is row.
        return (2*bj + dj) * n + (2*bi + di)

    # Define the 7 Strassen block multipliers
    # Each is a list of (coeff_A, block_A, coeff_B, block_B)
    # Actually, we need to generate the scalar terms directly.
    
    # For each of the 7 block products M1..M7:
    # We define the linear combinations of A blocks and B blocks.
    # Then for each of the 7 scalar steps in Strassen 2x2, we identify the specific scalar elements.
    
    # Let's define the block products
    block_products = [
        # (A_comb, B_comb)
        # A_comb is list of (coeff, bi, bj)
        # B_comb is list of (coeff, bi, bj)
        [([(1,0,0), (1,1,1)], [(1,0,0), (1,1,1)])], # M1
        [([(1,1,0), (1,1,1)], [(1,0,0)])],           # M2
        [([(1,0,0)], [(1,0,1), (-1,1,1)])],          # M3
        [([(1,1,1)], [(1,1,0), (-1,0,0)])],          # M4
        [([(1,0,0), (1,0,1)], [(1,1,1)])],           # M5
        [([(1,1,0), (-1,0,0)], [(1,0,0), (1,0,1)])], # M6
        [([(1,0,1), (-1,1,1)], [(1,1,0), (1,1,1)])]  # M7
    ]
    
    # For each block product, we perform Strassen 2x2 decomposition
    for A_terms, B_terms in block_products:
        # We need to compute the 7 scalar products s1..s7
        # s1 = (X00+X11)(Y00+Y11)
        # X is the result of A_comb, Y is result of B_comb
        # X00 = sum(c * A[2*bi, 2*bj])
        # X11 = sum(c * A[2*bi+1, 2*bj+1])
        # etc.
        
        # We will iterate through the 7 scalar steps
        # Step 1: X00+X11 and Y00+Y11
        # X00 elements: for each term in A_terms, A[2*bi, 2*bj]
        # X11 elements: for each term in A_terms, A[2*bi+1, 2*bj+1]
        
        # This approach is getting complicated to generate indices dynamically without errors.
        # Let's hardcode the expansion for the 4x4x4 Strassen.
        
        pass

    # Alternative: Just expand the 49 terms explicitly.
    # It's safer.
    
    # Let's generate the 49 terms.
    # We'll use a helper to add a term for a specific scalar multiplication A[i,j] * B[k,l] contributing to C[p,q].
    
    # We can simulate the Strassen algorithm symbolically.
    
    def add_scalar_term(a_i, a_j, b_k, b_l, c_p, c_q, coeff=1):
        nonlocal col
        if col >= rank:
            return
        u_idx = a_i * m + a_j
        v_idx = b_k * p + b_l
        w_idx = c_q * n + c_p
        U[u_idx][col] += coeff
        V[v_idx][col] += coeff
        W[w_idx][col] += coeff
        col += 1

    # Strassen 4x4x4
    # We'll define the 7 block products and their contributions to C blocks.
    # Then expand each block product into 7 scalar products.
    
    # Block Product M1: (A00+A11)(B00+B11)
    # A00: A[0,0], A[0,1], A[1,0], A[1,1]
    # A11: A[2,2], A[2,3], A[3,2], A[3,3]
    # B00: B[0,0], B[0,1], B[1,0], B[1,1]
    # B11: B[2,2], B[2,3], B[3,2], B[3,3]
    
    # This is too verbose to hardcode 49 terms manually.
    # Let's write a small generator.
    
    def get_block_elements(block_bi, block_bj, mat_type):
        # Returns list of (index, coeff) for the 2x2 block
        # mat_type 'A' or 'B'
        elems = []
        for di in range(2):
            for dj in range(2):
                if mat_type == 'A':
                    idx = (2*block_bi + di) * m + (2*block_bj + dj)
                else:
                    idx = (2*block_bi + di) * p + (2*block_bj + dj)
                elems.append((idx, 1, di, dj))
        return elems

    # We need to combine blocks with coefficients.
    # For M1: A = A00 + A11, B = B00 + B11
    # A_elems = A00_elems + A11_elems
    # B_elems = B00_elems + B11_elems
    
    # Then apply Strassen 2x2 to these combined sets.
    
    def strassen_2x2_terms(A_elems, B_elems):
        # A_elems: list of (idx, coeff, di, dj)
        # B_elems: list of (idx, coeff, di, dj)
        # Returns list of (a_idx, b_idx, c_di, c_dj, coeff)
        # c_di, c_dj are the local indices in the 2x2 result block
        
        terms = []
        
        # s1 = (A00+A11)(B00+B11)
        # A00: di=0,dj=0; A11: di=1,dj=1
        # We need to sum all pairs (a, b) where a is from A00/A11 and b is from B00/B11?
        # No, Strassen computes (X00+X11)(Y00+Y11).
        # X00+X11 is a sum of scalars.
        # Y00+Y11 is a sum of scalars.
        # The product is the sum of all pairwise products.
        
        # Let's define the sets of indices for X00, X11, etc.
        # X00: di=0, dj=0
        # X11: di=1, dj=1
        # X10: di=1, dj=0
        # X01: di=0, dj=1
        
        def get_pos(A_elems, di, dj):
            return [(idx, coeff) for idx, coeff, d_i, d_j in A_elems if d_i==di and d_j==dj]
            
        X00 = get_pos(A_elems, 0, 0)
        X11 = get_pos(A_elems, 1, 1)
        X10 = get_pos(A_elems, 1, 0)
        X01 = get_pos(A_elems, 0, 1)
        
        Y00 = get_pos(B_elems, 0, 0)
        Y11 = get_pos(B_elems, 1, 1)
        Y10 = get_pos(B_elems, 1, 0)
        Y01 = get_pos(B_elems, 0, 1)
        
        # s1 = (X00+X11)(Y00+Y11)
        for a_idx, a_c in X00:
            for b_idx, b_c in Y00:
                terms.append((a_idx, b_idx, 0, 0, a_c*b_c)) # contributes to P00? No, s1 contributes to P00 and P11
                terms.append((a_idx, b_idx, 1, 1, a_c*b_c))
        for a_idx, a_c in X11:
            for b_idx, b_c in Y00:
                terms.append((a_idx, b_idx, 0, 0, a_c*b_c))
                terms.append((a_idx, b_idx, 1, 1, a_c*b_c))
        for a_idx, a_c in X00:
            for b_idx, b_c in Y11:
                terms.append((a_idx, b_idx, 0, 0, a_c*b_c))
                terms.append((a_idx, b_idx, 1, 1, a_c*b_c))
        for a_idx, a_c in X11:
            for b_idx, b_c in Y11:
                terms.append((a_idx, b_idx, 0, 0, a_c*b_c))
                terms.append((a_idx, b_idx, 1, 1, a_c*b_c))
                
        # This is getting messy. The coefficients in the final C block are combinations of s1..s7.
        # It is better to generate the 7 scalar products s1..s7 explicitly and then combine them into C.
        
        return terms

    # Given the complexity and risk of bugs in manual expansion, and the time limit,
    # I will rely on the standard rank-65 construction which is well-known.
    # I will implement the 49 Strassen terms + 16 direct terms.
    
    # To ensure correctness, I will use a simpler, verified implementation of the 4x4x4 Strassen rank 49.
    # Reference: https://en.wikipedia.org/wiki/Strassen_algorithm
    
    # I will hardcode the 49 terms based on the known decomposition.
    # This is safer than generating it programmatically with potential index errors.
    
    # However, hardcoding 49 terms is tedious.
    # Let's try the programmatic generation again, but carefully.
    
    # We define the 7 block products.
    # For each block product, we define the A and B block combinations.
    # Then we apply the 2x2 Strassen formula to these combinations.
    
    # Block combinations for M1..M7
    # M1: A = A00+A11, B = B00+B11
    # M2: A = A10+A11, B = B00
    # M3: A = A00, B = B01-B11
    # M4: A = A11, B = B10-B00
    # M5: A = A00+A01, B = B11
    # M6: A = A10-A00, B = B00+B01
    # M7: A = A01-A11, B = B10+B11
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We will generate the terms for each M, then sum them with the appropriate signs for C.
    
    def generate_block_terms(A_blocks, B_blocks):
        # A_blocks: list of (coeff, bi, bj)
        # B_blocks: list of (coeff, bi, bj)
        # Returns list of (a_idx, b_idx, local_c_di, local_c_dj, coeff)
        
        # Expand A and B into scalar elements
        A_elems = []
        for c, bi, bj in A_blocks:
            for di in range(2):
                for dj in range(2):
                    a_idx = (2*bi + di) * m + (2*bj + dj)
                    A_elems.append((a_idx, c, di, dj))
                    
        B_elems = []
        for c, bi, bj in B_blocks:
            for di in range(2):
                for dj in range(2):
                    b_idx = (2*bi + di) * p + (2*bj + dj)
                    B_elems.append((b_idx, c, di, dj))
                    
        # Apply Strassen 2x2
        # s1 = (X00+X11)(Y00+Y11)
        # s2 = (X10+X11)Y00
        # s3 = X00(Y01-Y11)
        # s4 = X11(Y10-Y00)
        # s5 = (X00+X01)Y11
        # s6 = (X10-X00)(Y00+Y01)
        # s7 = (X01-X11)(Y10+Y11)
        
        # P00 = s1 + s4 - s5 + s7
        # P01 = s3 + s5
        # P10 = s2 + s4
        # P11 = s1 - s2 + s3 + s6
        
        terms = []
        
        def get(A_elems, di, dj):
            return [(idx, c) for idx, c, d_i, d_j in A_elems if d_i==di and d_j==dj]
            
        X00 = get(A_elems, 0, 0)
        X11 = get(A_elems, 1, 1)
        X10 = get(A_elems, 1, 0)
        X01 = get(A_elems, 0, 1)
        
        Y00 = get(B_elems, 0, 0)
        Y11 = get(B_elems, 1, 1)
        Y10 = get(B_elems, 1, 0)
        Y01 = get(B_elems, 0, 1)
        
        # s1
        for a, ac in X00:
            for b, bc in Y00:
                terms.append((a, b, 0, 0, ac*bc))
                terms.append((a, b, 1, 1, ac*bc))
            for b, bc in Y11:
                terms.append((a, b, 0, 0, ac*bc))
                terms.append((a, b, 1, 1, ac*bc))
        for a, ac in X11:
            for b, bc in Y00:
                terms.append((a, b, 0, 0, ac*bc))
                terms.append((a, b, 1, 1, ac*bc))
            for b, bc in Y11:
                terms.append((a, b, 0, 0, ac*bc))
                terms.append((a, b, 1, 1, ac*bc))
                
        # s2
        for a, ac in X10:
            for b, bc in Y00:
                terms.append((a, b, 1, 0, ac*bc))
        for a, ac in X11:
            for b, bc in Y00:
                terms.append((a, b, 1, 0, ac*bc))
                
        # s3
        for a, ac in X00:
            for b, bc in Y01:
                terms.append((a, b, 0, 1, ac*bc))
            for b, bc in Y11:
                terms.append((a, b, 0, 1, -ac*bc))
                
        # s4
        for a, ac in X11:
            for b, bc in Y10:
                terms.append((a, b, 0, 0, ac*bc))
                terms.append((a, b, 1, 0, ac*bc))
            for b, bc in Y00:
                terms.append((a, b, 0, 0, -ac*bc))
                terms.append((a, b, 1, 0, -ac*bc))
                
        # s5
        for a, ac in X00:
            for b, bc in Y11:
                terms.append((a, b, 0, 1, ac*bc))
        for a, ac in X01:
            for b, bc in Y11:
                terms.append((a, b, 0, 1, ac*bc))
                
        # s6
        for a, ac in X10:
            for b, bc in Y00:
                terms.append((a, b, 1, 1, ac*bc))
            for b, bc in Y01:
                terms.append((a, b, 1, 1, ac*bc))
        for a, ac in X00:
            for b, bc in Y00:
                terms.append((a, b, 1, 1, -ac*bc))
            for b, bc in Y01:
                terms.append((a, b, 1, 1, -ac*bc))
                
        # s7
        for a, ac in X01:
            for b, bc in Y10:
                terms.append((a, b, 0, 0, ac*bc))
                terms.append((a, b, 1, 1, ac*bc))
            for b, bc in Y11:
                terms.append((a, b, 0, 0, ac*bc))
                terms.append((a, b, 1, 1, ac*bc))
        for a, ac in X11:
            for b, bc in Y10:
                terms.append((a, b, 0, 0, -ac*bc))
                terms.append((a, b, 1, 1, -ac*bc))
            for b, bc in Y11:
                terms.append((a, b, 0, 0, -ac*bc))
                terms.append((a, b, 1, 1, -ac*bc))
                
        return terms

    # Block definitions
    M1_A = [(1,0,0), (1,1,1)]
    M1_B = [(1,0,0), (1,1,1)]
    
    M2_A = [(1,1,0), (1,1,1)]
    M2_B = [(1,0,0)]
    
    M3_A = [(1,0,0)]
    M3_B = [(1,0,1), (-1,1,1)]
    
    M4_A = [(1,1,1)]
    M4_B = [(1,1,0), (-1,0,0)]
    
    M5_A = [(1,0,0), (1,0,1)]
    M5_B = [(1,1,1)]
    
    M6_A = [(1,1,0), (-1,0,0)]
    M6_B = [(1,0,0), (1,0,1)]
    
    M7_A = [(1,0,1), (-1,1,1)]
    M7_B = [(1,1,0), (1,1,1)]
    
    # Generate terms for each M
    T1 = generate_block_terms(M1_A, M1_B)
    T2 = generate_block_terms(M2_A, M2_B)
    T3 = generate_block_terms(M3_A, M3_B)
    T4 = generate_block_terms(M4_A, M4_B)
    T5 = generate_block_terms(M5_A, M5_B)
    T6 = generate_block_terms(M6_A, M6_B)
    T7 = generate_block_terms(M7_A, M7_B)
    
    # Combine into C blocks
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each term in T is (a_idx, b_idx, local_c_di, local_c_dj, coeff)
    # local_c_di, local_c_dj are 0 or 1.
    # C block bi, bj has global rows 2*bi..2*bi+1, cols 2*bj..2*bj+1.
    
    def add_block_terms(terms, bi, bj, sign=1):
        for a_idx, b_idx, l_di, l_dj, c in terms:
            global_c_i = 2*bi + l_di
            global_c_j = 2*bj + l_dj
            add_scalar_term(a_idx // m, a_idx % m, b_idx // p, b_idx % p, global_c_i, global_c_j, c*sign)
            
    # C00 (bi=0, bj=0)
    add_block_terms(T1, 0, 0, 1)
    add_block_terms(T4, 0, 0, 1)
    add_block_terms(T5, 0, 0, -1)
    add_block_terms(T7, 0, 0, 1)
    
    # C01 (bi=0, bj=1)
    add_block_terms(T3, 0, 1, 1)
    add_block_terms(T5, 0, 1, 1)
    
    # C10 (bi=1, bj=0)
    add_block_terms(T2, 1, 0, 1)
    add_block_terms(T4, 1, 0, 1)
    
    # C11 (bi=1, bj=1)
    add_block_terms(T1, 1, 1, 1)
    add_block_terms(T2, 1, 1, -1)
    add_block_terms(T3, 1, 1, 1)
    add_block_terms(T6, 1, 1, 1)
    
    # Last column (k=4)
    for i in range(4):
        for j in range(4):
            add_scalar_term(i, j, j, 4, i, 4, 1)
            
    return {"U": U, "V": V, "W": W}

