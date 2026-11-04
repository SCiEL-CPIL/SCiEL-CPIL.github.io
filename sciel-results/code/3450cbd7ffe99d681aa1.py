import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6

    # We want to decompose the matrix multiplication tensor for A(4x4) * B(4x6) = C(4x6)
    # using a rank-minimal (or near-minimal) tensor decomposition.
    #
    # Strategy: Use a recursive/block approach.
    # 4x4 * 4x6 can be viewed as 2x2 blocks of A times 2x3 blocks of B.
    # A = [[A00, A01], [A10, A11]] where each block is 2x2
    # B = [[B00, B01], [B10, B11]] where each block is 2x3
    # C = [[C00, C01], [C10, C11]] where each block is 2x3
    #
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    #
    # Each C block (2x3) is the sum of two 2x2 * 2x3 products.
    #
    # For a 2x2 * 2x3 product, we can use:
    # - Strassen for the 2x2 * 2x2 part (first 2 columns of B): rank 7
    # - Schoolbook for the 2x2 * 2x1 part (last column of B): rank 4
    # Total per 2x2 * 2x3 product: rank 11
    #
    # But wait, we can do better. Let me think about this differently.
    #
    # Actually, let's use a known result. The rank of 4x4 * 4x6 matrix multiplication:
    # Using Strassen recursively:
    # 4x4 * 4x4 has rank at most 49 (Strassen^2)
    # 4x4 * 4x6 = 4x4 * (4x4 + 4x2)
    # 4x4 * 4x4: rank 49
    # 4x4 * 4x2: this is 4x4 * (2x2 + 2x2) but with different dimensions
    #
    # Let me think more carefully. 
    #
    # Alternative approach: Use the fact that 4x4 * 4x6 can be decomposed as:
    # Split B into B_left (4x4) and B_right (4x2)
    # C_left (4x4) = A * B_left: rank 49 (Strassen^2)
    # C_right (4x2) = A * B_right: rank ?
    #
    # For 4x4 * 4x2:
    # Split into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] (2x2 blocks)
    # B_right = [[B00, B01], [B10, B11]] where each is 2x2
    # C_right = [[C00, C01], [C10, C11]] where each is 2x2
    #
    # C00 = A00*B00 + A01*B10: 2x2*2x2 + 2x2*2x2
    # Each 2x2*2x2 has rank 7 (Strassen)
    # So C00 needs rank 14? No, we can share.
    #
    # Actually, let's just use a straightforward approach with rank 63 or similar.
    #
    # Let me use a different strategy: direct construction using known decompositions.
    #
    # For 4x4 * 4x6, the minimum rank is known to be at most 63 (from literature).
    # Let's construct a rank 63 decomposition.
    #
    # Actually, let's use a simpler approach: 
    # 4x4 * 4x6 can be done with rank 64 using:
    # - 4x4 * 4x4 part with rank 49 (Strassen^2)
    # - 4x4 * 4x2 part with rank 15 (using Strassen-like for 4x4 * 4x2)
    # Total: 49 + 15 = 64
    #
    # For 4x4 * 4x2:
    # Split A into 2x2 blocks, B_right into 2x2 blocks
    # C_right[i,j] = sum_k A[i,k] * B_right[k,j]
    # This is 4x4 * 4x2
    # Using Strassen-like:
    # We can compute 4x4 * 4x2 with rank 15 by:
    # - Treating it as two 4x4 * 4x1 problems? No, that's worse.
    # - Using the fact that 4x4 * 4x2 = (4x4 * 4x4) restricted to 2 columns
    #
    # Let me just use a known construction. I'll use rank 77 from the current program
    # but try to optimize it.
    #
    # Actually, let me try a different approach: use the Coppersmith-Winograd style
    # or just a direct tensor decomposition search.
    #
    # For now, let's use the rank 77 approach but clean it up and verify correctness.
    #
    # Wait, I can do better. Let me think about 4x4 * 4x6 more carefully.
    #
    # Using block Strassen:
    # A (4x4) = [[A00, A01], [A10, A11]] (2x2 blocks)
    # B (4x6) = [[B00, B01, B02], [B10, B11, B12]] (2x3 blocks)
    # C (4x6) = [[C00, C01, C02], [C10, C11, C12]] (2x3 blocks)
    #
    # Ci,j = Ai,0 * B0,j + Ai,1 * B1,j for i=0,1 and j=0,1,2
    #
    # For each i,j, Ci,j (2x3) = Ai,0 (2x2) * B0,j (2x3) + Ai,1 (2x2) * B1,j (2x3)
    #
    # Each 2x2 * 2x3 product has rank 9 (7 for Strassen on 2x2*2x2 part + 2 for the remaining column)
    # Wait, 2x2 * 2x3:
    #   = 2x2 * (2x2 + 2x1)
    #   = (2x2 * 2x2) + (2x2 * 2x1)
    #   Rank 7 + Rank 4 = Rank 11
    #
    # So each Ci,j needs rank 11, and we have 4 such blocks, total rank 44?
    # But the products share A and B factors, so we can't just sum ranks.
    #
    # Let me use a different approach: I'll construct the decomposition explicitly
    # using the 7 Strassen steps for the 4x4 * 4x4 part and handle the extra 2 columns separately.
    #
    # For the 4x4 * 4x4 part (columns 0-3 of B):
    # Use Strassen^2 with rank 49
    #
    # For the 4x4 * 4x2 part (columns 4-5 of B):
    # Use a rank 15 decomposition
    #
    # Total rank: 49 + 15 = 64
    #
    # Let me construct this.
    
    # First, let's define the Strassen steps for 2x2 * 2x2
    strassen_2x2 = [
        ({'00': 1, '11': 1}, {'00': 1, '11': 1}, {'00': 1, '11': 1}),
        ({'10': 1, '11': 1}, {'00': 1}, {'10': 1, '11': -1}),
        ({'00': 1}, {'01': 1, '11': -1}, {'01': 1, '11': 1}),
        ({'11': 1}, {'10': 1, '00': -1}, {'00': 1, '10': 1}),
        ({'00': 1, '01': 1}, {'11': 1}, {'00': -1, '01': 1}),
        ({'10': 1, '00': -1}, {'00': 1, '01': 1}, {'11': 1}),
        ({'01': 1, '11': -1}, {'10': 1, '11': 1}, {'00': 1}),
    ]
    
    # For 4x4 * 4x4 using Strassen^2:
    # Split into 2x2 blocks, apply Strassen at top level, then Strassen at bottom level
    # Top-level Strassen steps for 4x4 * 4x4:
    top_strassen = [
        ({'00': 1, '11': 1}, {'00': 1, '11': 1}, {'00': 1, '11': 1}),
        ({'10': 1, '11': 1}, {'00': 1}, {'10': 1, '11': -1}),
        ({'00': 1}, {'01': 1, '11': -1}, {'01': 1, '11': 1}),
        ({'11': 1}, {'10': 1, '00': -1}, {'00': 1, '10': 1}),
        ({'00': 1, '01': 1}, {'11': 1}, {'00': -1, '01': 1}),
        ({'10': 1, '00': -1}, {'00': 1, '01': 1}, {'11': 1}),
        ({'01': 1, '11': -1}, {'10': 1, '11': 1}, {'00': 1}),
    ]
    
    # Each top-level step involves 2x2 blocks of A and B (which are 2x2 matrices)
    # For each top-level step, we need to multiply 2x2 * 2x2 using Strassen (rank 7)
    # So total rank for 4x4 * 4x4: 7 * 7 = 49
    
    # Now for the 4x4 * 4x2 part (columns 4-5):
    # B_right is 4x2, split into 2x2 blocks: [[B00, B01], [B10, B11]]
    # C_right is 4x2, split into 2x2 blocks: [[C00, C01], [C10, C11]]
    #
    # Ci,j = Ai,0 * B0,j + Ai,1 * B1,j for i,j in {0,1}
    #
    # We can use a similar Strassen-like approach:
    # For each of the 4 output blocks Ci,j (2x2), we need:
    # Ci,j = Ai,0 * B0,j + Ai,1 * B1,j
    #
    # Using Strassen on the 2x2 * 2x2 products:
    # But we have sums of products, so we need to be careful.
    #
    # Actually, let's use a simpler approach: just use schoolbook for the 4x4 * 4x2 part.
    # 4x4 * 4x2 schoolbook: 4*4*2 = 32 multiplications
    # That's too many.
    #
    # Let me use a rank 15 decomposition for 4x4 * 4x2.
    #
    # Actually, I recall that 4x4 * 4x2 can be done with rank 15 using a specific construction.
    # Let me just use the rank 77 approach from the current program, which is known to work.
    
    # I'll stick with the rank 77 approach but implement it cleanly.
    
    rank = 77
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Helper functions for indexing
    def get_u_row(block, i1, j1):
        if block == '00': i, j = i1, j1
        elif block == '01': i, j = i1, j1 + 2
        elif block == '10': i, j = i1 + 2, j1
        elif block == '11': i, j = i1 + 2, j1 + 2
        else: raise ValueError
        return i * m + j
        
    def get_v_row(block, j1, k1):
        if block == '00': j, k = j1, k1
        elif block == '01': j, k = j1, k1 + 3
        elif block == '10': j, k = j1 + 2, k1
        elif block == '11': j, k = j1 + 2, k1 + 3
        else: raise ValueError
        return j * p + k
        
    def get_w_row(block, i1, k1):
        if block == '00': i, k = i1, k1
        elif block == '01': i, k = i1, k1 + 3
        elif block == '10': i, k = i1 + 2, k1
        elif block == '11': i, k = i1 + 2, k1 + 3
        else: raise ValueError
        return k * n + i

    col = 0
    for s in range(7):
        alpha, beta, gamma = strassen_2x2[s]
        
        # Part 1: Strassen for 2x2 * 2x2 (cols 0,1 of B block)
        for inner_s in range(7):
            i_alpha, i_beta, i_gamma = strassen_2x2[inner_s]
            
            for ai in range(2):
                for aj in range(2):
                    inner_key = f"{ai}{aj}"
                    inner_coeff = i_alpha.get(inner_key, 0)
                    if inner_coeff != 0:
                        for b, val in alpha.items():
                            U[get_u_row(b, ai, aj), col] += val * inner_coeff
            
            for bj in range(2):
                for bk in range(2):
                    inner_key = f"{bj}{bk}"
                    inner_coeff = i_beta.get(inner_key, 0)
                    if inner_coeff != 0:
                        for b, val in beta.items():
                            V[get_v_row(b, bj, bk), col] += val * inner_coeff
            
            for ci in range(2):
                for ck in range(2):
                    inner_key = f"{ci}{ck}"
                    inner_coeff = i_gamma.get(inner_key, 0)
                    if inner_coeff != 0:
                        for b, val in gamma.items():
                            W[get_w_row(b, ci, ck), col] += val * inner_coeff
            
            col += 1
        
        # Part 2: Schoolbook for 2x2 * 2x1 (col 2 of B block)
        for b, val in alpha.items():
            U[get_u_row(b, 0, 0), col] += val
        for b, val in beta.items():
            V[get_v_row(b, 0, 2), col] += val
        for b, val in gamma.items():
            W[get_w_row(b, 0, 2), col] += val
        col += 1
        
        for b, val in alpha.items():
            U[get_u_row(b, 0, 1), col] += val
        for b, val in beta.items():
            V[get_v_row(b, 1, 2), col] += val
        for b, val in gamma.items():
            W[get_w_row(b, 0, 2), col] += val
        col += 1
        
        for b, val in alpha.items():
            U[get_u_row(b, 1, 0), col] += val
        for b, val in beta.items():
            V[get_v_row(b, 0, 2), col] += val
        for b, val in gamma.items():
            W[get_w_row(b, 1, 2), col] += val
        col += 1
        
        for b, val in alpha.items():
            U[get_u_row(b, 1, 1), col] += val
        for b, val in beta.items():
            V[get_v_row(b, 1, 2), col] += val
        for b, val in gamma.items():
            W[get_w_row(b, 1, 2), col] += val
        col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
