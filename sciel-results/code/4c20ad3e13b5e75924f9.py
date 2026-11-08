import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = 6, 6, 6
    # We'll build the decomposition using Strassen on 2x2 grid of 3x3 blocks
    # 7 Strassen multiplications, each a 3x3 schoolbook (27 mults) = 189 total
    
    # U: (n*m, rank) = (36, rank)
    # V: (m*p, rank) = (36, rank)  
    # W: (p*n, rank) = (36, rank)
    
    # We'll collect terms as (U_row, V_row, W_row, u_val, v_val, w_val)
    terms = []
    
    # Define Strassen linear combinations
    # For 2x2 block Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Block indices: A11 = rows 0-2, cols 0-2; A12 = rows 0-2, cols 3-5
    #               A21 = rows 3-5, cols 0-2; A22 = rows 3-5, cols 3-5
    
    # For each Strassen mult, define:
    # - A_comb: dict mapping (local_row, local_col) -> list of (global_row, global_col, coeff)
    # - B_comb: dict mapping (local_row, local_col) -> list of (global_row, global_col, coeff)
    # - C_contrib: dict mapping (C_block, local_row, local_col) -> coeff
    
    # Actually, let me think of it as: for each of the 7 Strassen mults,
    # and for each of the 27 scalar mults within that 3x3 product,
    # we get a rank-1 term.
    
    # Let me define the 7 Strassen multiplications:
    # Each is: X * Y where X and Y are 3x3 matrices that are linear combinations of A/B blocks.
    
    # I'll represent each linear combination as a 3x3 array of (coeff, block) pairs
    # where block is one of {11, 12, 21, 22}
    
    def block_offset(block):
        """Return (row_offset, col_offset) for a block"""
        if block == 11: return (0, 0)
        if block == 12: return (0, 3)
        if block == 21: return (3, 0)
        if block == 22: return (3, 3)
    
    # Define the 7 Strassen mults
    # Each: (A_terms, B_terms, C_terms)
    # A_terms: 3x3 array, each entry is list of (coeff, block)
    # B_terms: same
    # C_terms: list of (coeff, C_block) where C_block is which C block this contributes to
    #           and the local position within that block
    
    # Actually, for the 3x3 product X*Y, the result is a 3x3 matrix.
    # Each entry of the result contributes to a specific C block.
    
    # For M1 = (A11+A22)(B11+B22):
    #   Result is 3x3, and it contributes to C11 and C22
    #   Specifically: C11 += M1, C22 += M1
    #   Wait no. Let me re-derive.
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # So M1 contributes +1 to C11 and +1 to C22.
    # M2 contributes +1 to C21 and -1 to C22.
    # M3 contributes +1 to C12 and +1 to C22.
    # M4 contributes +1 to C11 and +1 to C21.
    # M5 contributes -1 to C11 and +1 to C12.
    # M6 contributes +1 to C22.
    # M7 contributes +1 to C11.
    
    # For each Mj, the 3x3 result R_j[a,c] (a,c in 0..2) contributes:
    #   For each C_block in the contribution list with coeff s:
    #     C_block[a,c] += s * R_j[a,c]
    
    # And R_j[a,c] = sum_b X_j[a,b] * Y_j[b,c]
    
    # So for each (a,b,c):
    #   X_j[a,b] * Y_j[b,c] contributes to C_block[a,c] with coeff s
    
    # X_j[a,b] is a linear form in A: sum over (coeff_a, block_a) of coeff_a * A_block_a[a,b]
    # Y_j[b,c] is a linear form in B: sum over (coeff_b, block_b) of coeff_b * B_block_b[b,c]
    
    # The product is a bilinear form, which can be decomposed into rank-1 terms.
    # X_j[a,b] * Y_j[b,c] = (sum_{(ca,ba)} ca * A[ba][a,b]) * (sum_{(cb,bb)} cb * B[bb][b,c])
    
    # This is a product of two linear forms, which is a single rank-1 term in the tensor!
    # Because it's (linear in A) * (linear in B) * (contributes to specific C entries).
    
    # Wait, but it contributes to potentially multiple C entries. Let me think...
    
    # Actually, for a fixed (j, a, b, c), the term X_j[a,b] * Y_j[b,c] contributes to:
    #   For each (s, C_block) in C_contrib[j]:
    #     C[C_block][a,c] += s * X_j[a,b] * Y_j[b,c]
    
    # So this single bilinear term contributes to potentially multiple C entries.
    # In the tensor decomposition, this means we need multiple rank-1 terms, one for each C entry it contributes to.
    
    # Hmm, but actually in a rank-1 tensor term, we have U[i,j] * V[j,k] * W[k,i] for specific (i,j,k).
    # If X_j[a,b] * Y_j[b,c] contributes to multiple (i,k) pairs in C, then we need separate rank-1 terms for each.
    
    # Let me just enumerate all terms.
    
    # Define the Strassen multiplications:
    # A_comb[j][a][b] = list of (coeff, block)
    # B_comb[j][b][c] = list of (coeff, block)
    # C_contrib[j] = list of (coeff, C_block)
    
    A_comb = [None] * 7
    B_comb = [None] * 7
    C_contrib = [None] * 7
    
    # M1: X = A11 + A22, Y = B11 + B22
    # X[a,b] = A11[a,b] + A22[a,b]
    # Y[b,c] = B11[b,c] + B22[b,c]
    A_comb[0] = [[[(1, 11), (1, 22)] for _ in range(3)] for _ in range(3)]
    B_comb[0] = [[[(1, 11), (1, 22)] for _ in range(3)] for _ in range(3)]
    C_contrib[0] = [(1, 11), (1, 22)]
    
    # M2: X = A21 + A22, Y = B11
    A_comb[1] = [[[(1, 21), (1, 22)] for _ in range(3)] for _ in range(3)]
    B_comb[1] = [[[(1, 11)] for _ in range(3)] for _ in range(3)]
    C_contrib[1] = [(1, 21), (-1, 22)]
    
    # M3: X = A11, Y = B12 - B22
    A_comb[2] = [[[(1, 11)] for _ in range(3)] for _ in range(3)]
    B_comb[2] = [[[(1, 12), (-1, 22)] for _ in range(3)] for _ in range(3)]
    C_contrib[2] = [(1, 12), (1, 22)]
    
    # M4: X = A22, Y = B21 - B11
    A_comb[3] = [[[(1, 22)] for _ in range(3)] for _ in range(3)]
    B_comb[3] = [[[(-1, 11), (1, 21)] for _ in range(3)] for _ in range(3)]
    C_contrib[3] = [(1, 11), (1, 21)]
    
    # M5: X = A11 + A12, Y = B22
    A_comb[4] = [[[(1, 11), (1, 12)] for _ in range(3)] for _ in range(3)]
    B_comb[4] = [[[(1, 22)] for _ in range(3)] for _ in range(3)]
    C_contrib[4] = [(-1, 11), (1, 12)]
    
    # M6: X = A21 - A11, Y = B11 + B12
    A_comb[5] = [[[(-1, 11), (1, 21)] for _ in range(3)] for _ in range(3)]
    B_comb[5] = [[[(1, 11), (1, 12)] for _ in range(3)] for _ in range(3)]
    C_contrib[5] = [(1, 22)]
    
    # M7: X = A12 - A22, Y = B21 + B22
    A_comb[6] = [[[(-1, 22), (1, 12)] for _ in range(3)] for _ in range(3)]
    B_comb[6] = [[[(1, 21), (1, 22)] for _ in range(3)] for _ in range(3)]
    C_contrib[6] = [(1, 11)]
    
    # Now enumerate all terms
    # For each j (0..6), a (0..2), b (0..2), c (0..2):
    #   X_val = A_comb[j][a][b]  # list of (coeff, block)
    #   Y_val = B_comb[j][b][c]  # list of (coeff, block)
    #   For each (s, C_block) in C_contrib[j]:
    #     The term X*Y contributes s*X*Y to C[C_block][a,c]
    #     X = sum_{(ca,ba)} ca * A[ba_row+a, ba_col+b]  where ba_row,ba_col = offset of block ba
    #     Y = sum_{(cb,bb)} cb * B[bb_row+b, bb_col+c]
    #     This is a bilinear form. To express as rank-1 tensor terms, we need to expand.
    
    # Actually, X*Y = (sum ca * A[...]) * (sum cb * B[...])
    # This is NOT a single rank-1 term in general - it's a sum of products.
    # Wait, no. In the tensor decomposition, a rank-1 term is u_i * v_j * w_k for fixed (i,j,k).
    # The bilinear form (sum ca * A[i1,j1]) * (sum cb * B[j2,k2]) is a sum over (i1,j1,j2,k2) of ca*cb * A[i1,j1]*B[j2,k2].
    # Each such product A[i1,j1]*B[j2,k2] only contributes to C[i1,k2] if j1==j2.
    
    # Hmm, this is getting complicated. Let me think differently.
    
    # The matrix multiplication tensor T has T[i,j,k] = delta_{j,k} (in the sense that C[i,k] = sum_j A[i,j]*B[j,k]).
    # A rank-1 decomposition is: T = sum_r u_r ⊗ v_r ⊗ w_r
    # where (u_r ⊗ v_r ⊗ w_r)[i,j,k] = u_r[i,j] * v_r[j,k] * w_r[k,i]
    
    # For schoolbook: u_r[i,j]=1 if (i,j)=specific pair, v_r[j,k]=1 if (j,k)=specific pair, w_r[k,i]=1 if (k,i)=specific pair.
    
    # For our Strassen-based decomposition, each "multiplication" in Strassen's algorithm is a product of two linear forms in A and B, contributing to linear forms in C.
    
    # Let me just directly construct the U, V, W matrices by expanding everything.
    
    # For each Strassen mult j, and for each (a,b,c) in the 3x3 product:
    #   The local product is: sum_{(ca,ba)} ca * A[ba][a,b] * sum_{(cb,bb)} cb * B[bb][b,c]
    #   = sum_{(ca,ba),(cb,bb)} ca*cb * A[ba][a,b] * B[bb][b,c]
    
    #   A[ba][a,b] is A at global position (ba_row + a, ba_col + b)
    #   B[bb][b,c] is B at global position (bb_row + b, bb_col + c)
    
    #   For this product to contribute to C, we need the column index of A to match the row index of B:
    #   A is at col (ba_col + b), B is at row (bb_row + b). These match iff ba_col == bb_row.
    
    #   If they match, the product contributes to C at global position:
    #   C_row = ba_row + a, C_col = bb_col + c
    #   And this C position is in C_block at local position (a, c) if C_block matches.
    
    #   The contribution coefficient is: s * ca * cb where s is from C_contrib[j].
    
    #   So for each valid combination, we get a rank-1 term:
    #   U[ba_row+a, ba_col+b] += ca  (but we need separate terms for each rank-1 component)
    
    # Actually, I think the cleanest way is:
    # Each rank-1 term in the final decomposition corresponds to a specific (A_entry, B_entry, C_entry) triple.
    # Let me just enumerate all such triples with their coefficients.
    
    terms_dict = {}  # (a_idx, b_idx, c_idx) -> coeff
    # a_idx = i*6+j for A[i,j]
    # b_idx = j*6+k for B[j,k]  
    # c_idx = k*6+i for C[i,k] (trace convention)
    
    for j in range(7):
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    X_terms = A_comb[j][a][b]  # list of (coeff, block)
                    Y_terms = B_comb[j][b][c]  # list of (coeff, block)
                    for (ca, ba) in X_terms:
                        for (cb, bb) in Y_terms:
                            ba_row, ba_col = block_offset(ba)
                            bb_row, bb_col = block_offset(bb)
                            # A global position
                            A_i = ba_row + a
                            A_j = ba_col + b
                            # B global position
                            B_j = bb_row + b
                            B_k = bb_col + c
                            # Check if A_j == B_j (required for matrix mult)
                            if A_j != B_j:
                                continue
                            # C global position
                            C_i = A_i
                            C_k = B_k
                            # Check which C block this falls in
                            # C_block: rows 0-2 are top, 3-5 are bottom; cols 0-2 left, 3-5 right
                            if C_i < 3 and C_k < 3:
                                C_block = 11
                                C_local_a = C_i
                                C_local_c = C_k
                            elif C_i < 3 and C_k >= 3:
                                C_block = 12
                                C_local_a = C_i
                                C_local_c = C_k - 3
                            elif C_i >= 3 and C_k < 3:
                                C_block = 21
                                C_local_a = C_i - 3
                                C_local_c = C_k
                            else:
                                C_block = 22
                                C_local_a = C_i - 3
                                C_local_c = C_k - 3
                            
                            # Check if this C_block is in C_contrib[j]
                            for (s, contrib_block) in C_contrib[j]:
                                if contrib_block == C_block:
                                    # Check that local position matches (a, c)
                                    if C_local_a == a and C_local_c == c:
                                        coeff = s * ca * cb
                                        a_idx = A_i * 6 + A_j
                                        b_idx = B_j * 6 + B_k
                                        c_idx = C_k * 6 + C_i  # trace convention: W[k*n+i]
                                        key = (a_idx, b_idx, c_idx)
                                        terms_dict[key] = terms_dict.get(key, 0) + coeff
    
    # Now build U, V, W from terms_dict
    # Each term (a_idx, b_idx, c_idx) with coeff becomes a rank-1 term
    # But coeff might not be ±1, so we need to handle that.
    # U[a_idx, r] = coeff_a, V[b_idx, r] = coeff_b, W[c_idx, r] = coeff_c
    # where coeff_a * coeff_b * coeff_c = coeff
    # Simplest: U[a_idx, r] = coeff, V[b_idx, r] = 1, W[c_idx, r] = 1
    # Or: U=1, V=1, W=coeff
    # Or distribute: U=sqrt(coeff), etc. But we need integer/half-integer.
    
    # Let's use: U[a_idx, r] = coeff, V[b_idx, r] = 1, W[c_idx, r] = 1
    # This works as long as coeff is integer or half-integer.
    
    rank = len(terms_dict)
    U = [[0] * rank for _ in range(36)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    
    for r, ((a_idx, b_idx, c_idx), coeff) in enumerate(terms_dict.items()):
        U[a_idx][r] = coeff
        V[b_idx][r] = 1
        W[c_idx][r] = 1
    
    return {"U": U, "V": V, "W": W}
