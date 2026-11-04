"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Rank 189 decomposition:
    # Partition 6x6 into 2x2 blocks of 3x3 matrices.
    # Use Strassen (Rank 7) for the 2x2 block multiplication.
    # Use schoolbook (Rank 27) for each 3x3 block multiplication.
    # Total Rank = 7 * 27 = 189.
    
    rank = 189
    
    U = [[0] * rank for _ in range(n * m)]  # (36, 189): U[i*m+k, r] for A[i,k]
    V = [[0] * rank for _ in range(m * p)]  # (36, 189): V[k*p+j, r] for B[k,j]
    W = [[0] * rank for _ in range(p * n)]  # (36, 189): W[j*n+i, r] for C[i,j]
    
    # Strassen 7-term decomposition for 2x2 matrix multiplication
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Strassen terms: (U_offsets, U_coeffs, V_offsets, V_coeffs)
    # Offsets: 0=A11, 1=A12, 2=A21, 3=A22
    strassen_terms = [
        (([0, 3], [1, 1]), ([0, 3], [1, 1])),
        (([2, 3], [1, 1]), ([0], [1])),
        (([0], [1]), ([1, 3], [1, -1])),
        (([3], [1]), ([2, 0], [1, -1])),
        (([0, 1], [1, 1]), ([3], [1])),
        (([2, 0], [1, -1]), ([0, 1], [1, 1])),
        (([1, 3], [1, -1]), ([2, 3], [1, 1])),
    ]
    
    # C output contributions: list of (c_offset, coeff) for each term
    # C offsets: 0=C11, 1=C12, 2=C21, 3=C22
    c_contributions = [
        [(0, 1), (3, 1)],   # M1
        [(2, 1), (3, -1)],  # M2
        [(1, 1), (3, 1)],   # M3
        [(0, 1), (2, 1)],   # M4
        [(0, -1), (1, 1)],  # M5
        [(3, 1)],           # M6
        [(0, 1)],           # M7
    ]
    
    # Schoolbook 27-term decomposition for 3x3 matrix multiplication
    # A = [[a0,a1,a2],[a3,a4,a5],[a6,a7,a8]]
    # B = [[b0,b1,b2],[b3,b4,b5],[b6,b7,b8]]
    # C = A*B
    # C[i,j] = sum_k A[i,k] * B[k,j]
    
    # For each of the 27 products:
    # A[i,k] and B[k,j] contribute to C[i,j]
    
    # We'll iterate over all (i, k, j) triples for the 3x3 blocks
    # Block indices: 0, 1, 2 for rows/cols of 3x3 blocks
    
    col = 0
    # bi: block row of A (0..1, since 2x2 blocks of 3x3)
    # bk: block col of A / block row of B (0..1)
    # bj: block col of B (0..1)
    for bi in range(2):
        for bk in range(2):
            for bj in range(2):
                for t in range(7):
                    u_offsets, u_coeffs = strassen_terms[t][0]
                    v_offsets, v_coeffs = strassen_terms[t][1]
                    
                    # For each Strassen term, we need to decompose the 3x3 block products
                    # The U factor for this term involves a linear combination of A blocks
                    # The V factor involves a linear combination of B blocks
                    # Each block product is a 3x3 * 3x3 = 3x3 multiplication
                    
                    # For the U side: we need to expand the linear combination of A blocks
                    # For the V side: we need to expand the linear combination of B blocks
                    # The product is a bilinear form in the entries of A and B
                    
                    # For each (u_off, u_coeff) and (v_off, v_coeff) pair:
                    # We're computing (sum_u u_coeff * A_block[u_off]) * (sum_v v_coeff * B_block[v_off])
                    # This is a product of two 3x3 matrices (after linear combination)
                    # But actually, Strassen works on the block level, so:
                    # M_t = (linear combo of A blocks) * (linear combo of B blocks)
                    # Each such product is a 3x3 * 3x3 = 3x3 matrix product
                    # Which requires 27 scalar multiplications (schoolbook)
                    
                    # So for each Strassen term t, we have 27 scalar products
                    # For each scalar product (i, k, j) in the 3x3 blocks:
                    # A[i,k] * B[k,j] contributes to C[i,j]
                    
                    # But we need to account for the linear combinations in Strassen
                    # Let's think of it differently:
                    
                    # The Strassen decomposition says:
                    # C_block[bi,bj] = sum_t W_t * M_t
                    # where M_t = U_t * V_t (block product)
                    # and U_t, V_t are linear combinations of A and B blocks
                    
                    # For the scalar level:
                    # Each entry of C_block[bi,bj] is a sum over t of W_t * (entry of M_t)
                    # And each entry of M_t is a bilinear form in A and B entries
                    
                    # Let's just directly construct the rank-189 decomposition:
                    # For each (bi, bk, bj, t, i, k, j) where i,k,j in 0..2:
                    # We have a term that multiplies specific entries of A and B
                    
                    # Actually, let me re-approach this more carefully.
                    
                    # The 2x2 block Strassen gives us 7 block products.
                    # Each block product is a 3x3 * 3x3 matrix multiplication.
                    # Each 3x3 * 3x3 multiplication has rank 27 (schoolbook).
                    # So total rank = 7 * 27 = 189.
                    
                    # For each of the 7 Strassen terms:
                    # M_t = A_t * B_t where A_t and B_t are 3x3 matrices
                    # (linear combinations of the 4 A blocks and 4 B blocks respectively)
                    
                    # The 27 scalar products for A_t * B_t are:
                    # For i in 0..2, k in 0..2, j in 0..2:
                    #   (A_t[i,k]) * (B_t[k,j]) contributes to C_t[i,j]
                    
                    # Where A_t[i,k] = sum_{u_off} u_coeff * A_block[u_off][i,k]
                    # and B_t[k,j] = sum_{v_off} v_coeff * B_block[v_off][k,j]
                    
                    # So the bilinear form is:
                    # (sum_{u_off} u_coeff * A_block[u_off][i,k]) * (sum_{v_off} v_coeff * B_block[v_off][k,j])
                    # = sum_{u_off, v_off} u_coeff * v_coeff * A_block[u_off][i,k] * B_block[v_off][k,j]
                    
                    # This is a sum of products, but we want a rank-1 decomposition.
                    # The issue is that the Strassen terms already have linear combinations.
                    
                    # Let me reconsider. The standard way to compose algorithms:
                    # If we have a rank-r decomposition of 2x2 matrix mult, and a rank-s decomposition of 3x3 matrix mult,
                    # then the composed algorithm has rank r*s for 6x6 matrix mult.
                    
                    # The composition works as follows:
                    # For each rank-1 term in the 2x2 decomposition:
                    #   U_2x2 (a 2x2 matrix of linear forms in A blocks) * V_2x2 (a 2x2 matrix of linear forms in B blocks)
                    # Each entry of U_2x2 * V_2x2 is a product of a linear form in A and a linear form in B.
                    # But we need to decompose each such product into rank-1 terms using the 3x3 decomposition.
                    
                    # Actually, I think I'm overcomplicating this. Let me use a different approach.
                    
                    # The 2x2 Strassen algorithm computes 7 products of 3x3 matrices.
                    # Each 3x3 product can be done with 27 scalar multiplications.
                    # So we have 7 * 27 = 189 scalar multiplications total.
                    
                    # For the rank-189 decomposition, each rank-1 term corresponds to one scalar multiplication.
                    
                    # Let me define the 189 terms explicitly.
                    
                    # For Strassen term t (0..6):
                    #   A_t = sum_{u_off} u_coeffs[u_off] * A_block[u_off]
                    #   B_t = sum_{v_off} v_coeffs[v_off] * B_block[v_off]
                    #   C_t = A_t * B_t (3x3 matrix product)
                    #   C_t[i,j] = sum_k A_t[i,k] * B_t[k,j]
                    
                    # For the rank-1 decomposition, each term is:
                    #   (A_t[i,k]) * (B_t[k,j]) -> contributes to C_t[i,j] with coefficient 1
                    #   And C_t[i,j] contributes to C[bi*3+i, bj*3+j] with coefficient W_t
                    
                    # But A_t[i,k] is a linear combination of A entries, and B_t[k,j] is a linear combination of B entries.
                    # So (A_t[i,k]) * (B_t[k,j]) is not a single product but a product of linear combinations.
                    
                    # To get a rank-1 decomposition, we need to expand this:
                    # A_t[i,k] = sum_{u_off} u_coeff[u_off] * A[u_off_row + i, u_off_col + k]
                    # B_t[k,j] = sum_{v_off} v_coeff[v_off] * B[v_off_row + k, v_off_col + j]
                    
                    # So A_t[i,k] * B_t[k,j] = sum_{u_off, v_off} u_coeff[u_off] * v_coeff[v_off] * A[...] * B[...]
                    
                    # This is a sum of 4*4=16 products (since each linear combination has at most 4 terms).
                    # But each such product is rank-1.
                    
                    # Wait, this doesn't give us rank 189. Let me reconsider.
                    
                    # I think the correct way is:
                    # The 2x2 Strassen algorithm has 7 terms, each of which is a product of two 2x2 "matrices"
                    # where each entry is a 3x3 block.
                    
                    # For the rank-1 decomposition of the whole 6x6 multiplication:
                    # Each of the 7 Strassen terms corresponds to a "meta-term" that is itself a 3x3*3x3 product.
                    # Each 3x3*3x3 product has rank 27.
                    # So we have 7 * 27 = 189 rank-1 terms total.
                    
                    # For each Strassen term t and each (i,k,j) in the 3x3 product:
                    # The U factor is A_t[i,k] (a linear combination of 3x3 block entries)
                    # The V factor is B_t[k,j] (a linear combination of 3x3 block entries)
                    # The W factor is C_t[i,j] * W_t (contribution to the final C)
                    
                    # But A_t[i,k] and B_t[k,j] are linear combinations, so the U and V factors
                    # are themselves linear combinations of multiple A and B entries.
                    
                    # This is still not a single rank-1 term. I'm confused.
                    
                    # Let me just use the 3x3 block partitioning approach from the current program,
                    # which is known to work. The current program uses 3x3 blocks of 2x2 matrices,
                    # with schoolbook for the 3x3 block structure (27 products) and Strassen (rank 7)
                    # for each 2x2 block multiplication. Total rank = 27 * 7 = 189.
                    
                    # This is equivalent to what I was trying to do, just with the block sizes swapped.
                    # Let me just use the working code from the current program.
                    pass
                    
                    col += 27  # 27 terms per Strassen term
    
    # Since the above approach is getting complicated, let me just use the working 3x3 block of 2x2 approach
    # from the current program, which is known to produce a valid rank-189 decomposition.
    
    # Reset and use the proven approach
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_terms = [
        (([0, 3], [1, 1]), ([0, 3], [1, 1])),
        (([2, 3], [1, 1]), ([0], [1])),
        (([0], [1]), ([1, 3], [1, -1])),
        (([3], [1]), ([2, 0], [1, -1])),
        (([0, 1], [1, 1]), ([3], [1])),
        (([2, 0], [1, -1]), ([0, 1], [1, 1])),
        (([1, 3], [1, -1]), ([2, 3], [1, 1])),
    ]
    
    c_contributions = [
        [(0, 1), (3, 1)],   # M1
        [(2, 1), (3, -1)],  # M2
        [(1, 1), (3, 1)],   # M3
        [(0, 1), (2, 1)],   # M4
        [(0, -1), (1, 1)],  # M5
        [(3, 1)],           # M6
        [(0, 1)],           # M7
    ]
    
    col = 0
    for bi in range(3):
        for bk in range(3):
            for bj in range(3):
                for t in range(7):
                    u_offsets, u_coeffs = strassen_terms[t][0]
                    v_offsets, v_coeffs = strassen_terms[t][1]
                    
                    for off, coeff in zip(u_offsets, u_coeffs):
                        r, c = off // 2, off % 2
                        gr = 2 * bi + r
                        gc = 2 * bk + c
                        U[gr * 6 + gc][col] += coeff
                    
                    for off, coeff in zip(v_offsets, v_coeffs):
                        r, c = off // 2, off % 2
                        gr = 2 * bk + r
                        gc = 2 * bj + c
                        V[gr * 6 + gc][col] += coeff
                    
                    for c_off, coeff in c_contributions[t]:
                        r, c = c_off // 2, c_off % 2
                        gr = 2 * bi + r
                        gc = 2 * bj + c
                        W[gc * 6 + gr][col] += coeff
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
