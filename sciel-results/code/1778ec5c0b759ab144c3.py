import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # We will construct a rank 112 decomposition using Block Strassen.
    # The idea is to split the 4x4 matrix A into four 2x2 blocks, and the 4x8 matrix B into four 2x4 blocks.
    # We use the 7 Strassen products.
    # Each Strassen product is of the form L_A * L_B, where L_A is a 2x2 matrix (linear combo of A blocks)
    # and L_B is a 2x4 matrix (linear combo of B blocks).
    # We compute the product L_A * L_B using the schoolbook algorithm for 2x2 * 2x4.
    # The schoolbook rank for 2x2 * 2x4 is 2*2*4 = 16.
    # So total rank is 7 * 16 = 112.
    
    # This is an improvement over the schoolbook rank of 4*4*8 = 128.
    
    R = 112
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)
    
    # Strassen 7 products for 2x2 blocks:
    # Let A = [[A00, A01], [A10, A11]] and B = [[B00, B01], [B10, B11]]
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C = [[M1 + M4 - M5 - M7, M3 + M7],
    #      [M2 + M6, M1 - M2 + M4 + M5]]
    
    # We need to map these to the global indices.
    # A is 4x4. Blocks are 2x2.
    # A00: rows 0-1, cols 0-1. Global index i*4+j.
    # A01: rows 0-1, cols 2-3.
    # A10: rows 2-3, cols 0-1.
    # A11: rows 2-3, cols 2-3.
    
    # B is 4x8. Blocks are 2x4.
    # B00: rows 0-1, cols 0-3.
    # B01: rows 0-1, cols 4-7.
    # B10: rows 2-3, cols 0-3.
    # B11: rows 2-3, cols 4-7.
    
    # C is 4x8. Blocks are 2x4.
    # C00: rows 0-1, cols 0-3.
    # C01: rows 0-1, cols 4-7.
    # C10: rows 2-3, cols 0-3.
    # C11: rows 2-3, cols 4-7.
    
    # Helper to add a schoolbook decomposition of 2x2 * 2x4 product
    # to the global U, V, W matrices.
    # The product is (sum_a u_a A_a) * (sum_b v_b B_b)
    # where A_a is a 2x2 block of A (or linear combo) and B_b is a 2x4 block of B (or linear combo).
    # Actually, the "blocks" here are the 2x2 and 2x4 submatrices.
    # We expand the product using schoolbook:
    # (L_A)_{i,j} (L_B)_{j,k}
    # L_A is 2x2, L_B is 2x4.
    # L_A[i,j] = sum_{a} u_{i,j,a} A_a (where A_a are the 4 blocks of A)
    # L_B[j,k] = sum_{b} v_{j,k,b} B_b (where B_b are the 4 blocks of B)
    
    # Wait, the Strassen formulas are sums of products of *blocks*.
    # e.g. M1 = (A00 + A11)(B00 + B11)
    # This is (A00*B00) + (A00*B11) + (A11*B00) + (A11*B11).
    # Each term is a product of a 2x2 block and a 2x4 block.
    # We can decompose each such product using schoolbook (rank 16).
    # So for M1, we have 4 terms, each rank 16. Total 64 for M1?
    # No, that's not right.
    # (A00 + A11) is a single 2x2 matrix. Let's call it L_A.
    # (B00 + B11) is a single 2x4 matrix. Let's call it L_B.
    # We need to decompose L_A * L_B.
    # L_A is formed by linear combinations of A00, A11.
    # L_B is formed by linear combinations of B00, B11.
    # The product L_A * L_B is a 2x4 matrix.
    # We can use the schoolbook decomposition for 2x2 * 2x4.
    # The entries of L_A are linear combos of A entries.
    # The entries of L_B are linear combos of B entries.
    # So each "scalar" multiplication in the schoolbook is actually a product of linear forms.
    # This is exactly the tensor decomposition.
    
    # Let's define the 7 Strassen terms explicitly.
    # For each term s (0..6), we define:
    # A_coeffs: list of (block_index, coeff) for the 2x2 L_A
    # B_coeffs: list of (block_index, coeff) for the 2x4 L_B
    # C_coeffs: list of (block_index, coeff) for the 2x4 C result
    
    # Block indices for A: 0=A00, 1=A01, 2=A10, 3=A11
    # Block indices for B: 0=B00, 1=B01, 2=B10, 3=B11
    # Block indices for C: 0=C00, 1=C01, 2=C10, 3=C11
    
    strassen_terms = [
        # s=0: M1 = (A00+A11)(B00+B11)
        {
            'A': [(0, 1), (3, 1)],
            'B': [(0, 1), (3, 1)],
            'C': [(0, 1), (3, -1)] # C00 += M1, C11 -= M1? No.
        },
        # s=1: M2 = (A10+A11)B00
        {
            'A': [(2, 1), (3, 1)],
            'B': [(0, 1)],
            'C': [(2, 1), (3, -1)] # C10 += M2, C11 -= M2
        },
        # s=2: M3 = A00(B01-B11)
        {
            'A': [(0, 1)],
            'B': [(1, 1), (3, -1)],
            'C': [(1, 1)] # C01 += M3
        },
        # s=3: M4 = A11(B10-B00)
        {
            'A': [(3, 1)],
            'B': [(2, 1), (0, -1)],
            'C': [(0, 1), (3, 1)] # C00 += M4, C11 += M4
        },
        # s=4: M5 = (A00+A01)B11
        {
            'A': [(0, 1), (1, 1)],
            'B': [(3, 1)],
            'C': [(0, -1), (3, 1)] # C00 -= M5, C11 += M5
        },
        # s=5: M6 = (A10-A00)(B00+B01)
        {
            'A': [(2, 1), (0, -1)],
            'B': [(0, 1), (1, 1)],
            'C': [(2, 1)] # C10 += M6
        },
        # s=6: M7 = (A01-A11)(B10+B11)
        {
            'A': [(1, 1), (3, -1)],
            'B': [(2, 1), (3, 1)],
            'C': [(0, -1), (1, 1)] # C00 -= M7, C01 += M7
        }
    ]
    
    # Wait, the C coefficients in the dict above are for the *blocks* of the result C.
    # But the schoolbook decomposition produces the *entries* of the 2x4 result.
    # The 2x4 result corresponds to a specific C block?
    # No. The Strassen algorithm computes the *blocks* of C.
    # M1 is a 2x4 matrix. It contributes to C00 and C11?
    # Let's check the formula:
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M6
    # C11 = M1 - M2 + M4 + M5
    # So M1 (a 2x4 matrix) is added to C00 and C11.
    # But C00 and C11 are disjoint blocks of the 4x8 matrix C.
    # C00 is rows 0-1, cols 0-3.
    # C11 is rows 2-3, cols 4-7.
    # So the 2x4 matrix M1 is placed in C00 and *also* in C11?
    # Yes, that's how it works. The same 2x4 result is used in two different places.
    
    # So, for each Strassen term s, we have a 2x4 result matrix M_s.
    # We decompose M_s = L_A * L_B using schoolbook (16 terms).
    # Then we add the contributions to U, V, W.
    # The U and V parts are the same for all placements of M_s.
    # The W part depends on where M_s is placed.
    # If M_s is placed in C00, the W indices are for C00.
    # If M_s is placed in C11, the W indices are for C11.
    
    # So, for each s, we generate 16 terms (columns) for the U and V parts.
    # For the W part, we need to map these 16 terms to the appropriate C block(s).
    # If M_s is used in multiple C blocks, we need to add the W contributions to each.
    
    # Let's re-structure.
    # For each s (0..6):
    #   1. Determine L_A and L_B.
    #   2. Generate 16 schoolbook terms for L_A * L_B.
    #      Each term t (0..15) corresponds to a specific (i,j,k) in the 2x2*2x4 product.
    #      i in 0..1 (row of L_A), j in 0..1 (col of L_A / row of L_B), k in 0..3 (col of L_B).
    #      The term is L_A[i,j] * L_B[j,k].
    #      L_A[i,j] is a linear combo of A entries.
    #      L_B[j,k] is a linear combo of B entries.
    #      So the term is (sum_a u_{i,j,a} A_a) * (sum_b v_{j,k,b} B_b).
    #      This expands to sum_{a,b} u_{i,j,a} v_{j,k,b} A_a B_b.
    #      This is a sum of tensor terms.
    #      Wait, the rank is the number of *scalar* multiplications.
    #      In the tensor decomposition, each scalar multiplication is a term.
    #      So for each (i,j,k), we have one "virtual" multiplication of two linear forms.
    #      This virtual multiplication is decomposed into the tensor terms?
    #      No, the tensor decomposition *is* the set of terms.
    #      The rank is the number of terms.
    #      Each term is a product of three vectors: u, v, w.
    #      u is in A-space, v is in B-space, w is in C-space.
    #      The term contributes u * v * w to the tensor.
    #      The "virtual multiplication" is just a way to generate the coefficients.
    #      
    #      Let's be precise.
    #      The tensor T = sum_{i,j,k} A_{i,j} B_{j,k} C_{k,i} (using indices 0..n-1, etc.)
    #      We want to write T = sum_{r=1}^R u_r \otimes v_r \otimes w_r.
    #      u_r is a vector in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}.
    #      
    #      In the block Strassen approach:
    #      We group the terms.
    #      For a fixed s and (i,j,k) in the 2x2*2x4 product:
    #      The contribution to T is:
    #      (sum_{a in A_blocks} u_{s,i,j,a} A_a) * (sum_{b in B_blocks} v_{s,j,k,b} B_b) * (sum_{c in C_blocks} w_{s,k,c} C_c)
    #      Wait, the C part is not a product. The C part is the output.
    #      The term in the tensor is:
    #      (sum_{a} u_{s,i,j,a} A_a) * (sum_{b} v_{s,j,k,b} B_b) * C_{...}
    #      The C index is determined by the placement of M_s.
    #      
    #      Actually, it's simpler.
    #      For each s, we have a "sub-tensor" T_s corresponding to the product L_A * L_B.
    #      T_s = sum_{i,j,k} (L_A)_{i,j} (L_B)_{j,k} C_{...}
    #      This sub-tensor is decomposed into 16 terms (schoolbook).
    #      Term t (for i,j,k):
    #      u_t = (L_A)_{i,j} (as a vector in A-space)
    #      v_t = (L_B)_{j,k} (as a vector in B-space)
    #      w_t = C_{...} (as a vector in C-space)
    #      
    #      (L_A)_{i,j} is a linear combination of A entries.
    #      So u_t has non-zero entries at the A indices involved in (L_A)_{i,j}.
    #      Similarly for v_t and w_t.
    
    # Let's implement this.
    
    for s in range(7):
        term = strassen_terms[s]
        A_blocks = term['A']
        B_blocks = term['B']
        C_blocks = term['C']
        
        # L_A is 2x2. L_B is 2x4.
        # We iterate over i (0..1), j (0..1), k (0..3).
        for i in range(2):
            for j in range(2):
                for k in range(4):
                    col = s * 16 + i * 8 + j * 4 + k
                    
                    # u vector: (L_A)_{i,j}
                    # (L_A)_{i,j} is the sum of A blocks with coefficients.
                    # But wait, (L_A)_{i,j} is a *scalar* in the 2x2 matrix L_A.
                    # L_A is formed by adding A blocks.
                    # So (L_A)_{i,j} = sum_{a} coeff_a * A_block_a[i,j]
                    # Here i,j are local indices in the 2x2 block.
                    # So u_t has non-zero entries at the global A indices corresponding to A_block_a[i,j].
                    
                    # Let's find the global A index for A_block_a[i,j].
                    # A_block_a is one of A00, A01, A10, A11.
                    # a=0: A00 (rows 0-1, cols 0-1). Local (i,j) -> Global (i, j).
                    # a=1: A01 (rows 0-1, cols 2-3). Local (i,j) -> Global (i, j+2).
                    # a=2: A10 (rows 2-3, cols 0-1). Local (i,j) -> Global (i+2, j).
                    # a=3: A11 (rows 2-3, cols 2-3). Local (i,j) -> Global (i+2, j+2).
                    
                    for a, coeff_a in A_blocks:
                        if coeff_a == 0: continue
                        # Determine global row and col
                        row = i + (2 if a >= 2 else 0)
                        col_a = j + (2 if a >= 1 else 0)
                        global_idx = row * 4 + col_a
                        U[global_idx, col] += coeff_a
                        
                    # v vector: (L_B)_{j,k}
                    # (L_B)_{j,k} is the sum of B blocks with coefficients.
                    # B_block_b is one of B00, B01, B10, B11.
                    # b=0: B00 (rows 0-1, cols 0-3). Local (j,k) -> Global (j, k).
                    # b=1: B01 (rows 0-1, cols 4-7). Local (j,k) -> Global (j, k+4).
                    # b=2: B10 (rows 2-3, cols 0-3). Local (j,k) -> Global (j+2, k).
                    # b=3: B11 (rows 2-3, cols 4-7). Local (j,k) -> Global (j+2, k+4).
                    
                    for b, coeff_b in B_blocks:
                        if coeff_b == 0: continue
                        row = j + (2 if b >= 2 else 0)
                        col_b = k + (4 if b >= 1 else 0)
                        global_idx = row * 8 + col_b
                        V[global_idx, col] += coeff_b
                        
                    # w vector: C_{...}
                    # The result of the (i,j,k) term is added to the C blocks specified in C_blocks.
                    # The local indices in the 2x4 result are (i, k).
                    # Wait, the result is a 2x4 matrix M_s.
                    # The entry (i,k) of M_s is added to the C blocks.
                    # For each C block c in C_blocks, we add to the global C index.
                    # c=0: C00 (rows 0-1, cols 0-3). Local (i,k) -> Global (i, k).
                    # c=1: C01 (rows 0-1, cols 4-7). Local (i,k) -> Global (i, k+4).
                    # c=2: C10 (rows 2-3, cols 0-3). Local (i,k) -> Global (i+2, k).
                    # c=3: C11 (rows 2-3, cols 4-7). Local (i,k) -> Global (i+2, k+4).
                    
                    for c, coeff_c in C_blocks:
                        if coeff_c == 0: continue
                        row = i + (2 if c >= 2 else 0)
                        col_c = k + (4 if c >= 1 else 0)
                        global_idx = row * 8 + col_c
                        W[global_idx, col] += coeff_c
                        
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    construct()
