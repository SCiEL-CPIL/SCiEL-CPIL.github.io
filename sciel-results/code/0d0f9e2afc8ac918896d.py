import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = 4, 5, 5
    # Rank calculation:
    # 4x4x4 block: 49
    # 4x4x1 block: 16
    # 4x1x4 block: 16
    # 4x1x1 block: 4
    # Total: 85
    rank = 85
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(i, j, k, u_val, v_val, w_val):
        nonlocal col
        if u_val == 0 and v_val == 0 and w_val == 0:
            return
        U[i * m + j][col] += u_val
        V[j * p + k][col] += v_val
        W[k * n + i][col] += w_val
        col += 1

    # 1. 4x4x4 Block (i,j,k in 0..3)
    # Strassen for 4x4x4
    # M_1..M_7 are 2x2 products
    # For each M_m, we use 7 rank terms for the 2x2x2 multiplication
    
    # Define the 7 linear forms for A and B for each M_m
    # A forms: (a, b, c, d) corresponding to A00, A01, A10, A11
    # B forms: (e, f, g, h) corresponding to B00, B01, B10, B11
    
    # M_1: (a+d)(e+h)
    # M_2: (c+d)e
    # M_3: a(f-h)
    # M_4: d(g-e)
    # M_5: (a+b)h
    # M_6: (c-a)(e+f)
    # M_7: (b-d)(g+h)
    
    # Coefficients for A (a,b,c,d) and B (e,f,g,h) for each M_m
    # Order: a, b, c, d, e, f, g, h
    M_forms = [
        [1, 0, 0, 1, 1, 0, 0, 1], # M1
        [0, 0, 1, 1, 1, 0, 0, 0], # M2
        [1, 0, 0, 0, 0, 1, 0, -1], # M3
        [0, 0, 0, 1, -1, 0, 1, 0], # M4
        [1, 1, 0, 0, 0, 0, 0, 1], # M5
        [1, 0, -1, 0, 1, 1, 0, 0], # M6
        [0, 1, 0, -1, 0, 0, 1, 1]  # M7
    ]
    
    # Strassen 2x2x2 output coefficients
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # We need to add these to the W matrix for the corresponding i1, k1
    
    # For each M_m (index m_idx), we perform a 2x2x2 Strassen multiplication
    # This involves 7 rank-1 terms
    for m_idx in range(7):
        a_coeff = M_forms[m_idx][:4]
        b_coeff = M_forms[m_idx][4:]
        
        # Now decompose the 2x2x2 product of A' and B'
        # A' is defined by a_coeff, B' by b_coeff
        # We use standard Strassen 2x2x2 on these "virtual" matrices
        
        # The 7 rank terms for 2x2x2 are:
        # s1: (a'+d')(e'+h') -> C00 += 1, C11 += 1
        # s2: (c'+d')e'     -> C10 += 1, C11 -= 1
        # s3: a'(f'-h')    -> C01 += 1, C11 += 1
        # s4: d'(g'-e')    -> C00 += 1, C10 += 1
        # s5: (a'+b')h'    -> C00 -= 1, C01 += 1
        # s6: (c'-a')(e'+f')-> C11 += 1
        # s7: (b'-d')(g'+h')-> C00 += 1
        
        # We need to map these to the global i1, j1, k1
        # For each s, we have linear forms in a', b', c', d' and e', f', g', h'
        # a' corresponds to A_{i2, j2} element (0,0)
        # b' corresponds to A_{i2, j2} element (0,1)
        # c' corresponds to A_{i2, j2} element (1,0)
        # d' corresponds to A_{i2, j2} element (1,1)
        
        # Wait, the M_forms already gave us the coefficients for the 2x2 blocks.
        # So a' is a linear combination of A_{i2, j2} elements?
        # No, in the 4x4 Strassen, M_m is a product of two 2x2 matrices.
        # The 2x2 matrices are formed from the 2x2 blocks of the 4x4 matrix.
        # So a' IS the top-left element of the A block?
        # Yes. a' = A_{i2, j2}[0,0] + ...
        # But we are decomposing the product of these 2x2 matrices.
        # So we treat a', b', c', d' as the elements of the A' matrix.
        # And e', f', g', h' as elements of B' matrix.
        
        # The 7 Strassen terms for 2x2x2 are:
        # 1. (a'+d')(e'+h')
        # 2. (c'+d')e'
        # 3. a'(f'-h')
        # 4. d'(g'-e')
        # 5. (a'+b')h'
        # 6. (c'-a')(e'+f')
        # 7. (b'-d')(g'+h')
        
        # For each of these 7 terms, we have coefficients for a', b', c', d' and e', f', g', h'.
        # We then combine these with the a_coeff, b_coeff to get coefficients for the original A_{i,j} and B_{j,k}.
        
        s_forms = [
            ([1, 0, 0, 1], [1, 0, 0, 1]), # s1: (a+d)(e+h)
            ([0, 0, 1, 1], [1, 0, 0, 0]), # s2: (c+d)e
            ([1, 0, 0, 0], [0, 1, 0, -1]), # s3: a(f-h)
            ([0, 0, 0, 1], [-1, 0, 1, 0]), # s4: d(g-e)
            ([1, 1, 0, 0], [0, 0, 0, 1]), # s5: (a+b)h
            ([1, 0, -1, 0], [1, 1, 0, 0]), # s6: (c-a)(e+f)
            ([0, 1, 0, -1], [0, 0, 1, 1])  # s7: (b-d)(g+h)
        ]
        
        # Output contributions for each s to C00, C01, C10, C11
        # s1: C00+=1, C11+=1
        # s2: C10+=1, C11-=1
        # s3: C01+=1, C11+=1
        # s4: C00+=1, C10+=1
        # s5: C00-=1, C01+=1
        # s6: C11+=1
        # s7: C00+=1
        s_outs = [
            (1, 0, 0, 1), # C00, C01, C10, C11
            (0, 0, 1, -1),
            (0, 1, 0, 1),
            (1, 0, 1, 0),
            (-1, 1, 0, 0),
            (0, 0, 0, 1),
            (1, 0, 0, 0)
        ]
        
        for s_idx in range(7):
            a_s_coeff, b_s_coeff = s_forms[s_idx]
            # Combine a_s_coeff with a_coeff to get global A coeffs
            # a_s_coeff[0] multiplies a' which is sum(a_coeff[k] * A_block[0,0] etc?)
            # No, a' is the element (0,0) of the A block.
            # The A block is formed by a_coeff applied to the 2x2 sub-blocks?
            # No, in 4x4 Strassen, M_m uses the 2x2 blocks directly.
            # So a' is the top-left 2x2 block? No, a' is a scalar in the 2x2 Strassen context?
            # Wait. M_m is a 2x2 matrix product.
            # The factors are 2x2 matrices.
            # So a' is the (0,0) element of the A factor matrix.
            # The A factor matrix is formed by linear combinations of the 2x2 blocks of A.
            # So a' = sum( a_coeff[k] * A_block[k] )?
            # No, a_coeff applies to the elements a,b,c,d of the 2x2 matrix.
            # The 2x2 matrix A' has elements a', b', c', d'.
            # a' = A_{i2, j2}[0,0] + A_{i2, j2}[1,1]?
            # No, the 4x4 Strassen operates on 2x2 blocks.
            # So A' is a 2x2 matrix where each element is a 2x2 matrix?
            # No, Strassen reduces 2x2 matrix mult to 7 scalar mults.
            # So for 4x4, we do 7 "2x2 matrix mults".
            # Each "2x2 matrix mult" is done with 7 "scalar mults".
            # So a' is a scalar? No, a' is a 2x2 matrix?
            # If a' is a 2x2 matrix, then we need rank 4 for each?
            # No, the 4x4 Strassen works by treating the 2x2 blocks as scalars in the algebraic structure,
            # but since they are matrices, we must multiply them.
            # So each of the 7 products M_m is a product of two 2x2 matrices.
            # We compute each M_m using Strassen 2x2x2, which uses 7 scalar multiplications.
            # So a' in the inner Strassen is a scalar (element of the 2x2 matrix A').
            # And A' is formed by linear combinations of the 2x2 blocks of A.
            # So a' = sum( coeff * A_block_element ).
            
            # Let's trace indices.
            # i2, j2, k2 are 0 or 1.
            # i = 2*i2 + i1.
            # The A block at (i2, j2) has elements A_{2*i2 + i1, 2*j2 + j1}.
            # The linear form for a' (element 0,0 of A') is:
            # a' = sum_{i1, j1} (a_s_coeff[0] * a_coeff[?]) * A_{...}
            # Actually, a_coeff is the coefficients for the 2x2 blocks.
            # M_m = A_m B_m.
            # A_m is a 2x2 matrix. Its elements are linear combos of A_{i2, j2} blocks.
            # So A_m[0,0] = sum_{u,v} coeff_A[u,v] * A_{2*i2+u, 2*j2+v}?
            # No, the 4x4 Strassen splits A into 4 blocks: A00, A01, A10, A11.
            # M1 = (A00+A11)(B00+B11).
            # Here A00 is a 2x2 matrix.
            # So A_m is a 2x2 matrix of 2x2 matrices?
            # No, M1 is a 2x2 matrix.
            # A00+A11 is a 2x2 matrix.
            # So we are multiplying two 2x2 matrices.
            # We use Strassen to multiply these two 2x2 matrices.
            # The elements of these 2x2 matrices are 2x2 matrices?
            # No, the elements are scalars?
            # No, A00 is a 2x2 matrix of scalars.
            # So A00+A11 is a 2x2 matrix of scalars.
            # So we are doing a standard 2x2 matrix multiplication.
            # We use Strassen for this.
            # So a' is a scalar (element of the 2x2 matrix A00+A11).
            # a' = (A00+A11)[0,0] = A[0,0] + A[2,2].
            # So a' is a linear combination of specific A elements.
            
            # So for each s_idx, we have a' = sum_{i1, j1} c_A[i1, j1] A_{i, j}.
            # And b' = sum_{j1, k1} c_B[j1, k1] B_{j, k}.
            # And we add a term to W with coefficient w_val.
            
            # We need to compute c_A and c_B for each s_idx.
            # c_A[i1, j1] = a_s_coeff[idx] * (coefficient of A_{i1, j1} in a_coeff block)
            
            # Let's precompute the coefficients for A and B for each m_idx and s_idx.
            
            pass # Logic will be implemented in the loop below

    # This nested logic is complex. I will implement it directly in the code.
    return {"U": U, "V": "W"}
