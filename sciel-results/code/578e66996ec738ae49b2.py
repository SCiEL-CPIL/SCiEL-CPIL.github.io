import numpy as np

def construct():
    n, m, p = 4, 4, 4
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 decomposition
    # Indices 0..3 for 2x2 blocks
    # A: 00, 01, 10, 11
    # B: 00, 01, 10, 11
    # C: 00, 01, 10, 11
    
    # 7 Strassen terms for 2x2
    # U2[i][s], V2[j][s], W2[k][s]
    # s is 0..6
    
    U2 = [
        [1, 0, 1, 0, 1, -1, 0], # 00
        [0, 0, 0, 0, 1, 0, 1],  # 01
        [0, 1, 0, 0, 0, 1, 0],  # 10
        [1, 1, 0, 1, 0, 0, -1]  # 11
    ]
    
    V2 = [
        [1, 1, 0, -1, 0, 1, 0], # 00
        [0, 0, 1, 0, 0, 1, 0],  # 01
        [0, 0, 0, 1, 0, 0, 1],  # 10
        [1, 0, -1, 0, 1, 0, 1]  # 11
    ]
    
    W2 = [
        [1, 0, 0, 1, -1, 0, 1], # 00
        [0, 0, 1, 0, 1, 0, 0],  # 01
        [0, 1, 0, 1, 0, 0, 0],  # 10
        [1, -1, 1, 0, 0, 1, 0]  # 11
    ]
    
    # The 4x4 matrix multiplication can be decomposed into 49 scalar multiplications
    # by treating the 4x4 matrices as 2x2 blocks of 2x2 matrices.
    # Each block multiplication uses 7 Strassen terms.
    # Total 7*7 = 49 terms.
    
    # Block indices for 2x2 blocks
    # Block (u,v) of A is at rows 2u..2u+1, cols 2v..2v+1
    # Block (u,v) of B is at rows 2u..2u+1, cols 2v..2v+1
    # Block (u,v) of C is at rows 2u..2u+1, cols 2v..2v+1
    
    # For each pair of Strassen terms (k, l) from 0..6:
    # We compute term k of A-blocks times term l of B-blocks.
    # This contributes to C-blocks according to the 2x2 Strassen output structure.
    
    # Let's iterate over the 49 terms
    for k in range(7):
        for l in range(7):
            col = 7 * k + l
            
            # Determine which A-blocks contribute to Strassen term k
            # U2[row][k] gives the coefficient for A-block 'row' in term k
            # 'row' is 0..3 corresponding to (0,0), (0,1), (1,0), (1,1)
            for r in range(4):
                a_coeff = U2[r][k]
                if a_coeff != 0:
                    # r corresponds to block (u, v) where u = r//2, v = r%2
                    u = r // 2
                    v = r % 2
                    # Fill U for all elements in block (u, v)
                    for dr in range(2):
                        for dc in range(2):
                            i = 2 * u + dr
                            j = 2 * v + dc
                            U[i * m + j][col] += a_coeff
                            
            # Determine which B-blocks contribute to Strassen term l
            # V2[col_idx][l] gives the coefficient for B-block 'col_idx' in term l
            # 'col_idx' is 0..3 corresponding to (0,0), (0,1), (1,0), (1,1)
            # Note: V2 is indexed by the "column" block index in the 2x2 sense?
            # In Strassen, V is for B. The indices of B blocks are (j, k) in 2x2 sense.
            # Let's stick to the previous convention: V2[row][s] where row is 0..3 for B blocks.
            # B block (u, v) corresponds to index u*2+v.
            for c in range(4):
                b_coeff = V2[c][l]
                if b_coeff != 0:
                    u = c // 2
                    v = c % 2
                    # Fill V for all elements in block (u, v)
                    for dr in range(2):
                        for dc in range(2):
                            j = 2 * u + dr
                            k_col = 2 * v + dc
                            V[j * p + k_col][col] += b_coeff
                            
            # Determine which C-blocks receive the product of term k and term l
            # The product of Strassen term k of A and term l of B contributes to C-blocks.
            # We need the 2x2 Strassen output coefficients.
            # Let's define G[k][l][c] where c is the C-block index (0..3).
            # This is derived from the Strassen algorithm logic.
            # S1 = A00 * (B00 + B11) -> C00 += S1, C11 += S1
            # S2 = (A10 + A11) * B00 -> C10 += S2, C11 += S2
            # S3 = A00 * (B01 - B11) -> C01 += S3, C11 += S3
            # S4 = A11 * (B10 - B00) -> C00 += S4, C10 += S4
            # S5 = (A00 + A01) * B11 -> C01 += S5, C11 += S5
            # S6 = (A10 - A00) * (B00 + B01) -> C10 += S6, C11 -= S6
            # S7 = (A01 - A11) * (B10 + B11) -> C00 += S7, C10 -= S7
            
            # We need to map (k, l) to contributions in C.
            # This requires knowing which A-blocks are in term k and which B-blocks are in term l,
            # and then applying the Strassen combination rules.
            # Actually, it's easier to use the W2 matrix if we interpret it correctly.
            # W2[c][s] is the coefficient of C-block c in Strassen term s?
            # No, in the 2x2 decomposition, we have 7 scalar products.
            # The output C is a sum of these 7 products, each scaled by W2.
            # Specifically, C_c = sum_s W2[c][s] * (Product_s).
            # Here Product_s is the scalar product of the s-th Strassen term of A and B.
            # So for the 4x4 case, the "s-th Strassen term" is actually a 2x2 matrix product?
            # No, the 49 terms are scalar.
            # Term (k,l) is (Sum_{ab} U2[ab][k] A_ab) * (Sum_{cd} V2[cd][l] B_cd).
            # This product contributes to C blocks.
            # The contribution to C_block(c) is sum_{k,l} (U2-term-k) * (V2-term-l) * Gamma[k][l][c].
            # Gamma is the 2x2 Strassen output coefficient matrix for the pair (k,l).
            
            # Let's compute Gamma[k][l][c] on the fly or precompute.
            # For a given k and l, the A-part involves specific A-blocks, and B-part involves specific B-blocks.
            # The resulting product is a scalar.
            # How does this scalar map to C-blocks?
            # In the 2x2 Strassen, the 7th term is just one scalar product.
            # The 4x4 decomposition uses 49 scalar products.
            # Each scalar product is formed by a linear combination of A-blocks and a linear combination of B-blocks.
            # The resulting scalar is added to specific C-blocks with certain coefficients.
            
            # We can determine the coefficients by expanding the 2x2 Strassen formula.
            # C = [C00, C01; C10, C11]
            # C00 = S1 + S4 + S7
            # C01 = S3 + S5
            # C10 = S2 + S6
            # C11 = S1 + S2 + S3 - S6 + S7
            
            # Here S_i are the 7 scalar products in the 2x2 case.
            # In the 4x4 case, we have 49 products P_{kl}.
            # P_{kl} = (Term_k of A) * (Term_l of B).
            # Term_k of A is a 2x2 matrix? No, Term_k of A is a scalar combination of A-blocks.
            # Wait. In the 2x2 Strassen, A and B are 2x2 matrices.
            # The terms S1..S7 are scalars.
            # In the 4x4 case, A is a 2x2 matrix of 2x2 blocks.
            # We apply Strassen to the 2x2 matrix of blocks.
            # The "elements" of the 2x2 matrix of blocks are 2x2 matrices.
            # So S1 = A00 * (B00 + B11). Here A00 is a 2x2 matrix, B00 is a 2x2 matrix.
            # So S1 is a 4x4 matrix (product of two 2x2 matrices).
            # To decompose S1 into scalar multiplications, we use Strassen again on the 2x2 blocks of A00 and B00?
            # No, we just expand the matrix multiplication A00 * (B00 + B11) into 4x4 scalar multiplications?
            # That would be 16 multiplications for each of the 7 Strassen terms? 7*16 = 112? No.
            
            # The standard recursive Strassen for 4x4:
            # 1. Split A, B into 2x2 blocks (each 2x2).
            # 2. Compute 7 "Strassen terms" which are 2x2 matrix products.
            #    e.g., M1 = A00 * (B00 + B11). This is a 2x2 matrix multiplication.
            # 3. Each 2x2 matrix multiplication is decomposed into 7 scalar multiplications.
            #    So M1 is decomposed into 7 scalar multiplications.
            #    Total 7 * 7 = 49 scalar multiplications.
            
            # So, Term (k,l) corresponds to:
            # k: The Strassen term index for the outer 2x2 block multiplication (0..6).
            # l: The Strassen term index for the inner 2x2 block multiplication (0..6).
            
            # For outer term k:
            # It involves specific outer blocks of A and B.
            # e.g., k=0: A00 * (B00 + B11).
            # The "A-part" is the block A00. The "B-part" is the block (B00 + B11).
            # To compute the product of A00 (2x2) and (B00+B11) (2x2), we use the inner Strassen decomposition.
            # The inner Strassen term l involves specific 1x1 elements of A00 and (B00+B11).
            
            # So, for outer term k and inner term l:
            # The scalar multiplication is:
            # (Linear combo of elements in A00 based on inner term l) * (Linear combo of elements in (B00+B11) based on inner term l)
            
            # We need to map:
            # 1. Outer term k -> Which outer blocks of A and B are involved, and with what signs.
            # 2. Inner term l -> Which elements within those outer blocks are involved, and with what signs.
            
            # Let's define the outer Strassen coefficients.
            # Outer U2[k][block] and Outer V2[k][block].
            # These are the same as U2 and V2 used above.
            
            # Let's define the inner Strassen coefficients.
            # These are also U2 and V2.
            
            # So for col = 7*k + l:
            # 1. Find the outer blocks involved in term k.
            #    For each outer block (u,v) of A with coeff a_outer:
            #       For each outer block (u',v') of B with coeff b_outer:
            #          The product involves A_block(u,v) * B_block(u',v').
            #          We decompose this 2x2 matrix product using inner Strassen term l.
            #          For inner term l:
            #             For each element (r,c) in A_block(u,v) with coeff a_inner:
            #                Global A index: i = 2u+r, j = 2v+c.
            #                U[i,j] += a_outer * a_inner.
            #             For each element (r',c') in B_block(u',v') with coeff b_inner:
            #                Global B index: j = 2u'+r', k = 2v'+c'.
            #                V[j,k] += b_outer * b_inner.
            #          The result contributes to C blocks.
            #          The contribution to C_block(p,q) is determined by the outer Strassen output and the inner Strassen output.
            #          Outer Strassen: Term k contributes to C_block(p,q) with coeff g_outer.
            #          Inner Strassen: Term l contributes to the 2x2 product result at position (r_out, c_out) with coeff g_inner.
            #          Wait, the inner Strassen computes the 2x2 product.
            #          The result of A_block(u,v) * B_block(u',v') is a 2x2 matrix R.
            #          R[rr, cc] = sum_l (Inner_U_l[rr,cc] * Inner_V_l[...])?
            #          No, the inner Strassen decomposition is:
            #          R[rr, cc] = sum_l (Inner_W_l[rr, cc] * (Inner_A_l * Inner_B_l)).
            #          So for a fixed inner term l, the scalar product (Inner_A_l * Inner_B_l) contributes to R[rr, cc] with coeff Inner_W_l[rr, cc].
            #          
            #          Now, the outer Strassen says:
            #          C_block(p,q) += g_outer[k][p,q] * (A_block(u,v) * B_block(u',v')).
            #          So C_block(p,q) += g_outer[k][p,q] * sum_l (Inner_W_l[rr, cc] * Scalar_l).
            #          Thus, for a specific l, the contribution to C_block(p,q) at element (2p+rr, 2q+cc) is:
            #          g_outer[k][p,q] * Inner_W_l[rr, cc] * Scalar_l.
            
            # So, for each col (k,l):
            #   For each outer block pair (u,v) in A and (u',v') in B involved in term k:
            #       a_outer = U2_outer[u*2+v][k]
            #       b_outer = V2_outer[u'*2+v'][k] # Note: V2 indices for B blocks
            #       if a_outer != 0 and b_outer != 0:
            #           For each inner element (r,c) in A_block(u,v) involved in inner term l:
            #               a_inner = U2_inner[r*2+c][l]
            #               if a_inner != 0:
            #                   i = 2u+r, j = 2v+c
            #                   U[i,j] += a_outer * a_inner
            #           For each inner element (r',c') in B_block(u',v') involved in inner term l:
            #               b_inner = V2_inner[r'*2+c'][l]
            #               if b_inner != 0:
            #                   j = 2u'+r', k_idx = 2v'+c'
            #                   V[j,k_idx] += b_outer * b_inner
            #           
            #           # Now fill W
            #           # We need to know which C blocks are affected by outer term k.
            #           # And which positions within those C blocks are affected by inner term l.
            #           
            #           # Outer Strassen output coeffs:
            #           # We need a table Outer_W[k][p,q] where p,q is the C block index (0..3).
            #           # This is derived from the Strassen formula.
            #           # S1 (k=0): C00+=1, C11+=1
            #           # S2 (k=1): C10+=1, C11+=1
            #           # S3 (k=2): C01+=1, C11+=1
            #           # S4 (k=3): C00+=1, C10+=1
            #           # S5 (k=4): C01+=1, C11+=1
            #           # S6 (k=5): C10+=1, C11-=1
            #           # S7 (k=6): C00+=1, C10-=1
            
            #           # Inner Strassen output coeffs:
            #           # Inner_W[l][rr, cc] where rr,cc is 0..1.
            #           # This is the standard 2x2 Strassen W matrix.
            #           # W2[rr*2+cc][l]
            
            #           # For each C block (p,q) affected by outer term k:
            #           #   g_outer = Outer_W[k][p*2+q]
            #           #   if g_outer != 0:
            #           #       For each position (rr, cc) in the 2x2 block:
            #           #           g_inner = W2[rr*2+cc][l]
            #           #           if g_inner != 0:
            #           #               i_row = 2p+rr
            #           #               k_col = 2q+cc
            #           #               w_idx = k_col * n + i_row
            #           #               W[w_idx][col] += g_outer * g_inner
            
            # This logic is correct. I will implement this.

            # Precompute Outer_W (same as W2 but interpreted as block contributions)
            # W2[c][s] where c is 0..3 (C block), s is 0..6 (Strassen term).
            # This is exactly the W2 defined above.
            
            # So for each k, l:
            for r in range(4):
                a_outer = U2[r][k]
                if a_outer == 0: continue
                u = r // 2
                v = r % 2
                for c in range(4):
                    b_outer = V2[c][k] # Wait, V2 indices for B blocks in term k?
                    # In Strassen, term k uses specific A blocks and specific B blocks.
                    # U2[r][k] is coeff for A_block r in term k.
                    # V2[c][k] is coeff for B_block c in term k.
                    # Yes.
                    if b_outer == 0: continue
                    u_prime = c // 2
                    v_prime = c % 2
                    
                    # Fill U
                    for dr in range(2):
                        for dc in range(2):
                            a_inner = U2[dr*2+dc][l]
                            if a_inner != 0:
                                i = 2*u + dr
                                j = 2*v + dc
                                U[i*m + j][col] += a_outer * a_inner
                                
                    # Fill V
                    for dr in range(2):
                        for dc in range(2):
                            b_inner = V2[dr*2+dc][l]
                            if b_inner != 0:
                                j = 2*u_prime + dr
                                k_idx = 2*v_prime + dc
                                V[j*p + k_idx][col] += b_outer * b_inner
                                
                    # Fill W
                    # For each C block (p, q) that term k contributes to:
                    for p in range(2):
                        for q in range(2):
                            g_outer = W2[p*2+q][k]
                            if g_outer == 0: continue
                            for rr in range(2):
                                for cc in range(2):
                                    g_inner = W2[rr*2+cc][l]
                                    if g_inner == 0: continue
                                    i_row = 2*p + rr
                                    k_col = 2*q + cc
                                    w_idx = k_col * n + i_row
                                    W[w_idx][col] += g_outer * g_inner

    return {"U": U, "V": V, "W": W}
