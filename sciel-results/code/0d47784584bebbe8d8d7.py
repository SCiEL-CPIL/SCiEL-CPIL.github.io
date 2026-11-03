import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = 4, 5, 5
    # U shape: (n*m, rank) = (20, rank)
    # V shape: (m*p, rank) = (25, rank)
    # W shape: (p*n, rank) = (20, rank)
    
    # We will accumulate the terms
    terms = []
    
    # Helper to add a term
    # u_idx: (i, j) in A
    # v_idx: (j, k) in B
    # w_idx: (i, k) in C
    # coeff_u, coeff_v, coeff_w: scalar coefficients (usually 1, -1)
    def add_term(i_u, j_u, i_v, j_v, i_w, j_w, cu=1, cv=1, cw=1):
        # Note: The tensor decomposition is sum of u_t \otimes v_t \otimes w_t
        # If we have a linear combination in U, we can split it into multiple terms
        # or keep coefficients in U, V, W.
        # The problem allows integer/half-integer entries.
        # So we can just add to the vectors.
        pass

    # Instead of passing indices, let's build the matrices directly.
    # We will create a list of (U_vec, V_vec, W_vec)
    
    def get_strassen_2x2_terms(a_block, b_block, offset_i, offset_j, offset_k):
        """
        Generates 7 terms for 2x2 multiplication.
        a_block, b_block are 2x2 arrays of "variable indices" or we just return the structure.
        Actually, it's easier to return the list of (U_indices, V_indices, W_indices, coeffs).
        """
        pass

    # Let's hardcode the 49 terms for 4x4x4 using Strassen.
    # This is complex to generate dynamically without errors.
    # Alternative: Use the known rank 49 decomposition.
    
    # I will implement a recursive generator.
    
    def generate_strassen_decomp(n, m, p):
        """
        Returns list of terms for M_{n,m,p} using Strassen if n=m=p=2^k, else schoolbook?
        Here we need specific sub-blocks.
        """
        pass

    # Given the complexity and time limit, I will implement the 85-term decomposition explicitly.
    
    # 1. M_{4,4,4} part (Rank 49)
    # We define the 7 Strassen forms for 2x2.
    # S1 = (A11+A22)(B11+B22)
    # S2 = (A21+A22)B11
    # S3 = A11(B12-B22)
    # S4 = A22(B21-B11)
    # S5 = (A11+A12)B22
    # S6 = (A21-A11)(B11+B12)
    # S7 = (A12-A22)(B21+B22)
    
    # For 4x4, we apply this to 2x2 blocks.
    # The "multiplication" of the linear combinations of blocks is done via another Strassen level.
    
    # Let's create the 49 terms by iterating over the 7 top-level forms and 7 bottom-level forms.
    
    # Top level forms (indices 0..6)
    # Bottom level forms (indices 0..6)
    
    # This approach is prone to index errors.
    # I will use a simpler, verified decomposition for 4x4x4 if possible, or just schoolbook for 4x4x4 (rank 64) and see if 85 is worth the risk.
    # Rank 64 + 16 + 16 + 4 = 100. No improvement.
    # Rank 49 is the target.
    
    # Let's try to implement the 49 terms carefully.
    
    # I will use a standard library or numpy to help? No, must be exact.
    
    # I'll write a function to generate the 49 terms for 4x4x4.
    
    def strassen_2x2_forms():
        # Returns list of (u_coeffs, v_coeffs, w_coeffs) for the 7 multiplications
        # u_coeffs is dict {(i,j): coeff} for A
        # v_coeffs is dict {(i,j): coeff} for B
        # w_coeffs is dict {(i,j): coeff} for C
        # But we need the decomposition of the PRODUCT of these linear forms.
        # So we need to decompose the bilinear form.
        
        # Form 1: (A11+A22)(B11+B22)
        # A part: A11, A22
        # B part: B11, B22
        # C part: C11, C12, C21, C22
        # The product is (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22
        # This is not a single simple tensor. It is a sum of 4 simple tensors if we expand.
        # But Strassen says we can compute the 4 outputs C11..C22 from 7 such products.
        # The "rank" comes from the fact that we compute 7 intermediate products M1..M7.
        # Each Mk is a product of two linear combinations.
        # The decomposition of M_{2,2,2} into rank 7 means:
        # C = sum_{k=1}^7 u_k \otimes v_k \otimes w_k
        # where u_k, v_k are the linear combinations, and w_k is the vector of coefficients for C.
        
        # So for M_{2,2,2}, the 7 terms are:
        # T1: u=(1,0,0,1), v=(1,0,0,1), w=(1, 0, 0, -1) ? No.
        # Let's look up the explicit tensor decomposition for Strassen.
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # The 7 terms are (u, v, w):
        # 1. u = A11+A22, v = B11+B22, w = e11 + e22 - e12 - e21 ? No.
        # w is the vector of coefficients in C.
        # For M1, it contributes to C11 (+1), C22 (+1), C12 (0), C21 (0)?
        # No, the formula is:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # So:
        # M1: w = (1, 0, 0, 1) -> C11, C22
        # M2: w = (0, 0, 1, -1) -> C21, C22
        # M3: w = (0, 1, 0, 1) -> C12, C22
        # M4: w = (1, 0, 1, 0) -> C11, C21
        # M5: w = (0, 1, 0, -1) -> C12, C22 (with -1 for C22? No, C22 has -M2+M3+M6, C12 has M3+M5. So M5 contributes +1 to C12, 0 to C22? Wait. C22 formula doesn't have M5. So w5 = (0, 1, 0, 0)? No.
        
        # Let's re-verify Strassen formulas.
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
        
        # So the w vectors (for C11, C12, C21, C22) are:
        # w1 = (1, 0, 0, 1)
        # w2 = (0, 0, 1, -1)
        # w3 = (0, 1, 0, 1)
        # w4 = (1, 0, 1, 0)
        # w5 = (-1, 1, 0, 0)  # -M5 in C11, +M5 in C12
        # w6 = (0, 0, 0, 1)    # +M6 in C22
        # w7 = (1, 0, 0, 0)    # +M7 in C11
        
        # And u, v vectors (for A11, A12, A21, A22 and B11, B12, B21, B22):
        # u1 = (1, 0, 0, 1), v1 = (1, 0, 0, 1)
        # u2 = (0, 0, 1, 1), v2 = (1, 0, 0, 0)
        # u3 = (1, 0, 0, 0), v3 = (0, 1, 0, -1)
        # u4 = (0, 0, 0, 1), v4 = (0, 0, 1, -1)
        # u5 = (1, 1, 0, 0), v5 = (0, 0, 0, 1)
        # u6 = (0, 0, 1, -1), v6 = (1, 1, 0, 0)
        # u7 = (0, 1, 0, -1), v7 = (0, 0, 1, 1)
        
        return [
            ((1,0,0,1), (1,0,0,1), (1,0,0,1)),
            ((0,0,1,1), (1,0,0,0), (0,0,1,-1)),
            ((1,0,0,0), (0,1,0,-1), (0,1,0,1)),
            ((0,0,0,1), (0,0,1,-1), (1,0,1,0)),
            ((1,1,0,0), (0,0,0,1), (-1,1,0,0)),
            ((0,0,1,-1), (1,1,0,0), (0,0,0,1)),
            ((0,1,0,-1), (0,0,1,1), (1,0,0,0))
        ]

    # Now for 4x4x4, we apply this to the 2x2 blocks.
    # The "A" in the top-level Strassen is a 2x2 matrix of 2x2 blocks.
    # So u1 = A11_block + A22_block.
    # This is a 4x4 matrix.
    # We need to decompose the product (A11_block + A22_block) * (B11_block + B22_block) into simple tensors.
    # We do this by applying the 2x2 Strassen decomposition to the 2x2 blocks of the "A" and "B" parts.
    
    # This is recursive.
    # Rank 49.
    
    # I will implement this recursion.
    
    def get_4x4_strassen_terms():
        terms = []
        top_forms = strassen_2x2_forms()
        
        # Map 4x4 indices to 2x2 block indices
        # Block (bi, bj) covers rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
        # A index (i, j) -> block (i//2, j//2), local (i%2, j%2)
        
        for tu, tv, tw in top_forms:
            # tu is tuple of 4 coeffs for A blocks (A00, A01, A10, A11)
            # tv is tuple of 4 coeffs for B blocks
            # tw is tuple of 4 coeffs for C blocks
            
            # We need to form the linear combination of A blocks.
            # Let A_lin be the 4x4 matrix sum(tu[bi,bj] * A_block(bi,bj))
            # Similarly B_lin.
            # The product A_lin * B_lin contributes to C_lin with coefficients tw.
            
            # We decompose A_lin * B_lin using 2x2 Strassen on the 2x2 structure of A_lin and B_lin?
            # No, A_lin is a 4x4 matrix.
            # We can view A_lin as a 2x2 matrix of 2x2 blocks.
            # The blocks of A_lin are linear combinations of the original A blocks.
            
            # Let's just expand the 49 terms directly.
            # Each top-level term corresponds to a product of two 4x4 matrices.
            # We decompose that 4x4 product using the 7 bottom-level Strassen terms.
            
            # For each bottom-level term (bu, bv, bw):
            # bu is coeffs for A_lin blocks (4 coeffs)
            # bv is coeffs for B_lin blocks (4 coeffs)
            # bw is coeffs for C_lin blocks (4 coeffs)
            
            # The A part of the term is:
            # sum_{bi,bj} bu[bi,bj] * A_lin_block(bi,bj)
            # A_lin_block(bi,bj) = sum_{ai,aj} tu[ai,aj] * A_block(ai,aj)
            # So A_part_block(bi,bj) = sum_{ai,aj} (bu[bi,bj] * tu[ai,aj]) * A_block(ai,aj)
            
            # This results in a 4x4 matrix for A.
            # Similarly for B.
            # And C_part is a 4x4 matrix.
            
            # We add this as one simple tensor.
            
            for bu, bv, bw in top_forms: # Reusing the same forms for the inner level
                # Construct U, V, W vectors of size 16 (4x4)
                u_vec = [0]*16
                v_vec = [0]*16
                w_vec = [0]*16
                
                for bi in range(2):
                    for bj in range(2):
                        # Coeff for A_block(ai,aj) in this term
                        # A_lin_block(bi,bj) uses tu[ai,aj]
                        # Inner term uses bu[bi,bj]
                        for ai in range(2):
                            for aj in range(2):
                                coeff_a = bu[bi*2+bj] * tu[ai*2+aj]
                                if coeff_a != 0:
                                    # Map block (ai,aj) and local (0,0) to global index
                                    # Wait, bu[bi,bj] applies to the WHOLE block A_lin_block(bi,bj).
                                    # A_lin_block(bi,bj) is a 2x2 matrix.
                                    # So we need to add coeff to all 4 elements of A_block(ai,aj)?
                                    # No. A_lin_block(bi,bj) is a 2x2 matrix.
                                    # Its elements are sums of A elements.
                                    # The inner Strassen term bu[bi,bj] multiplies the ENTIRE block A_lin_block(bi,bj).
                                    # So for each element in A_lin_block(bi,bj), we add coeff.
                                    # A_lin_block(bi,bj) element (li, lj) is sum_{ai,aj} tu[ai,aj] * A_block(ai,aj)[li, lj].
                                    # So U[global_idx(li, lj, bi, bj)] += bu[bi,bj] * tu[ai,aj]
                                    
                                    # Global index for A element in block (ai,aj) at local (li, lj)
                                    # Row: 2*ai + li, Col: 2*aj + lj
                                    # But bu[bi,bj] selects the block (bi,bj) of A_lin.
                                    # A_lin_block(bi,bj) is formed from A_blocks.
                                    # The structure is:
                                    # A_lin = [ A_lin_00 A_lin_01; A_lin_10 A_lin_11 ]
                                    # A_lin_bi,bj = sum_{ai,aj} tu[ai,aj] A_ai,aj
                                    
                                    # The inner Strassen term uses A_lin_bi,bj as a 2x2 block.
                                    # So the contribution to U is:
                                    # For each element in A_lin_bi,bj (which is 2x2), add coeff.
                                    # The element (li, lj) of A_lin_bi,bj is sum_{ai,aj} tu[ai,aj] A_ai,aj[li, lj].
                                    
                                    # So for each li, lj in 0..1:
                                    for li in range(2):
                                        for lj in range(2):
                                            # Global row: 2*bi + li ? No.
                                            # A_lin is 4x4.
                                            # A_lin_block(bi,bj) occupies rows 2*bi..2*bi+1, cols 2*bj..2*bj+1.
                                            # So global row = 2*bi + li, global col = 2*bj + lj.
                                            # The value is sum_{ai,aj} tu[ai,aj] A_ai,aj[li, lj].
                                            # So U[ (2*bi+li)*4 + (2*bj+lj) ] += bu[bi*2+bj] * tu[ai*2+aj]
                                            
                                            # Wait, the inner Strassen term bu[bi,bj] multiplies the block A_lin_bi,bj.
                                            # So yes, we add bu[bi,bj] * (value of A_lin_bi,bj at li, lj).
                                            # Value is sum tu[ai,aj] A_ai,aj[li, lj].
                                            # So U[idx] += bu[bi,bj] * tu[ai,aj].
                                            
                                            # This logic is correct.
                                            pass # Implementation in final code

    # This is getting too complex for a quick rewrite.
    # I will stick to the schoolbook rank 100 if I can't guarantee correctness of 85.
    # But 85 is a big jump.
    
    # I will implement the 85 decomposition.
    
    return {"U": U, "V": V, "W": W}
