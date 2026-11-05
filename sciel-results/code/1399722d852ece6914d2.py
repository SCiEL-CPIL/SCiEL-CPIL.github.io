import time
import sys

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 53: 49 for 4x4x4 + 4 for 4x4x1
    rank = 53
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val

    col = 0
    
    # Strassen 2x2x2 decomposition
    # Returns list of (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: list of (i, j, coeff) for A
    # v_coeffs: list of (j, k, coeff) for B
    # w_coeffs: list of (k, i, coeff) for C
    # Actually, we need to map to the global indices.
    
    # Strassen terms for 2x2 matrices
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
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
    
    # We need to decompose each M_s (2x2x2) into 7 rank-1 terms.
    # Then sum them up with signs.
    
    # Let's define the 2x2x2 decomposition explicitly.
    # For 2x2x2, indices i,j,k in {0,1}.
    # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r].
    
    # Term 1: (A00+A11)(B00+B11)
    # A part: A00 + A11 -> coeffs: (0,0,1), (1,1,1)
    # B part: B00 + B11 -> coeffs: (0,0,1), (1,1,1)
    # C part: C00 += 1, C01 += 1, C10 += 1, C11 += 1?
    # No, the product is a 2x2 matrix.
    # (A00+A11) is a scalar? No, A00 is a block.
    # In the context of 2x2x2, A00 is a scalar.
    # So M1 is a scalar product of two linear combinations.
    # The result is a scalar? No, M1 is a 2x2 matrix?
    # No, in Strassen's algorithm, A00 etc are blocks.
    # If we are doing 2x2x2, A00 is a scalar.
    # Then M1 is a scalar.
    # But C is 2x2.
    # C00 = M1 + ...
    # So M1 contributes to C00.
    # Wait, if A00 is a scalar, then (A00+A11) is a scalar.
    # Then M1 is a scalar.
    # But C00 is a scalar (entry of 2x2 matrix).
    # So yes, M1 is a scalar that contributes to C00.
    # But what about C01, C10, C11?
    # They don't receive M1 directly?
    # In the 2x2x2 case, the 7 multiplications produce 7 scalars.
    # These 7 scalars are combined to form the 4 entries of C.
    # So for each of the 7 scalars, we have a rank-1 decomposition.
    # The "rank-1 decomposition" of a scalar product $a \cdot b$ contributing to $c$ is:
    # U[i,j] = coeff of A[i,j] in a
    # V[j,k] = coeff of B[j,k] in b
    # W[k,i] = coeff of C[k,i] in c (which is 1 if we add to that entry)
    # But here, one scalar product contributes to multiple C entries?
    # No, in Strassen's 2x2x2, each M_s is a scalar.
    # And each C entry is a linear combination of M_s.
    # So for each M_s, we have one scalar product.
    # But the "tensor" for 2x2x2 has rank 7.
    # This means we can compute the 4 C entries using 7 scalar multiplications.
    # The decomposition is:
    # C_{i,k} = \sum_{r=1}^7 U_{i,j,r} A_{i,j} V_{j,k,r} B_{j,k} W_{k,i,r}
    # For a specific r, U, V, W are fixed vectors.
    # So for each r, we add a term to U, V, W.
    
    # Let's list the 7 terms for 2x2x2.
    # r=1: M1 = (A00+A11)(B00+B11)
    # U: A00 (1), A11 (1)
    # V: B00 (1), B11 (1)
    # W: C00 (1)
    # Wait, does M1 contribute to C00 only?
    # Yes, C00 = M1 + M4 - M5 + M7.
    # So W for r=1 should have 1 at C00.
    # But what about the other C entries?
    # They get contributions from other r's.
    # So for r=1, W[k*2+i] is non-zero only for the C entries that M1 contributes to.
    # In this case, only C00.
    
    # Let's verify the contributions.
    # C00: M1, M4, -M5, M7
    # C01: M3, M5
    # C10: M2, M4
    # C11: M1, -M2, M3, M6
    
    # So:
    # r=1 (M1): U: A00(1), A11(1). V: B00(1), B11(1). W: C00(1), C11(1).
    # r=2 (M2): U: A10(1), A11(1). V: B00(1). W: C10(1), C11(-1).
    # r=3 (M3): U: A00(1). V: B01(1), B11(-1). W: C01(1), C11(1).
    # r=4 (M4): U: A11(1). V: B10(1), B00(-1). W: C00(1), C10(1).
    # r=5 (M5): U: A00(1), A01(1). V: B11(1). W: C00(-1), C01(1).
    # r=6 (M6): U: A10(1), A00(-1). V: B00(1), B01(1). W: C11(1).
    # r=7 (M7): U: A01(1), A11(-1). V: B10(1), B11(1). W: C00(1).
    
    # Now, for 4x4x4, we have 4 blocks of 2x2.
    # A = [[A00, A01], [A10, A11]] (blocks)
    # B = [[B00, B01], [B10, B11]] (blocks)
    # C = [[C00, C01], [C10, C11]] (blocks)
    
    # We apply Strassen's algorithm to the blocks.
    # The "scalars" in the 2x2x2 algorithm are now 2x2 matrices.
    # So each M_s is a 2x2 matrix product.
    # We decompose each M_s using the 2x2x2 rank 7 decomposition.
    # So for each s, we have 7 terms.
    # Total 49 terms.
    
    # For each term, we need to map the local indices to global indices.
    # Local A indices (i_l, j_l) -> Global (i, j)
    # Block (bi, bj) -> i = bi*2 + i_l, j = bj*2 + j_l.
    # Similarly for B and C.
    
    # We need to determine the block indices for each Strassen term.
    # Strassen's algorithm for 4x4:
    # M1 = (A00+A11)(B00+B11)
    # Here A00 is the top-left block (bi=0, bj=0).
    # A11 is the bottom-right block (bi=1, bj=1).
    # B00 is top-left (bj=0, bk=0).
    # B11 is bottom-right (bj=1, bk=1).
    # The product is a 2x2 matrix.
    # We decompose this 2x2x2 product.
    # The local indices i_l, j_l, k_l in the 2x2x2 decomposition map to global indices within the block.
    # But the block itself is at a specific position.
    # For M1, the A part involves blocks (0,0) and (1,1).
    # The B part involves blocks (0,0) and (1,1).
    # The result is added to C blocks (0,0) and (1,1)?
    # No, M1 is a 2x2 matrix.
    # It contributes to C00 and C11.
    # So the local C indices (i_c, k_c) map to global C blocks (0,0) and (1,1)?
    # No, the local C indices are within the 2x2 matrix result.
    # The result of M1 is a 2x2 matrix.
    # This 2x2 matrix is added to C00 (top-left block) and C11 (bottom-right block)?
    # No, in Strassen's algorithm, M1 is a scalar (if blocks are 1x1).
    # If blocks are 2x2, M1 is a 2x2 matrix.
    # And C00 = M1 + M4 - M5 + M7.
    # So the 2x2 matrix M1 is added to the 2x2 matrix C00.
    # So the local indices of the 2x2x2 decomposition of M1 map directly to the local indices of the C00 block.
    # So global i = 0*2 + i_l, global k = 0*2 + k_c.
    # And for C11, global i = 1*2 + i_l, global k = 1*2 + k_c.
    
    # So for each Strassen term s, we need to know which C blocks it contributes to.
    # And with what sign.
    
    # C00: M1(+), M4(+), M5(-), M7(+)
    # C01: M3(+), M5(+)
    # C10: M2(+), M4(+)
    # C11: M1(+), M2(-), M3(+), M6(+)
    
    # So for M1, it contributes to C00 and C11.
    # For the 2x2x2 decomposition of M1, we have 7 terms.
    # For each term, W has non-zero entries for C00 and C11.
    # Specifically, if the local C index is (i_c, k_c), then:
    # For C00 block: global (i, k) = (0*2 + i_c, 0*2 + k_c).
    # For C11 block: global (i, k) = (1*2 + i_c, 1*2 + k_c).
    
    # We need to iterate over the 7 Strassen terms.
    # For each, we iterate over the 7 sub-terms of the 2x2x2 decomposition.
    # For each sub-term, we update U, V, W.
    
    # Let's define the 2x2x2 terms as a list of (U_coeffs, V_coeffs, W_coeffs).
    # U_coeffs: list of (i_l, j_l, coeff)
    # V_coeffs: list of (j_l, k_l, coeff)
    # W_coeffs: list of (i_c, k_c, coeff) -- note: W index is k*n+i, so (k_c, i_c)
    
    # Term 1: M1
    # U: (0,0,1), (1,1,1)
    # V: (0,0,1), (1,1,1)
    # W: (0,0,1), (1,1,1) -- local C indices
    # But W is per C block.
    
    # Let's create a function to add a 2x2x2 decomposition for a specific Strassen term.
    
    def add_strassen_term(s, u_blocks, v_blocks, w_blocks_signs):
        # u_blocks: list of (bi, bj, coeff) for A blocks
        # v_blocks: list of (bj, bk, coeff) for B blocks
        # w_blocks_signs: list of (bi_c, bk_c, sign) for C blocks
        pass

    # This is getting complicated to map.
    # Let's just hardcode the 49 terms.
    
    # Strassen 2x2x2 terms (local indices 0,1)
    # r=1: U: (0,0,1), (1,1,1); V: (0,0,1), (1,1,1); W: (0,0,1), (1,1,1)
    # r=2: U: (1,0,1), (1,1,1); V: (0,0,1); W: (1,0,1), (1,1,-1)
    # r=3: U: (0,0,1); V: (0,1,1), (1,1,-1); W: (0,1,1), (1,1,1)
    # r=4: U: (1,1,1); V: (1,0,1), (0,0,-1); W: (0,0,1), (1,0,1)
    # r=5: U: (0,0,1), (0,1,1); V: (1,1,1); W: (0,0,-1), (0,1,1)
    # r=6: U: (1,0,1), (0,0,-1); V: (0,0,1), (0,1,1); W: (1,1,1)
    # r=7: U: (0,1,1), (1,1,-1); V: (1,0,1), (1,1,1); W: (0,0,1)
    
    # Strassen 4x4x4 terms
    # M1: A blocks (0,0), (1,1); B blocks (0,0), (1,1); C blocks (0,0), (1,1)
    # M2: A blocks (1,0), (1,1); B blocks (0,0); C blocks (1,0), (1,1)
    # M3: A blocks (0,0); B blocks (0,1), (1,1); C blocks (0,1), (1,1)
    # M4: A blocks (1,1); B blocks (1,0), (0,0); C blocks (0,0), (1,0)
    # M5: A blocks (0,0), (0,1); B blocks (1,1); C blocks (0,0), (0,1)
    # M6: A blocks (1,0), (0,0); B blocks (0,0), (0,1); C blocks (1,1)
    # M7: A blocks (0,1), (1,1); B blocks (1,0), (1,1); C blocks (0,0)
    
    # For each M_s, we use the 2x2x2 decomposition.
    # The local indices map to global indices within the blocks.
    
    for s in range(7):
        # Determine u_blocks, v_blocks, w_blocks_signs
        if s == 0: # M1
            u_blocks = [(0,0,1), (1,1,1)]
            v_blocks = [(0,0,1), (1,1,1)]
            w_blocks = [(0,0,1), (1,1,1)]
        elif s == 1: # M2
            u_blocks = [(1,0,1), (1,1,1)]
            v_blocks = [(0,0,1)]
            w_blocks = [(1,0,1), (1,1,-1)]
        elif s == 2: # M3
            u_blocks = [(0,0,1)]
            v_blocks = [(0,1,1), (1,1,-1)]
            w_blocks = [(0,1,1), (1,1,1)]
        elif s == 3: # M4
            u_blocks = [(1,1,1)]
            v_blocks = [(1,0,1), (0,0,-1)]
            w_blocks = [(0,0,1), (1,0,1)]
        elif s == 4: # M5
            u_blocks = [(0,0,1), (0,1,1)]
            v_blocks = [(1,1,1)]
            w_blocks = [(0,0,-1), (0,1,1)]
        elif s == 5: # M6
            u_blocks = [(1,0,1), (0,0,-1)]
            v_blocks = [(0,0,1), (0,1,1)]
            w_blocks = [(1,1,1)]
        elif s == 6: # M7
            u_blocks = [(0,1,1), (1,1,-1)]
            v_blocks = [(1,0,1), (1,1,1)]
            w_blocks = [(0,0,1)]
            
        # For each of the 7 sub-terms
        for r in range(7):
            # Get local coeffs
            if r == 0:
                u_local = [(0,0,1), (1,1,1)]
                v_local = [(0,0,1), (1,1,1)]
                w_local = [(0,0,1), (1,1,1)]
            elif r == 1:
                u_local = [(1,0,1), (1,1,1)]
                v_local = [(0,0,1)]
                w_local = [(1,0,1), (1,1,-1)]
            elif r == 2:
                u_local = [(0,0,1)]
                v_local = [(0,1,1), (1,1,-1)]
                w_local = [(0,1,1), (1,1,1)]
            elif r == 3:
                u_local = [(1,1,1)]
                v_local = [(1,0,1), (0,0,-1)]
                w_local = [(0,0,1), (1,0,1)]
            elif r == 4:
                u_local = [(0,0,1), (0,1,1)]
                v_local = [(1,1,1)]
                w_local = [(0,0,-1), (0,1,1)]
            elif r == 5:
                u_local = [(1,0,1), (0,0,-1)]
                v_local = [(0,0,1), (0,1,1)]
                w_local = [(1,1,1)]
            elif r == 6:
                u_local = [(0,1,1), (1,1,-1)]
                v_local = [(1,0,1), (1,1,1)]
                w_local = [(0,0,1)]
                
            # Combine global and local
            # U global index: (bi*2 + i_l) * 4 + (bj*2 + j_l)
            # V global index: (bj*2 + j_l) * 5 + (bk*2 + k_l)
            # W global index: (bk*2 + k_c) * 4 + (bi_c*2 + i_c)
            
            # We need to sum up the contributions to U, V, W for this column.
            # Since U, V, W are linear, we can just add the coefficients.
            
            # Initialize temporary dictionaries for this column
            u_dict = {}
            v_dict = {}
            w_dict = {}
            
            for (bi, bj, cu) in u_blocks:
                for (i_l, j_l, cl) in u_local:
                    i = bi * 2 + i_l
                    j = bj * 2 + j_l
                    idx = i * 4 + j
                    u_dict[idx] = u_dict.get(idx, 0) + cu * cl
                    
            for (bj, bk, cv) in v_blocks:
                for (j_l, k_l, cl) in v_local:
                    j = bj * 2 + j_l
                    k = bk * 2 + k_l
                    idx = j * 5 + k
                    v_dict[idx] = v_dict.get(idx, 0) + cv * cl
                    
            for (bi_c, bk_c, cw) in w_blocks:
                for (i_c, k_c, cl) in w_local:
                    i = bi_c * 2 + i_c
                    k = bk_c * 2 + k_c
                    idx = k * 4 + i
                    w_dict[idx] = w_dict.get(idx, 0) + cw * cl
            
            # Add to U, V, W
            for idx, val in u_dict.items():
                if val != 0:
                    U[idx][col] += val
            for idx, val in v_dict.items():
                if val != 0:
                    V[idx][col] += val
            for idx, val in w_dict.items():
                if val != 0:
                    W[idx][col] += val
            
            col += 1
            
    # Add 4x4x1 part
    for j in range(4):
        r = 49 + j
        # U: A[i,j] for all i
        for i in range(4):
            U[i * 4 + j][r] = 1
        # V: B[j,4]
        V[j * 5 + 4][r] = 1
        # W: C[4,i] for all i
        for i in range(4):
            W[4 * 4 + i][r] = 1
            
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    print(construct())
