import time
import random
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Block decomposition:
    # A (4x5): rows [2,2], cols [2,3]
    # B (5x5): rows [2,3], cols [2,3]
    # C (4x5): rows [2,2], cols [2,3]
    
    a_row_splits = [2, 2]
    a_col_splits = [2, 3]
    b_row_splits = [2, 3]
    b_col_splits = [2, 3]
    
    a_row_offsets = [0]
    for s in a_row_splits:
        a_row_offsets.append(a_row_offsets[-1] + s)
    a_col_offsets = [0]
    for s in a_col_splits:
        a_col_offsets.append(a_col_offsets[-1] + s)
    b_row_offsets = [0]
    for s in b_row_splits:
        b_row_offsets.append(b_row_offsets[-1] + s)
    b_col_offsets = [0]
    for s in b_col_splits:
        b_col_offsets.append(b_col_offsets[-1] + s)
    
    # Collect all rank-1 terms as (U_idx, V_idx, W_idx, coeff)
    U_terms = []
    V_terms = []
    W_terms = []
    
    def add_term(u_idx, v_idx, w_idx, coeff):
        if coeff != 0:
            U_terms.append(u_idx)
            V_terms.append(v_idx)
            W_terms.append(w_idx)
            # Store coefficient separately; we'll build matrices with it
            # Actually, let's store as tuples and build later
            pass
    
    # Strassen's algorithm for 2x2x2: 7 terms
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For a 2x2x2 block at (row_off, col_off, col2_off):
    # A indices: rows [row_off, row_off+1], cols [col_off, col_off+1]
    # B indices: rows [col_off, col_off+1], cols [col2_off, col2_off+1]
    # C indices: rows [row_off, row_off+1], cols [col2_off, col2_off+1]
    
    def strassen_2x2x2(row_off, col_off, col2_off):
        """Add 7 rank-1 terms for a 2x2x2 block using Strassen."""
        # A11 = (row_off, col_off), A12 = (row_off, col_off+1)
        # A21 = (row_off+1, col_off), A22 = (row_off+1, col_off+1)
        # B11 = (col_off, col2_off), B12 = (col_off, col2_off+1)
        # B21 = (col_off+1, col2_off), B22 = (col_off+1, col2_off+1)
        
        # Each term is a sum/difference of products
        # We need to express each Mi as a linear combination of rank-1 terms
        
        # M1 = (A11+A22)(B11+B22) = A11*B11 + A11*B22 + A22*B11 + A22*B22
        # M2 = (A21+A22)B11 = A21*B11 + A22*B11
        # M3 = A11(B21-B22) = A11*B21 - A11*B22
        # M4 = A22(B12-B11) = A22*B12 - A22*B11
        # M5 = (A11+A12)B22 = A11*B22 + A12*B22
        # M6 = (A21-A11)(B11+B12) = A21*B11 + A21*B12 - A11*B11 - A11*B12
        # M7 = (A12-A22)(B21+B22) = A12*B21 + A12*B22 - A22*B21 - A22*B22
        
        # Now expand C in terms of Aij*Bkl:
        # C11 = M1+M4-M5+M7
        #   = (A11*B11 + A11*B22 + A22*B11 + A22*B22)
        #   + (A22*B12 - A22*B11)
        #   - (A11*B22 + A12*B22)
        #   + (A12*B21 + A12*B22 - A22*B21 - A22*B22)
        #   = A11*B11 + A11*B22 + A22*B11 + A22*B22 + A22*B12 - A22*B11
        #     - A11*B22 - A12*B22 + A12*B21 + A12*B22 - A22*B21 - A22*B22
        #   = A11*B11 + A22*B12 + A12*B21 - A22*B21
        #   Wait, let me redo this more carefully.
        
        # Actually, the standard approach is to just list the 7 intermediate products
        # and then express C as combinations. But for tensor decomposition, we need
        # to express each C element as a sum of products of linear combinations.
        
        # Let me just directly compute the rank-1 decomposition.
        # The key insight: each Mi is a product of two linear forms.
        # We can write each Mi as a sum of rank-1 terms.
        
        # A11 index: U = row_off*m + col_off, V = col_off*p + col2_off
        # A12 index: U = row_off*m + col_off+1, V = col_off*p + col2_off+1
        # A21 index: U = (row_off+1)*m + col_off, V = col_off+1*p + col2_off
        # A22 index: U = (row_off+1)*m + col_off+1, V = col_off+1*p + col2_off+1
        
        a11_u = row_off * m + col_off
        a12_u = row_off * m + col_off + 1
        a21_u = (row_off + 1) * m + col_off
        a22_u = (row_off + 1) * m + col_off + 1
        
        b11_v = col_off * p + col2_off
        b12_v = col_off * p + col2_off + 1
        b21_v = (col_off + 1) * p + col2_off
        b22_v = (col_off + 1) * p + col2_off + 1
        
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        
        # Let me just directly compute what C elements are:
        # C11 = A11*B11 + A12*B21
        # C12 = A11*B12 + A12*B22
        # C21 = A21*B11 + A22*B21
        # C22 = A21*B12 + A22*B22
        
        # Using Strassen:
        # M1 = (A11+A22)(B11+B22) -> contributes to C11, C22
        # M2 = (A21+A22)B11 -> contributes to C21, C22
        # M3 = A11(B21-B22) -> contributes to C12
        # M4 = A22(B12-B11) -> contributes to C11, C21
        # M5 = (A11+A12)B22 -> contributes to C11, C12
        # M6 = (A21-A11)(B11+B12) -> contributes to C22
        # M7 = (A12-A22)(B21+B22) -> contributes to C11
        
        # Each Mi is a product of two linear combinations.
        # For the tensor decomposition, we can add each Mi as a "virtual" rank-1 term
        # but that's not quite right. We need to expand.
        
        # Actually, the cleanest way: each Mi = (sum of A terms) * (sum of B terms)
        # = sum over all pairs (A_i * B_j)
        # So M1 has 4 rank-1 terms, M2 has 2, M3 has 2, M4 has 2, M5 has 2, M6 has 4, M7 has 4
        # Total: 4+2+2+2+2+4+4 = 20 rank-1 terms, which is worse than schoolbook 8.
        
        # The key is that we DON'T expand. Instead, we keep the linear combinations.
        # In the tensor decomposition framework, each rank-1 term is:
        # U vector (linear combination of A entries) * V vector (linear combination of B entries) * W vector (contribution to C)
        
        # So for M1: U has 1s at a11 and a22, V has 1s at b11 and b22, W has contributions to C11 and C22.
        
        # Let me define the 7 Strassen terms properly.
        
        # Term 1: M1 = (A11+A22)(B11+B22)
        # U: {a11: 1, a22: 1}
        # V: {b11: 1, b22: 1}
        # W: contributes to C11 (+1) and C22 (+1)
        
        # Term 2: M2 = (A21+A22)B11
        # U: {a21: 1, a22: 1}
        # V: {b11: 1}
        # W: contributes to C21 (+1) and C22 (-1)
        
        # Term 3: M3 = A11(B21-B22)
        # U: {a11: 1}
        # V: {b21: 1, b22: -1}
        # W: contributes to C12 (+1)
        
        # Term 4: M4 = A22(B12-B11)
        # U: {a22: 1}
        # V: {b12: 1, b11: -1}
        # W: contributes to C11 (+1) and C21 (+1)
        
        # Term 5: M5 = (A11+A12)B22
        # U: {a11: 1, a12: 1}
        # V: {b22: 1}
        # W: contributes to C11 (-1) and C12 (+1)
        
        # Term 6: M6 = (A21-A11)(B11+B12)
        # U: {a21: 1, a11: -1}
        # V: {b11: 1, b12: 1}
        # W: contributes to C22 (+1)
        
        # Term 7: M7 = (A12-A22)(B21+B22)
        # U: {a12: 1, a22: -1}
        # V: {b21: 1, b22: 1}
        # W: contributes to C11 (+1)
        
        # Let me verify C11:
        # C11 = M1 + M4 - M5 + M7
        # From M1: A11*B11 + A11*B22 + A22*B11 + A22*B22
        # From M4: A22*B12 - A22*B11
        # From -M5: -A11*B22 - A12*B22
        # From M7: A12*B21 + A12*B22 - A22*B21 - A22*B22
        # Sum: A11*B11 + A11*B22 + A22*B11 + A22*B22 + A22*B12 - A22*B11
        #       - A11*B22 - A12*B22 + A12*B21 + A12*B22 - A22*B21 - A22*B22
        # = A11*B11 + A22*B12 + A12*B21 - A22*B21
        # Hmm, that doesn't look right. Let me recheck.
        
        # Actually I think I'm making an error. Let me just use the standard Strassen
        # formulation directly.
        
        # Standard Strassen:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B12 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Let me verify C12 = M3 + M5:
        # M3 = A11*B21 - A11*B22
        # M5 = A11*B22 + A12*B22
        # C12 = A11*B21 + A12*B22 ✓
        
        # C21 = M2 + M4:
        # M2 = A21*B11 + A22*B11
        # M4 = A22*B12 - A22*B11
        # C21 = A21*B11 + A22*B12 ✓
        
        # C22 = M1 - M2 + M3 + M6:
        # M1 = A11*B11 + A11*B22 + A22*B11 + A22*B22
        # -M2 = -A21*B11 - A22*B11
        # M3 = A11*B21 - A11*B22
        # M6 = A21*B11 + A21*B12 - A11*B11 - A11*B12
        # Sum: A11*B11 + A11*B22 + A22*B11 + A22*B22 - A21*B11 - A22*B11
        #      + A11*B21 - A11*B22 + A21*B11 + A21*B12 - A11*B11 - A11*B12
        # = A22*B22 + A11*B21 + A21*B12 - A11*B12
        # Hmm, that's not A21*B12 + A22*B22. Let me recheck M6.
        
        # M6 = (A21 - A11)(B11 + B12) = A21*B11 + A21*B12 - A11*B11 - A11*B12
        # So C22 = M1 - M2 + M3 + M6
        # = (A11*B11 + A11*B22 + A22*B11 + A22*B22)
        #   + (-A21*B11 - A22*B11)
        #   + (A11*B21 - A11*B22)
        #   + (A21*B11 + A21*B12 - A11*B11 - A11*B12)
        # = A11*B11 + A11*B22 + A22*B11 + A22*B22 - A21*B11 - A22*B11
        #   + A11*B21 - A11*B22 + A21*B11 + A21*B12 - A11*B11 - A11*B12
        # = A22*B22 + A11*B21 + A21*B12 - A11*B12
        # That's wrong. I must have the Strassen formulas slightly off.
        
        # Let me look up the correct Strassen formulas:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B12 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Let me re-verify C22 more carefully:
        # M1 = A11B11 + A11B22 + A22B11 + A22B22
        # -M2 = -A21B11 - A22B11
        # M3 = A11B21 - A11B22
        # M6 = A21B11 + A21B12 - A11B11 - A11B12
        # Sum:
        # A11B11: 1 - 1 = 0
        # A11B22: 1 - 1 = 0
        # A22B11: 1 - 1 = 0
        # A22B22: 1
        # A21B11: -1 + 1 = 0
        # A11B21: 1
        # A21B12: 1
        # A11B12: -1
        # So C22 = A22B22 + A11B21 + A21B12 - A11B12
        # But it should be A21B12 + A22B22.
        
        # I think there's an error in my Strassen formulas. Let me use a different source.
        # The correct Strassen is:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B12 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Let me try a different verification. Take A = [[1,2],[3,4]], B = [[5,6],[7,8]]
        # C = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
        
        # A11=1, A12=2, A21=3, A22=4
        # B11=5, B12=6, B21=7, B22=8
        
        # M1 = (1+4)(5+8) = 5*13 = 65
        # M2 = (3+4)*5 = 35
        # M3 = 1*(7-8) = -1
        # M4 = 4*(6-5) = 4
        # M5 = (1+2)*8 = 24
        # M6 = (3-1)*(5+6) = 2*11 = 22
        # M7 = (2-4)*(7+8) = -2*15 = -30
        
        # C11 = 65 + 4 - 24 + (-30) = 15. But should be 19. Wrong.
        
        # Let me try yet another set of Strassen formulas. Maybe I have the indices mixed up.
        # Let me just use schoolbook for 2x2x2 to be safe, and focus on getting the overall
        # structure right. The main improvement will come from the block structure.
        
        # Actually, let me just implement schoolbook for all blocks for now,
        # but structure the code so we can easily swap in Strassen later.
        pass
    
    # For now, use schoolbook for all sub-blocks
    for i in range(len(a_row_splits)):
        for j in range(len(a_col_splits)):
            for k in range(len(b_col_splits)):
                ni = a_row_splits[i]
                mj = a_col_splits[j]
                pk = b_col_splits[k]
                
                for ii in range(ni):
                    for jj in range(mj):
                        for kk in range(pk):
                            gi = a_row_offsets[i] + ii
                            gj = a_col_offsets[j] + jj
                            gk = b_col_offsets[k] + kk
                            
                            U_terms.append(gi * m + gj)
                            V_terms.append(gj * p + gk)
                            W_terms.append(gk * n + gi)
    
    rank = len(U_terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col_idx in range(rank):
        U[U_terms[col_idx]][col_idx] = 1
        V[V_terms[col_idx]][col_idx] = 1
        W[W_terms[col_idx]][col_idx] = 1
    
    # Verify the decomposition with NumPy
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    T_reconstructed = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr)
    
    # The multiplication tensor should have T[(i*m+j), (j*p+k), (k*n+i)] = 1 for all valid i,j,k
    # and 0 elsewhere
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = (i * m + j, j * p + k, k * n + i)
                if abs(T_reconstructed[idx] - 1.0) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback: use pure schoolbook
        terms_fb = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    terms_fb.append(({a: 1}, {b: 1}, {c: 1}))
        
        rank_fb = len(terms_fb)
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        
        for col_idx, (u_dict, v_dict, w_dict) in enumerate(terms_fb):
            for u_idx, coeff in u_dict.items():
                U_fb[u_idx][col_idx] = coeff
            for v_idx, coeff in v_dict.items():
                V_fb[v_idx][col_idx] = coeff
            for w_idx, coeff in w_dict.items():
                W_fb[w_idx][col_idx] = coeff
        
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
