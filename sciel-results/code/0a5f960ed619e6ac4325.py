"""Matrix multiplication 4x4x4 tensor decomposition using recursive Strassen (rank 49)."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    
    Convention:
      C[i,j] = sum_k A[i,k] * B[k,j]
      U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] summed over r gives the tensor.
      Here n=m=p=4.
    """
    t_start = time.time()
    deadline = 120.0  # Safety margin before 150s kill
    
    n = m = p = 4
    rank = 49
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Helper to flatten 2x2 block matrices (indices 0..3) to global indices
    # Block (bi, bj) element (er, ec) -> Global (2*bi+er, 2*bj+ec)
    def block_to_global(bi, bj, er, ec):
        return (2*bi+er) * 4 + (2*bj+ec)

    # Define the 7 Strassen block products M_1...M_7
    # Each M is defined by L (linear combo of A blocks) and R (linear combo of B blocks)
    # A blocks: A00, A01, A10, A11 (2x2 matrices)
    # B blocks: B00, B01, B10, B11 (2x2 matrices)
    
    # We represent each L/R as a list of 4 2x2 matrices (for A00, A01, A10, A11)
    # where each 2x2 matrix contains coefficients for the elements.
    # Actually, it's easier to work with the "symbolic" expansion directly.
    
    # Let's define the 7 outer Strassen terms.
    # Term t corresponds to M_t = L_t * R_t
    # L_t and R_t are 2x2 block matrices where each entry is a linear combination of the 2x2 sub-blocks of A and B.
    
    # For implementation, we will define the 49 scalar products directly.
    # There are 7 outer terms, each expanding to 7 inner Strassen scalar products.
    
    # Inner Strassen expansion for 2x2 * 2x2:
    # Let L = [[l00, l01], [l10, l11]] and R = [[r00, r01], [r10, r11]]
    # s1 = (l00 + l11) * (r00 + r11)
    # s2 = (l10 + l11) * r00
    # s3 = l00 * (r01 - r11)
    # s4 = l11 * (r10 - r00)
    # s5 = (l00 + l01) * r11
    # s6 = (l10 - l00) * (r00 + r01)
    # s7 = (l01 - l11) * (r10 + r11)
    
    # Contribution to P (2x2 result):
    # P00 = s1 + s4 - s5 + s7
    # P01 = s3 + s5
    # P10 = s2 + s4
    # P11 = s1 - s2 + s3 + s6
    
    # We will iterate over the 7 outer terms.
    # For each outer term, we need the L and R block matrices.
    # Each block of L is a 2x2 matrix of coefficients over the elements of A blocks.
    # It is more efficient to pre-compute the mapping of "Outer Term t, Inner Term k" 
    # to (U_coeff_vector, V_coeff_vector, W_coeff_vector).
    
    # Let's define the L and R for each outer term explicitly as dictionaries of block indices.
    # Block index: 0=A00, 1=A01, 2=A10, 3=A11 for L; 0=B00, 1=B01, 2=B10, 3=B11 for R.
    
    # Term 0: M1 = (A00 + A11)(B00 + B11)
    # L blocks: A00:1, A11:1. Others 0.
    # R blocks: B00:1, B11:1. Others 0.
    # C contrib: C00 += M1, C11 += M1
    
    # Term 1: M2 = (A10 + A11)B00
    # L blocks: A10:1, A11:1.
    # R blocks: B00:1.
    # C contrib: C10 += M2, C11 -= M2
    
    # Term 2: M3 = A00(B01 - B11)
    # L blocks: A00:1.
    # R blocks: B01:1, B11:-1.
    # C contrib: C01 += M3, C11 += M3
    
    # Term 3: M4 = A11(B10 - B00)
    # L blocks: A11:1.
    # R blocks: B10:1, B00:-1.
    # C contrib: C00 += M4, C10 += M4
    
    # Term 4: M5 = (A00 + A01)B11
    # L blocks: A00:1, A01:1.
    # R blocks: B11:1.
    # C contrib: C00 -= M5, C01 += M5
    
    # Term 5: M6 = (A10 - A00)(B00 + B01)
    # L blocks: A10:1, A00:-1.
    # R blocks: B00:1, B01:1.
    # C contrib: C11 += M6
    
    # Term 6: M7 = (A01 - A11)(B10 + B11)
    # L blocks: A01:1, A11:-1.
    # R blocks: B10:1, B11:1.
    # C contrib: C00 += M7
    
    # Data structures for the 7 outer terms
    # l_blocks[bi, bj] is a dict {block_id: coeff}
    # r_blocks[bi, bj] is a dict {block_id: coeff}
    # c_contrib is a list of (c_bi, c_bj, coeff) for the 4 blocks of C
    
    outer_terms = []
    
    # Helper to create L/R block dicts
    def make_blocks(coeffs_00, coeffs_01, coeffs_10, coeffs_11):
        # coeffs_XX are dicts {block_id: coeff}
        return [coeffs_00, coeffs_01, coeffs_10, coeffs_11]
        
    # Term 0
    L0 = make_blocks({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1})
    R0 = make_blocks({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1})
    C0 = [(0,0,1), (1,1,1)]
    outer_terms.append((L0, R0, C0))
    
    # Term 1
    L1 = make_blocks({2:1, 3:1}, {2:1, 3:1}, {2:1, 3:1}, {2:1, 3:1})
    R1 = make_blocks({0:1}, {0:1}, {0:1}, {0:1})
    C1 = [(1,0,1), (1,1,-1)]
    outer_terms.append((L1, R1, C1))
    
    # Term 2
    L2 = make_blocks({0:1}, {0:1}, {0:1}, {0:1})
    R2 = make_blocks({1:1, 3:-1}, {1:1, 3:-1}, {1:1, 3:-1}, {1:1, 3:-1})
    C2 = [(0,1,1), (1,1,1)]
    outer_terms.append((L2, R2, C2))
    
    # Term 3
    L3 = make_blocks({3:1}, {3:1}, {3:1}, {3:1})
    R3 = make_blocks({2:1, 0:-1}, {2:1, 0:-1}, {2:1, 0:-1}, {2:1, 0:-1})
    C3 = [(0,0,1), (1,0,1)]
    outer_terms.append((L3, R3, C3))
    
    # Term 4
    L4 = make_blocks({0:1, 1:1}, {0:1, 1:1}, {0:1, 1:1}, {0:1, 1:1})
    R4 = make_blocks({3:1}, {3:1}, {3:1}, {3:1})
    C4 = [(0,0,-1), (0,1,1)]
    outer_terms.append((L4, R4, C4))
    
    # Term 5
    L5 = make_blocks({2:1, 0:-1}, {2:1, 0:-1}, {2:1, 0:-1}, {2:1, 0:-1})
    R5 = make_blocks({0:1, 1:1}, {0:1, 1:1}, {0:1, 1:1}, {0:1, 1:1})
    C5 = [(1,1,1)]
    outer_terms.append((L5, R5, C5))
    
    # Term 6
    L6 = make_blocks({1:1, 3:-1}, {1:1, 3:-1}, {1:1, 3:-1}, {1:1, 3:-1})
    R6 = make_blocks({2:1, 3:1}, {2:1, 3:1}, {2:1, 3:1}, {2:1, 3:1})
    C6 = [(0,0,1)]
    outer_terms.append((L6, R6, C6))
    
    # Inner Strassen coefficients
    # s_idx: (L_coeffs, R_coeffs, P_coeffs)
    # L_coeffs: { (lr, lc): coeff } for the 4 elements of the 2x2 L matrix
    # R_coeffs: { (rr, rc): coeff } for the 4 elements of the 2x2 R matrix
    # P_coeffs: { (pr, pc): coeff } for the 4 elements of the 2x2 P matrix
    
    inner_strassen = [
        # s1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # s2
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # s3
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # s4
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # s5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # s6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # s7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    # Pre-compute global indices for A and B blocks
    # A block (bi, bj) element (er, ec) -> U index
    # B block (bi, bj) element (er, ec) -> V index
    # C block (bi, bj) element (er, ec) -> W index
    
    a_idx = {}
    b_idx = {}
    c_idx = {}
    for bi in range(2):
        for bj in range(2):
            for er in range(2):
                for ec in range(2):
                    gr = 2*bi + er
                    gc = 2*bj + ec
                    a_idx[(bi, bj, er, ec)] = gr * 4 + gc # U index: i*4+k
                    b_idx[(bi, bj, er, ec)] = gr * 4 + gc # V index: k*4+j
                    c_idx[(bi, bj, er, ec)] = gc * 4 + gr # W index: j*4+i
    
    # Build U, V, W
    rank_idx = 0
    for t, (L, R, C_contrib) in enumerate(outer_terms):
        for k, (l_coeffs, r_coeffs, p_coeffs) in enumerate(inner_strassen):
            if time.time() - t_start > deadline:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
            # 1. Calculate U contributions
            # U[row, rank_idx] += sum over (lr, lc) and (bi, bj, er, ec) of 
            #    l_coeffs[(lr,lc)] * L_block[(lr,lc)].get(bi_id, 0) * ...
            
            # L is a list of 4 dicts for A00, A01, A10, A11? 
            # No, L[0] is for A00 block? 
            # In my definition above: L0 = make_blocks({0:1, 3:1}, ...)
            # L[0] corresponds to the block at position (0,0) of the 2x2 block matrix?
            # Wait, the Strassen L is a 2x2 matrix of blocks.
            # L[0,0] is the top-left block of L.
            # In my make_blocks call, the arguments were coeffs_00, coeffs_01, coeffs_10, coeffs_11.
            # So L[0] is the dict for L[0,0]. L[1] is L[0,1]. L[2] is L[1,0]. L[3] is L[1,1].
            
            # Let's iterate over the 4 elements of the 2x2 L matrix (lr, lc)
            for (lr, lc), coeff_l in l_coeffs.items():
                # Get the block dict for L[lr, lc]
                # Index in L list: lr*2 + lc
                block_dict_L = L[lr*2 + lc]
                for block_id_A, coeff_A in block_dict_L.items():
                    # block_id_A: 0=A00, 1=A01, 2=A10, 3=A11
                    bi_A = block_id_A // 2
                    bj_A = block_id_A % 2
                    total_coeff = coeff_l * coeff_A
                    
                    if total_coeff == 0: continue
                    
                    # Add to U for all elements in this block
                    for er in range(2):
                        for ec in range(2):
                            u_i = a_idx[(bi_A, bj_A, er, ec)]
                            U[u_i, rank_idx] += total_coeff
            
            # 2. Calculate V contributions
            for (rr, rc), coeff_r in r_coeffs.items():
                block_dict_R = R[rr*2 + rc]
                for block_id_B, coeff_B in block_dict_R.items():
                    bi_B = block_id_B // 2
                    bj_B = block_id_B % 2
                    total_coeff = coeff_r * coeff_B
                    
                    if total_coeff == 0: continue
                    
                    for er in range(2):
                        for ec in range(2):
                            v_i = b_idx[(bi_B, bj_B, er, ec)]
                            V[v_i, rank_idx] += total_coeff
            
            # 3. Calculate W contributions
            # P[pr, pc] gets coeff p_coeffs[(pr,pc)]
            # C[bi, bj] gets contribution from P based on C_contrib list
            # C_contrib is list of (c_bi, c_bj, c_coeff)
            # The P block at (pr, pc) corresponds to C block (pr, pc) in the standard Strassen output?
            # Yes, the 2x2 block multiplication result P maps directly to C blocks.
            # However, C_contrib tells us which C blocks receive this M_t and with what sign.
            # Actually, C_contrib defines how M_t (which is a 2x2 block matrix) is added to C.
            # M_t[pr, pc] is the block at row pr, col pc of M_t.
            # This block is added to C[c_bi, c_bj] with factor c_coeff.
            # But note: M_t is a 2x2 matrix of 2x2 blocks.
            # So M_t[pr, pc] is a 2x2 matrix.
            # This entire 2x2 matrix is added to C[c_bi, c_bj] (which is also 2x2).
            
            for (pr, pc), coeff_p in p_coeffs.items():
                # This scalar product s_k contributes coeff_p to M_t[pr, pc]
                # Now distribute M_t[pr, pc] to C blocks
                for c_bi, c_bj, c_coeff in C_contrib:
                    total_coeff = coeff_p * c_coeff
                    if total_coeff == 0: continue
                    
                    # M_t[pr, pc] is a 2x2 block of the C matrix?
                    # No, M_t is the result of L_t * R_t.
                    # L_t and R_t are 2x2 block matrices.
                    # So M_t[pr, pc] is a 2x2 matrix (scalar level? No, block level).
                    # Wait, L_t is a 2x2 matrix of 2x2 blocks.
                    # So M_t[pr, pc] is a 2x2 matrix resulting from the block product.
                    # But we are doing "inner Strassen" on the 2x2 blocks.
                    # So s_k is a scalar (linear combo of A) * (linear combo of B).
                    # This scalar contributes to the *elements* of the block M_t[pr, pc].
                    # Specifically, the inner Strassen formula gives P[er, ec] for the 2x2 block.
                    # So s_k contributes coeff_p to P[pr, pc][er, ec]?
                    # No, the inner Strassen decomposes the multiplication of the 2x2 blocks L_block and R_block.
                    # L_block is a 2x2 matrix of scalars (coeffs of A).
                    # R_block is a 2x2 matrix of scalars (coeffs of B).
                    # The product is a 2x2 matrix P_block.
                    # s_k contributes coeff_p to P_block[er, ec].
                    # So we need to add total_coeff to W for C[c_bi, c_bj][er, ec].
                    
                    for er in range(2):
                        for ec in range(2):
                            # The element P_block[er, ec] corresponds to C element (2*c_bi+er, 2*c_bj+ec)
                            w_i = c_idx[(c_bi, c_bj, er, ec)]
                            W[w_i, rank_idx] += total_coeff
            
            rank_idx += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
