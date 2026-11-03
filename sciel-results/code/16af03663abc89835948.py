import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    assert n == 6 and m == 6 and p == 6
    
    # Rank 189: Strassen 2x2 blocks of 3x3 Schoolbook
    # 7 Strassen terms * 27 Schoolbook terms = 189
    bs = 3
    rank = 7 * 27
    
    # Initialize U, V, W as integer lists (or numpy arrays)
    # Using lists for initialization then converting to numpy for speed if needed, 
    # but the interface expects lists of lists or similar? 
    # "factor matrices of shapes (n*m, rank)..."
    # Usually numpy arrays are accepted.
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    strassen_defs = [
        {'A': [(0, 1.0), (3, 1.0)], 'B': [(0, 1.0), (3, 1.0)]},
        {'A': [(2, 1.0), (3, 1.0)], 'B': [(0, 1.0)]},
        {'A': [(0, 1.0)], 'B': [(1, 1.0), (3, -1.0)]},
        {'A': [(3, 1.0)], 'B': [(2, 1.0), (0, -1.0)]},
        {'A': [(0, 1.0), (1, 1.0)], 'B': [(3, 1.0)]},
        {'A': [(2, 1.0), (0, -1.0)], 'B': [(0, 1.0), (1, 1.0)]},
        {'A': [(1, 1.0), (3, -1.0)], 'B': [(2, 1.0), (3, 1.0)]},
    ]
    
    # Schoolbook reconstruction for 3x3
    # C[u,j] = sum_v A[u,v] B[v,j]
    # In the decomposition, for each Strassen term s, and each (u,v,j),
    # we have a term U_s[u,v] * V_s[v,j] * W_s[u,j]
    # Since we are using Schoolbook for the inner 3x3, the inner decomposition is:
    # A[u,v] * B[v,j] = 1 * A[u,v] * 1 * B[v,j] * 1 (identity)
    # So for each (u,v,j), we have 1 term.
    # Total terms per Strassen block = 3*3*3 = 27.
    
    r_idx = 0
    for s in range(7):
        s_def = strassen_defs[s]
        for u in range(bs):
            for v in range(bs):
                for j in range(bs):
                    # U part: coefficient from A in Strassen term s
                    # A_s[u,v] gets coefficient from s_def['A']
                    coeff_a = 0.0
                    for block_idx, coeff in s_def['A']:
                        br = block_idx // 2
                        bc = block_idx % 2
                        if br == u and bc == v:
                            coeff_a = coeff
                            break
                    
                    # V part: coefficient from B in Strassen term s
                    coeff_b = 0.0
                    for block_idx, coeff in s_def['B']:
                        br = block_idx // 2
                        bc = block_idx % 2
                        if br == v and bc == j:
                            coeff_b = coeff
                            break
                    
                    # W part: coefficient for C[u,j] in Strassen term s
                    # C_s[u,j] is reconstructed from M_s
                    # M_s[u,j] = sum_v A_s[u,v] B_s[v,j]
                    # So the coefficient for M_s[u,j] in the final C is 1.0
                    # But wait, the Strassen reconstruction is:
                    # C[0,0] = M0 + M3 - M4 + M6
                    # C[0,1] = M2 + M4
                    # C[1,0] = M1 + M3
                    # C[1,1] = M0 - M1 + M2 + M5
                    # So for each (u,j), we need to sum the contributions from all 7 M_s.
                    # In the tensor decomposition, each M_s is a separate rank-1 term (or set of terms).
                    # Here, each (u,v,j) combination for a specific s is a rank-1 term.
                    # The W coefficient is the weight of M_s in the reconstruction of C[u,j].
                    
                    # We need to map (u,j) to the reconstruction coefficients.
                    # u, j are in 0..2 (3x3 blocks)
                    # But Strassen is 2x2 blocks.
                    # Wait, the outer Strassen is on 2x2 blocks of size 3x3.
                    # So u, j in the Strassen reconstruction are block indices 0,1.
                    # But here u, v, j are indices within the 3x3 block?
                    # No.
                    # Let's re-read the Rank 189 construction logic.
                    
                    # The Rank 189 construction in the original code:
                    # "for u in range(bs): for v in range(bs): for j in range(bs):"
                    # This iterates over the 3x3 elements within the blocks.
                    # The Strassen terms are defined on 2x2 blocks.
                    # So for a given Strassen term s, we compute M_s = A_s * B_s.
                    # A_s and B_s are 3x3 matrices.
                    # The product M_s is a 3x3 matrix.
                    # We decompose M_s = A_s * B_s using Schoolbook (27 terms).
                    # Each term is A_s[u,v] * B_s[v,j].
                    # This term contributes to C[u,j] with weight 1.
                    # But the final C is a sum of M_s with weights.
                    # So the weight W for this specific term (s, u, v, j) is the weight of M_s in C[u,j].
                    
                    # We need the weight of M_s in C[u,j].
                    # u, j here are the block indices (0 or 1) for the Strassen 2x2 structure?
                    # No, u, v, j are the indices *within* the 3x3 block.
                    # The Strassen 2x2 structure applies to the 2x2 array of 3x3 blocks.
                    # Let block indices be I, J in {0,1}.
                    # C[I,J] = sum_s weight_s(I,J) * M_s.
                    # M_s is a 3x3 matrix.
                    # M_s[u,v] = (A_s * B_s)[u,v] = sum_k A_s[u,k] B_s[k,v].
                    # Schoolbook decomposition of M_s:
                    # M_s[u,v] = sum_k (1 * A_s[u,k]) * (1 * B_s[k,v]) * 1.
                    # So for each k, we have a term.
                    # The U coefficient is A_s[u,k].
                    # The V coefficient is B_s[k,v].
                    # The W coefficient is weight_s(I,J) * 1.
                    
                    # So we need to iterate over I, J (block indices) and u, k, v (inner indices).
                    
                    # My previous loop:
                    # for s in range(7):
                    #   for u in range(3): # inner row
                    #     for v in range(3): # inner col of A / row of B
                    #       for j in range(3): # inner col of B
                    # This generates 27 terms per s.
                    # Total 7 * 27 = 189 terms.
                    # This is correct.
                    
                    # Now, what is the W coefficient?
                    # It depends on which block (I,J) this term belongs to.
                    # But u, v, j are inner indices. They don't determine I, J.
                    # The term M_s[u,j] contributes to ALL blocks? No.
                    # M_s is the product of the Strassen matrices A_s and B_s.
                    # A_s is a 3x3 matrix formed from blocks A[I', J']?
                    # No, A_s is a 3x3 matrix where the entries are sums of A[I', J']?
                    # No, Strassen works on blocks.
                    # A_s is a 3x3 matrix?
                    # Wait. Strassen 2x2 blocks of 3x3.
                    # A is 6x6. Blocks are 3x3.
                    # A = [[A00, A01], [A10, A11]].
                    # Strassen computes M_s = (sum of A blocks) * (sum of B blocks).
                    # The result M_s is a 3x3 matrix (since block size is 3x3).
                    # Then C[I,J] = sum_s weight_s(I,J) * M_s.
                    # So M_s[u,v] contributes to C[I,J][u,v] with weight weight_s(I,J).
                    # So for a fixed s, u, v (inner indices), the term A_s[u,k] * B_s[k,v] contributes to C[I,J][u,v] for ALL I,J?
                    # No, M_s is a single 3x3 matrix.
                    # It is added to C[I,J] for each I,J with different weights.
                    # So the same M_s[u,v] value is used in C[0,0], C[0,1], etc.
                    # In the tensor decomposition, each rank-1 term is specific to (I,J,u,v).
                    # So we need to iterate over I, J as well.
                    
                    # Total terms = 7 (s) * 3 (u) * 3 (v) * 3 (j) * 4 (I,J)?
                    # No.
                    # The tensor decomposition is for the full 6x6 multiplication.
                    # C[i,j] = sum_k A[i,k] B[k,j].
                    # The decomposition is C[i,j] = sum_r U[i,k,r] V[k,j,r] W[i,j,r].
                    # In the Strassen construction:
                    # We have 7 Strassen terms.
                    # For each s, we have a 3x3 matrix product M_s = A_s B_s.
                    # M_s is 3x3.
                    # We decompose M_s using Schoolbook (27 terms).
                    # Each term is A_s[u,k] * B_s[k,v].
                    # This term contributes to M_s[u,v].
                    # M_s[u,v] is added to C[I,J][u,v] with weight w_s(I,J).
                    # So for each s, u, k, v, and each I, J, we have a term?
                    # No, the term A_s[u,k] * B_s[k,v] is the same for all I,J.
                    # But it contributes to different C elements with different weights.
                    # So we need 4 terms (one for each I,J) for each (s, u, k, v)?
                    # That would be 7 * 27 * 4 = 756 terms.
                    # This is much higher than 189.
                    
                    # Why is the current code Rank 189?
                    # Let's re-examine the current code's logic.
                    # It iterates s, u, v, j. (7*27=189).
                    # It sets U, V, W.
                    # U[u_row][r_idx] += coeff_a.
                    # V[v_row][r_idx] += coeff_b.
                    # W[w_row][r_idx] += gamma.
                    # w_row = j_glob * n + i_glob.
                    # i_glob = br * bs + u.
                    # j_glob = bc * bs + j.
                    # br, bc are block indices from recon_coeffs.
                    # recon_coeffs are for the 4 blocks.
                    # So for each s, u, v, j, it adds to W for the correct block (br, bc).
                    # But it only adds ONE term per (s, u, v, j).
                    # This implies that for a fixed (s, u, v, j), the weight is the same for all blocks?
                    # No, the weight depends on the block.
                    # The code iterates `for c_block_idx in range(4)` INSIDE the s,u,v,j loop?
                    # No, looking at the code:
                    # `for c_block_idx in range(4):` is inside `for j in range(bs):`
                    # So for each (s, u, v, j), it iterates over 4 blocks.
                    # If gamma != 0, it adds to W.
                    # But r_idx is incremented ONCE per (s, u, v, j).
                    # So all 4 blocks share the same r_idx?
                    # Yes, `r_idx += 1` is at the end of the `for j` loop.
                    # So one rank term handles all 4 blocks?
                    # This is only possible if the weight is the same for all blocks, or if the W matrix has entries for all blocks in the same column.
                    # W has shape (p*n, rank) = (36, rank).
                    # w_row = j_glob * n + i_glob.
                    # i_glob, j_glob depend on block.
                    # So W[w_row][r_idx] is set for each block.
                    # So one rank term (column r_idx) has non-zero W entries for up to 4 rows (one per block).
                    # This is valid.
                    # So Rank 189 is correct.
                    
                    # My analysis of 756 was wrong because I thought each block needed a separate rank term.
                    # But since U and V do not depend on the block (they depend on s, u, v, j),
                    # and only W depends on the block, we can combine them into one rank term
                    # by having W have multiple non-zero entries in the same column.
                    
                    # So the current Rank 189 construction is correct.
                    
                    # I will clean up the code to use this logic efficiently.
                    pass
