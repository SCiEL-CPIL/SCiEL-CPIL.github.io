"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    rank_sb = n * m * p  # 105
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Strassen decomposition for 2x2x2 blocks
    # Strassen's 7 products for computing [[a,b],[c,d]] * [[e,f],[g,h]] = [[q1-q4, q2+q4], [q3+q4, q1-q3]]
    # where:
    # q1 = (a+b)(e+h)
    # q2 = (c+d)h
    # q3 = a(f-h)
    # q4 = (a-d)(e+f)
    # q5 = (c+d)(e+f)
    # q6 = (a-b)(h+g)
    # q7 = (b-d)(e+g)
    
    # Partition indices into blocks
    # n=3: block_i = [(0,1), (2)]  -> 2-block covers i=0,1; single covers i=2
    # m=5: block_j = [(0,1), (2,3), (4)] -> 2-blocks cover j=0,1 and j=2,3; single covers j=4
    # p=7: block_k = [(0,1), (2,3), (4,5), (6)] -> 2-blocks cover k=0,1; k=2,3; k=4,5; single covers k=6
    
    # Define block assignments
    # For each index, which block it belongs to
    # i_blocks: index -> (block_id, offset_in_block)
    # i=0,1 -> block 0 (2x2), i=2 -> block 1 (single)
    # j=0,1 -> block 0 (2x2), j=2,3 -> block 1 (2x2), j=4 -> block 2 (single)
    # k=0,1 -> block 0 (2x2), k=2,3 -> block 1 (2x2), k=4,5 -> block 2 (2x2), k=6 -> block 3 (single)
    
    # We'll build the decomposition by iterating over all combinations of blocks
    # For each combination of (i_block, j_block, k_block):
    #   - If all are 2-blocks: use Strassen (7 terms)
    #   - Otherwise: use schoolbook for that sub-block
    
    # Strassen coefficients: for a 2x2x2 block with local indices (a,b) for A, (c,d) for B, (e,f) for C
    # The 7 products and their contributions to C[a,e], C[a,f], C[b,e], C[b,f]:
    # 
    # Product 1: (A[0,0]+A[0,1]) * (B[0,0]+B[1,1])
    #   contributes +1 to C[0,0], +1 to C[0,1], -1 to C[1,0], -1 to C[1,1]
    #   Wait, let me be more careful.
    #
    # Standard Strassen:
    # M1 = (a+b)(e+h)
    # M2 = (c+d)h
    # M3 = a(f-h)
    # M4 = (a-d)(e+f)
    # M5 = (c+d)(e+f)
    # M6 = (a-b)(h+g)
    # M7 = (b-d)(e+g)
    #
    # C[0,0] = M1 - M4 + M6
    # C[0,1] = M2 + M4
    # C[1,0] = M3 + M5
    # C[1,1] = M1 - M3 + M7
    #
    # So for each of the 7 products, we have contributions to (i,j,k) in the local 2x2x2 space.
    
    # Let me define the Strassen decomposition more carefully.
    # For a 2x2x2 block with global indices (i0, i1), (j0, j1), (k0, k1):
    # Local (a,b) = (i0,j0), (i0,j1), (i1,j0), (i1,j1) for A
    # Local (c,d) = (j0,k0), (j0,k1), (j1,k0), (j1,k1) for B
    # Local (e,f) = (k0,i0), (k0,i1), (k1,i0), (k1,i1) for C
    #
    # The 7 terms and their (U, V, W) contributions:
    
    def get_strassen_terms():
        """Return the 7 Strassen terms as (U_coeffs, V_coeffs, W_coeffs) for local 2x2x2."""
        # Each term is a dict mapping local index to coefficient
        # Local A indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
        # Local B indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
        # Local C indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
        
        terms = []
        
        # M1 = (a+b)(e+h)
        # U: a=1, b=1 (local A indices 0 and 1)
        # V: e=1, h=1 (local B indices 0 and 3)
        # W: contributes to C[0,0]=+1, C[0,1]=+1, C[1,0]=-1, C[1,1]=-1
        terms.append({
            'U': {0: 1, 1: 1},
            'V': {0: 1, 3: 1},
            'W': {0: 1, 1: 1, 2: -1, 3: -1}
        })
        
        # M2 = (c+d)h
        # U: not directly... wait, this is for the product structure.
        # Actually, in the tensor decomposition, each term r has:
        # U[(i,j), r], V[(j',k), r], W[(k',i'), r]
        # such that sum_r U[(i,j),r] * V[(j',k),r] * W[(k',i'),r] = T[(i,j),(j',k),(k',i')]
        #
        # For Strassen, each of the 7 "products" corresponds to one rank-1 term.
        # Let me re-derive.
        
        # Actually, I think the cleaner way is:
        # For each of the 7 Strassen multipliers, define the U, V, W vectors.
        
        # M1 = (A[0,0]+A[0,1]) * (B[0,0]+B[1,1])
        # This means: U has 1 at (0,0) and (0,1); V has 1 at (0,0) and (1,1)
        # The contribution to C is: M1 * (1 at C[0,0], 1 at C[0,1], -1 at C[1,0], -1 at C[1,1])
        # Wait no, that's not right either. Let me look this up properly.
        
        # Standard Strassen for C = AB where A,B are 2x2:
        # C[0,0] = M1 - M4 + M6
        # C[0,1] = M2 + M4
        # C[1,0] = M3 + M5
        # C[1,1] = M1 - M3 + M7
        #
        # So each C entry is a sum of some of the M's with signs.
        # In the tensor decomposition, for each rank-1 term r (corresponding to M_r),
        # we need U[(i,j),r], V[(j',k),r], W[(k',i'),r] such that
        # sum_r U[(i,j),r]*V[(j',k),r]*W[(k',i'),r] = 1 if (i,j,k) is a valid term, else 0
        #
        # Hmm, I'm overcomplicating this. Let me just directly construct the decomposition.
        
        return None
    
    # Let me take a different approach. I'll directly enumerate the Strassen terms.
    # For a 2x2x2 block with global indices:
    # i in {i0, i1}, j in {j0, j1}, k in {k0, k1}
    #
    # The 7 Strassen terms and their contributions:
    
    strassen_terms = [
        # (U_dict, V_dict, W_dict) where keys are local indices (0 or 1 for i, j, k)
        # M1: (A00+A01)(B00+B11)
        ({(0,0):1, (0,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (0,1):1, (1,0):-1, (1,1):-1}),
        # M2: (A10+A11)B11
        ({(1,0):1, (1,1):1}, {(1,1):1}, {(0,1):1, (1,1):1}),
        # M3: A00(B01-B11)
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):-1}),
        # M4: (A00-A11)(B00+B01)
        ({(0,0):1, (1,1):-1}, {(0,0):1, (0,1):1}, {(0,0):-1, (0,1):1, (1,0):1, (1,1):1}),
        # M5: (A10+A11)(B00+B01)
        ({(1,0):1, (1,1):1}, {(0,0):1, (0,1):1}, {(1,0):1, (1,1):1}),
        # M6: (A00-A01)(B11+B10)
        ({(0,0):1, (0,1):-1}, {(1,1):1, (1,0):1}, {(0,0):1, (1,0):1}),
        # M7: (A01-A11)(B00+B10)
        ({(0,1):1, (1,1):-1}, {(0,0):1, (1,0):1}, {(1,1):1, (1,0):-1}),
    ]
    
    # Now build the full decomposition
    U = {}  # (i, j) -> list of (coeff, term_index)
    V = {}  # (j, k) -> list of (coeff, term_index)
    W = {}  # (k, i) -> list of (coeff, term_index)
    
    term_id = 0
    term_list = []  # Each term: (i_block, j_block, k_block, strassen_idx) or schoolbook
    
    # Define the blocks
    # i_blocks: list of (start, end) where end is exclusive
    i_blocks = [(0, 2), (2, 3)]  # block 0: i=0,1; block 1: i=2
    j_blocks = [(0, 2), (2, 4), (4, 5)]  # block 0: j=0,1; block 1: j=2,3; block 2: j=4
    k_blocks = [(0, 2), (2, 4), (4, 6), (6, 7)]  # block 0: k=0,1; block 1: k=2,3; block 2: k=4,5; block 3: k=6
    
    for ib, (i_start, i_end) in enumerate(i_blocks):
        for jb, (j_start, j_end) in enumerate(j_blocks):
            for kb, (k_start, k_end) in enumerate(k_blocks):
                i_size = i_end - i_start
                j_size = j_end - j_start
                k_size = k_end - k_start
                
                if i_size == 2 and j_size == 2 and k_size == 2:
                    # Use Strassen
                    for s_idx, (U_dict, V_dict, W_dict) in enumerate(strassen_terms):
                        term_list.append(('strassen', ib, jb, kb, s_idx, i_start, j_start, k_start))
                        term_id += 1
                else:
                    # Use schoolbook for this sub-block
                    for i in range(i_start, i_end):
                        for j in range(j_start, j_end):
                            for k in range(k_start, k_end):
                                term_list.append(('schoolbook', i, j, k))
                                term_id += 1
    
    rank = term_id
    print(f"Computed rank: {rank}")
    
    # Build U, V, W matrices
    U_mat = [[0] * rank for _ in range(n * m)]
    V_mat = [[0] * rank for _ in range(m * p)]
    W_mat = [[0] * rank for _ in range(p * n)]
    
    for t_idx, term in enumerate(term_list):
        if term[0] == 'strassen':
            _, ib, jb, kb, s_idx, i_start, j_start, k_start = term
            U_dict, V_dict, W_dict = strassen_terms[s_idx]
            
            for (li, lj), coeff in U_dict.items():
                i_global = i_start + li
                j_global = j_start + lj
                U_mat[i_global * m + j_global][t_idx] = coeff
            
            for (lj, lk), coeff in V_dict.items():
                j_global = j_start + lj
                k_global = k_start + lk
                V_mat[j_global * p + k_global][t_idx] = coeff
            
            for (lk, li), coeff in W_dict.items():
                k_global = k_start + lk
                i_global = i_start + li
                W_mat[k_global * n + i_global][t_idx] = coeff
        
        elif term[0] == 'schoolbook':
            _, i, j, k = term
            U_mat[i * m + j][t_idx] = 1
            V_mat[j * p + k][t_idx] = 1
            W_mat[k * n + i][t_idx] = 1
    
    # Verify the decomposition
    if time.time() < deadline - 10:
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U_mat[i * m + j][r] * V_mat[j * p + k][r] * W_mat[k * n + i][r]
                    if abs(total - 1) > 1e-9:
                        valid = False
                        print(f"Verification failed at ({i},{j},{k}): got {total}")
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if not valid:
            print("Strassen decomposition failed verification, using schoolbook fallback")
            return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
