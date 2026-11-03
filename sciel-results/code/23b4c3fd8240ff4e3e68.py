import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2_factors():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    Shapes: (4, 7), (4, 7), (4, 7)
    U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B01-B00)
    U[3, 3] = 1
    V[0, 3] = -1; V[1, 3] = 1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[0, 5] = -1; U[2, 5] = 1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=8
    # We decompose into two 4x4x4 multiplications.
    # B = [B1, B2] where B1, B2 are 4x4.
    # C = [C1, C2] where C1 = A B1, C2 = A B2.
    # We need a rank-49 decomposition for 4x4x4.
    
    start_time = time.time()
    deadline = start_time + 120
    
    # Get 2x2 factors
    U2, V2, W2 = _strassen_2x2_factors()
    
    # Construct 4x4x4 factors using recursive Strassen.
    # The 4x4 multiplication is decomposed into 7 blocks of 2x2 multiplications.
    # Each 2x2 multiplication is decomposed into 7 factors.
    # Total 49 factors.
    
    # The 4x4 Strassen formulas:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Here A00, A01, etc are 2x2 blocks of the 4x4 matrix A.
    # A is 4x4. Indices 0..3.
    # A00 = A[0:2, 0:2]
    # A01 = A[0:2, 2:4]
    # A10 = A[2:4, 0:2]
    # A11 = A[2:4, 2:4]
    
    # We need to map the 2x2 factors to the 4x4 indices.
    # For a term like (A00+A11), the U factor for the 4x4 matrix will have non-zero entries
    # in the rows corresponding to A00 and A11.
    # A00 corresponds to rows 0,1 and cols 0,1. In flattened U (16xR), indices are 0,1,4,5.
    # A11 corresponds to rows 2,3 and cols 2,3. In flattened U, indices are 8,9,12,13.
    
    # Let's define a helper to create a 4x4 factor from 2x2 block coefficients.
    # We have 7 outer Strassen terms. For each term, we have 7 inner Strassen factors.
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # Mapping of 2x2 block indices to 4x4 flattened indices
    # Block (br, bc) where br, bc in {0,1}
    # Rows in 4x4: br*2 .. br*2+1
    # Cols in 4x4: bc*2 .. bc*2+1
    # Flattened index in U (A): row*4 + col
    def get_a_indices(br, bc):
        idxs = []
        for r in range(2):
            for c in range(2):
                idxs.append((br*2+r)*4 + (bc*2+c))
        return idxs
    
    def get_b_indices(br, bc):
        # B is 4x4.
        # V index is j*4+k.
        # Block (br, bc) of B corresponds to rows br*2..br*2+1 and cols bc*2..bc*2+1.
        # In V, the row is j (from A) and col is k (from B)?
        # Wait, V shape is (m*p, rank) = (4*4, rank) = (16, rank).
        # Index is j*4+k.
        # j is row of A / col of A?
        # A is n x m = 4 x 4.
        # B is m x p = 4 x 4.
        # V[j*p+k, r]. j is index in m, k is index in p.
        # In the 2x2 block context for B:
        # The "row" of the block is j (from A's col), "col" is k (from B's col).
        # So for block (br, bc) of B, j is in br*2..br*2+1, k is in bc*2..bc*2+1.
        idxs = []
        for j in range(2):
            for k in range(2):
                idxs.append((br*2+j)*4 + (bc*2+k))
        return idxs
        
    def get_c_indices(br, bc):
        # C is p x n = 4 x 4.
        # W shape is (p*n, rank) = (16, rank).
        # Index is k*4+i.
        # k is row of C (from B's col), i is col of C (from A's row).
        # Block (br, bc) of C: k in br*2..br*2+1, i in bc*2..bc*2+1.
        idxs = []
        for k in range(2):
            for i in range(2):
                idxs.append((br*2+k)*4 + (bc*2+i))
        return idxs

    # Outer Strassen terms definitions
    # Each term is (u_blocks, v_blocks, w_coeffs)
    # u_blocks: list of (block_idx, coeff) for A
    # v_blocks: list of (block_idx, coeff) for B
    # w_coeffs: dict mapping C_block_idx to coeff
    
    # Block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # A blocks: 00, 01, 10, 11
    # B blocks: 00, 01, 10, 11
    # C blocks: 00, 01, 10, 11
    
    terms = [
        # M1
        {
            'u': [(0, 1), (3, 1)],
            'v': [(0, 1), (3, 1)],
            'w': {0: 1, 3: 1}
        },
        # M2
        {
            'u': [(2, 1), (3, 1)],
            'v': [(0, 1)],
            'w': {2: 1, 3: -1}
        },
        # M3
        {
            'u': [(0, 1)],
            'v': [(2, 1), (3, -1)],
            'w': {1: 1, 3: 1}
        },
        # M4
        {
            'u': [(3, 1)],
            'v': [(0, -1), (1, 1)],
            'w': {0: 1, 2: 1}
        },
        # M5
        {
            'u': [(0, 1), (1, 1)],
            'v': [(3, 1)],
            'w': {0: -1, 1: 1}
        },
        # M6
        {
            'u': [(0, -1), (2, 1)],
            'v': [(0, 1), (1, 1)],
            'w': {3: 1}
        },
        # M7
        {
            'u': [(1, 1), (3, -1)],
            'v': [(2, 1), (3, 1)],
            'w': {0: -1}
        }
    ]
    
    col = 0
    for t_idx, term in enumerate(terms):
        # For each outer term, we use the 7 inner Strassen factors
        for r in range(7):
            # Get the 2x2 factors for this inner Strassen step
            u2 = U2[:, r]
            v2 = V2[:, r]
            w2 = W2[:, r]
            
            # Expand u2 to 4x4 U
            # u2 has 4 entries corresponding to A2x2 blocks?
            # No, u2 is for the 2x2 multiplication of the *blocks*.
            # The "elements" of the 2x2 A-block-matrix are the 2x2 blocks themselves.
            # So u2[0] corresponds to A_block_00, u2[1] to A_block_01, etc.
            # We need to map these to the 4x4 indices.
            
            for b_idx in range(4):
                coeff_u = u2[b_idx]
                if coeff_u != 0:
                    # b_idx 0 -> block (0,0), 1 -> (0,1), 2 -> (1,0), 3 -> (1,1)
                    br = b_idx // 2
                    bc = b_idx % 2
                    # Get indices in A
                    # The term 'u' in 'terms' specifies which A-blocks are involved in the *outer* Strassen sum.
                    # Wait, the logic is:
                    # The outer term specifies a combination of A-blocks and B-blocks.
                    # E.g., M1 uses A00 and A11.
                    # The inner Strassen factor r specifies how to multiply the 2x2 blocks.
                    # This is incorrect.
                    
                    # Correct Logic:
                    # The outer Strassen decomposes the 4x4 multiplication into 7 products of 2x2 blocks.
                    # Each product is of the form (Sum of A-blocks) * (Sum of B-blocks).
                    # We compute this product using Strassen's algorithm for 2x2 matrices?
                    # No, the blocks are 2x2 matrices. Their product is a 2x2 matrix multiplication.
                    # So for each outer term, we need to decompose the 2x2x2 multiplication of the blocks.
                    # The 2x2x2 multiplication uses the 7 factors from _strassen_2x2_factors().
                    
                    # So, for outer term t_idx:
                    # We have a linear combination of A-blocks: A_comb = sum c_i A_block_i
                    # And B-blocks: B_comb = sum d_j B_block_j
                    # We need to compute A_comb * B_comb.
                    # This is a 2x2 matrix multiplication.
                    # We use the 7 factors of 2x2 multiplication.
                    # For each of these 7 factors (r), we have:
                    # A_factor = sum u2[b] A_block_b
                    # B_factor = sum v2[b] B_block_b
                    # C_factor = sum w2[b] C_block_b
                    # The contribution to the total C is:
                    # sum_r (A_factor * B_factor) * w2_coeff?
                    # No, the Strassen decomposition is:
                    # C = sum_r (U_r A) (V_r B) W_r^T
                    # Here, the "elements" are 2x2 blocks.
                    # So for the outer term t_idx, which contributes to C with coefficients w_t:
                    # The term is M_t = (A_comb_t) (B_comb_t).
                    # M_t is a 2x2 matrix.
                    # We decompose M_t using the 7 inner factors.
                    # For inner factor r:
                    # A_part = sum_b u2[b] (A_comb_t)_b
                    # B_part = sum_b v2[b] (B_comb_t)_b
                    # C_part = sum_b w2[b] (C_block_b)
                    # The contribution to the final C (4x4) is:
                    # For each C_block_b in C_part with coeff w2[b]:
                    #   Add w2[b] * (A_part * B_part) to C_block_b.
                    # A_part is a linear combination of A-blocks.
                    # B_part is a linear combination of B-blocks.
                    # A_part * B_part is a 2x2 matrix product.
                    # This product is added to C_block_b.
                    
                    # So, for a specific (t_idx, r):
                    # We need to construct U_col, V_col, W_col.
                    # U_col will have non-zeros in the A-blocks involved in A_part.
                    # V_col will have non-zeros in the B-blocks involved in B_part.
                    # W_col will have non-zeros in the C-blocks involved in C_part.
                    
                    pass # Logic is complex, will implement in code below

    # Re-implementing the construction loop correctly
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    col = 0
    for t_idx, term in enumerate(terms):
        # term['u'] is list of (block_idx, coeff) for A
        # term['v'] is list of (block_idx, coeff) for B
        # term['w'] is dict {block_idx: coeff} for C
        
        # We need to compute the product of the two linear combinations of 2x2 blocks.
        # Let A_comb = sum_{(b,c)} c * A_b
        # Let B_comb = sum_{(b,c)} c * B_b
        # Product P = A_comb * B_comb.
        # We decompose this 2x2 multiplication using the 7 inner factors.
        
        for r in range(7):
            u2 = U2[:, r]
            v2 = V2[:, r]
            w2 = W2[:, r]
            
            # u2[b] is the coefficient of the b-th element of the 2x2 A-matrix (which consists of blocks)
            # Wait, u2 is for the 2x2 multiplication of *scalars* if the blocks were scalars.
            # Here the "elements" are 2x2 matrices.
            # So u2[0] multiplies A_block_00, u2[1] multiplies A_block_01, etc.
            
            # Construct U_col
            # For each b in 0..3:
            #   coeff = u2[b]
            #   If coeff != 0, we need to add coeff * (term['u'] contribution to block b) to U_col?
            # No.
            # The inner Strassen factor r operates on the 2x2 matrix of blocks.
            # The "A" of the inner multiplication is the 2x2 matrix of blocks A_comb.
            # The "B" is B_comb.
            # The inner factor r picks out a linear combination of the blocks of A_comb.
            # A_comb_block_b = sum_{(b', c')} c' * A_b' where (b',c') is the block index b.
            # This is getting confusing.
            
            # Let's simplify.
            # The 4x4 multiplication is bilinear.
            # We can just use the 49 factors generated by the tensor product of the two Strassen decompositions?
            # No, it's recursive.
            
            # Let's just use the 49 factors from a known source or generate them via the tensor product of 7x7?
            # If we use tensor product of two rank-7 decompositions, we get rank 49.
            # This corresponds to multiplying 4x4 matrices by viewing them as 2x2 matrices of 2x2 blocks,
            # and using Strassen for the 2x2 block multiplication, and Strassen for the 2x2 block internal multiplication.
            # This is exactly what I am trying to do.
            
            # Let's trace one factor.
            # Outer term t_idx.
            # Inner factor r.
            # U_col:
            #   For each block b in 0..3:
            #     u2[b] is the weight for A_block_b in the inner factor.
            #     But A_block_b itself is a linear combination of A's sub-blocks?
            #     No, A_block_b IS a 2x2 block of A.
            #     So A_block_b = A[br*2:br*2+2, bc*2:bc*2+2].
            #     The term 'u' in 'terms' specifies which of these blocks are in the outer sum.
            #     E.g., M1 uses A00 and A11.
            #     So A_comb_00 = A00, A_comb_11 = A11, others 0.
            #     The inner factor r uses u2[0] for A_comb_00 and u2[3] for A_comb_11.
            #     So the total coefficient for A00 is u2[0] * 1 (from term).
            #     The total coefficient for A11 is u2[3] * 1 (from term).
            #     So U_col should have u2[0] at indices of A00 and u2[3] at indices of A11.
            
            for b in range(4):
                u_coeff = u2[b]
                if u_coeff == 0:
                    continue
                # Find the term coefficient for this block
                # term['u'] is a list of (block_idx, coeff)
                term_coeff = 0
                for tb, tc in term['u']:
                    if tb == b:
                        term_coeff = tc
                        break
                if term_coeff == 0:
                    continue
                total_coeff = u_coeff * term_coeff
                if total_coeff == 0:
                    continue
                br = b // 2
                bc = b % 2
                idxs = get_a_indices(br, bc)
                for idx in idxs:
                    U4[idx, col] = total_coeff
                    
            # Similarly for V
            for b in range(4):
                v_coeff = v2[b]
                if v_coeff == 0:
                    continue
                term_coeff = 0
                for tb, tc in term['v']:
                    if tb == b:
                        term_coeff = tc
                        break
                if term_coeff == 0:
                    continue
                total_coeff = v_coeff * term_coeff
                if total_coeff == 0:
                    continue
                br = b // 2
                bc = b % 2
                idxs = get_b_indices(br, bc)
                for idx in idxs:
                    V4[idx, col] = total_coeff
                    
            # Similarly for W
            for b in range(4):
                w_coeff = w2[b]
                if w_coeff == 0:
                    continue
                term_coeff = term['w'].get(b, 0)
                if term_coeff == 0:
                    continue
                total_coeff = w_coeff * term_coeff
                if total_coeff == 0:
                    continue
                br = b // 2
                bc = b % 2
                idxs = get_c_indices(br, bc)
                for idx in idxs:
                    W4[idx, col] = total_coeff
            
            col += 1
            
    # Now we have U4, V4, W4 for 4x4x4.
    # We need to extend to 4x4x8.
    # B is 4x8. Split into B1 (4x4) and B2 (4x4).
    # C is 4x8. Split into C1 (4x4) and C2 (4x4).
    # C1 = A B1, C2 = A B2.
    # We use the same A factors for both.
    # U is 16x49 for A.
    # For B1, we use V4. For B2, we use V4.
    # V is 32x98.
    # V[0:16, 0:49] = V4 (for B1)
    # V[16:32, 49:98] = V4 (for B2)
    # W is 32x98.
    # W[0:16, 0:49] = W4 (for C1)
    # W[16:32, 49:98] = W4 (for C2)
    
    U = np.zeros((16, 98))
    V = np.zeros((32, 98))
    W = np.zeros((32, 98))
    
    U[:, 0:49] = U4
    U[:, 49:98] = U4
    
    V[0:16, 0:49] = V4
    V[16:32, 49:98] = V4
    
    W[0:16, 0:49] = W4
    W[16:32, 49:98] = W4
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
