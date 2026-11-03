"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 3x5x7 matrix multiplication.
    
    Strategy: Block decomposition.
    Split A (3x5) into rows [2, 1] and columns [3, 2].
    Split B (5x7) into rows [3, 2] and columns [5, 2].
    
    This yields four blocks of multiplication:
    1. A[0:2, 0:3] * B[0:3, :] -> Result part for rows 0:2, cols 0:7
       Dimensions: 2x3 * 3x7. Rank 42 (schoolbook).
       (Could potentially use Strassen-like for 2x3 or 3x3, but 2x3x7 is small.
        Actually, let's look closer at the blocks.)
        
    Let's refine the splitting to maximize Strassen usage (2x2x2 rank 7).
    
    A is 3x5. B is 5x7.
    
    Split A rows: 2, 1
    Split A cols: 2, 2, 1
    Split B rows: 2, 2, 1
    Split B cols: 2, 2, 2, 1
    
    Blocks of C (3x7):
    C is sum over k of A_i,k * B_k,j.
    
    Let's enumerate the non-zero scalar multiplication blocks (i, k, j):
    i in {0, 1} (2 blocks), k in {0, 1, 2} (3 blocks), j in {0, 1, 2, 3} (4 blocks)
    Wait, A has 3 rows, B has 7 cols.
    A rows split: 2, 1.
    B cols split: 2, 2, 2, 1.
    Middle dimension (5) split: 2, 2, 1.
    
    Total blocks = (2+1) * (2+2+1) * (2+2+2+1) is not how it works.
    It is sum over k.
    For each k (0,1,2 corresponding to sizes 2,2,1):
      For each i (0,1 corresponding to sizes 2,1):
        For each j (0,1,2,3 corresponding to sizes 2,2,2,1):
          Block A_ik is size (row_i) x (col_k).
          Block B_kj is size (row_k) x (col_j).
          Product is (row_i) x (col_j).
          
    Let's list them:
    k=0 (size 2):
      i=0 (size 2), j=0 (size 2): 2x2 * 2x2 -> 2x2. Use Strassen (rank 7).
      i=0 (size 2), j=1 (size 2): 2x2 * 2x2 -> 2x2. Use Strassen (rank 7).
      i=0 (size 2), j=2 (size 2): 2x2 * 2x2 -> 2x2. Use Strassen (rank 7).
      i=0 (size 2), j=3 (size 1): 2x2 * 2x1 -> 2x1. Use Schoolbook (4).
      i=1 (size 1), j=0 (size 2): 1x2 * 2x2 -> 1x2. Use Schoolbook (4).
      i=1 (size 1), j=1 (size 2): 1x2 * 2x2 -> 1x2. Use Schoolbook (4).
      i=1 (size 1), j=2 (size 2): 1x2 * 2x2 -> 1x2. Use Schoolbook (4).
      i=1 (size 1), j=3 (size 1): 1x2 * 2x1 -> 1x1. Use Schoolbook (2).
      
    k=1 (size 2):
      i=0 (size 2), j=0 (size 2): 2x2 * 2x2 -> 2x2. Use Strassen (rank 7).
      i=0 (size 2), j=1 (size 2): 2x2 * 2x2 -> 2x2. Use Strassen (rank 7).
      i=0 (size 2), j=2 (size 2): 2x2 * 2x2 -> 2x2. Use Strassen (rank 7).
      i=0 (size 2), j=3 (size 1): 2x2 * 2x1 -> 2x1. Use Schoolbook (4).
      i=1 (size 1), j=0 (size 2): 1x2 * 2x2 -> 1x2. Use Schoolbook (4).
      i=1 (size 1), j=1 (size 2): 1x2 * 2x2 -> 1x2. Use Schoolbook (4).
      i=1 (size 1), j=2 (size 2): 1x2 * 2x2 -> 1x2. Use Schoolbook (4).
      i=1 (size 1), j=3 (size 1): 1x2 * 2x1 -> 1x1. Use Schoolbook (2).
      
    k=2 (size 1):
      i=0 (size 2), j=0 (size 2): 2x1 * 1x2 -> 2x2. Schoolbook (4).
      i=0 (size 2), j=1 (size 2): 2x1 * 1x2 -> 2x2. Schoolbook (4).
      i=0 (size 2), j=2 (size 2): 2x1 * 1x2 -> 2x2. Schoolbook (4).
      i=0 (size 2), j=3 (size 1): 2x1 * 1x1 -> 2x1. Schoolbook (2).
      i=1 (size 1), j=0 (size 2): 1x1 * 1x2 -> 1x2. Schoolbook (2).
      i=1 (size 1), j=1 (size 2): 1x1 * 1x2 -> 1x2. Schoolbook (2).
      i=1 (size 1), j=2 (size 2): 1x1 * 1x2 -> 1x2. Schoolbook (2).
      i=1 (size 1), j=3 (size 1): 1x1 * 1x1 -> 1x1. Schoolbook (1).
      
    Total Rank:
    k=0: 7+7+7+4+4+4+4+2 = 39
    k=1: 7+7+7+4+4+4+4+2 = 39
    k=2: 4+4+4+2+2+2+2+1 = 21
    Total = 39 + 39 + 21 = 99.
    
    This matches the known improvement mentioned in the prompt (105 -> 99).
    The prompt says "Uses 6 Strassen 2x2x2 blocks... reducing total rank from 105 to 99".
    My count of Strassen blocks:
    k=0: i=0, j=0,1,2 (3 blocks)
    k=1: i=0, j=0,1,2 (3 blocks)
    Total 6 Strassen blocks. Correct.
    
    I will implement this explicitly.
    """
    n, m, p = 3, 5, 7
    
    # Split indices
    # A (n x m): rows [2, 1], cols [2, 2, 1]
    a_row_splits = [2, 1]
    a_col_splits = [2, 2, 1]
    
    # B (m x p): rows [2, 2, 1], cols [2, 2, 2, 1]
    b_row_splits = [2, 2, 1]
    b_col_splits = [2, 2, 2, 1]
    
    # Pre-compute offsets
    a_row_offsets = [0]
    for s in a_row_splits[:-1]: a_row_offsets.append(a_row_offsets[-1] + s)
    
    a_col_offsets = [0]
    for s in a_col_splits[:-1]: a_col_offsets.append(a_col_offsets[-1] + s)
    
    b_row_offsets = [0]
    for s in b_row_splits[:-1]: b_row_offsets.append(b_row_offsets[-1] + s)
    
    b_col_offsets = [0]
    for s in b_col_splits[:-1]: b_col_offsets.append(b_col_offsets[-1] + s)
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_block(a_r0, a_r1, a_c0, a_c1, b_r0, b_r1, b_c0, b_c1, rank, generator):
        """
        Adds a block multiplication to the tensors.
        generator(rank) yields (u_val, v_val, w_val, u_idx, v_idx, w_idx) or similar.
        Actually, simpler: just return lists of U, V, W columns for this block.
        """
        pass # Placeholder

    # We need a helper to generate standard basis indices for a block.
    # U index: i*m + j
    # V index: j*p + k  -- Wait, the middle dimension is m=5.
    # In the tensor product for C = A B:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So U is indexed by (i, j), V by (j, k), W by (k, i).
    # Dimensions:
    # U: (n*m) x rank
    # V: (m*p) x rank
    # W: (p*n) x rank
    
    # Let's define the indices for each block.
    # Block (i_block, k_block, j_block)
    # i_block in range(len(a_row_splits)) -> size a_row_splits[i_block]
    # k_block in range(len(a_col_splits)) -> size a_col_splits[k_block]
    # j_block in range(len(b_col_splits)) -> size b_col_splits[j_block]
    # Note: The middle index j in A corresponds to row index in B.
    # So k_block iterates over the column splits of A and row splits of B.
    # They must match. a_col_splits == b_row_splits. They are [2, 2, 1].
    
    for k_b in range(len(a_col_splits)):
        k_start = a_col_offsets[k_b]
        k_size = a_col_splits[k_b]
        
        for i_b in range(len(a_row_splits)):
            i_start = a_row_offsets[i_b]
            i_size = a_row_splits[i_b]
            
            for j_b in range(len(b_col_splits)):
                j_start = b_col_offsets[j_b]
                j_size = b_col_splits[j_b]
                
                # Block A: rows i_start..i_start+i_size, cols k_start..k_start+k_size
                # Block B: rows k_start..k_start+k_size, cols j_start..j_start+j_size
                # Result: rows i_start..i_start+i_size, cols j_start..j_start+j_size
                
                if i_size == 2 and k_size == 2 and j_size == 2:
                    # Strassen 2x2x2
                    # U indices: (i, j) relative to block
                    # V indices: (j, k) relative to block
                    # W indices: (k, i) relative to block
                    # But we need global indices.
                    # Global i = i_start + di
                    # Global j = k_start + dk
                    # Global k = j_start + dj
                    
                    # Strassen formulas:
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
                    
                    # We need to add 7 columns to U, V, W.
                    # For each Strassen factor M_m, we have terms for U, V, W.
                    # U corresponds to A entries. V to B entries. W to C entries.
                    
                    # M1: A11+A22, B11+B22
                    # A11 is at (i_start+0, k_start+0) -> U idx (i_start+0)*m + (k_start+0)
                    # A22 is at (i_start+1, k_start+1) -> U idx (i_start+1)*m + (k_start+1)
                    # B11 is at (k_start+0, j_start+0) -> V idx (k_start+0)*p + (j_start+0)
                    # B22 is at (k_start+1, j_start+1) -> V idx (k_start+1)*p + (j_start+1)
                    
                    # We will store (U_index, value, V_index, value, W_index, value) for each term?
                    # No, U, V, W are matrices. We add a column to each.
                    # For a given Strassen term, U has non-zeros, V has non-zeros, W has non-zeros.
                    # Actually, for Strassen, it's a product of sums.
                    # M1 = (A11 + A22) * (B11 + B22)
                    # This is not a single scalar product, it's a sum of products.
                    # Wait, the tensor decomposition requires rank-r terms.
                    # M1 is one rank term? No.
                    # M1 = A11*B11 + A11*B22 + A22*B11 + A22*B22.
                    # This requires 4 rank terms if done via schoolbook.
                    # But Strassen says M1 is computed in one "step"?
                    # No, Strassen computes 7 intermediate products.
                    # The "rank" refers to the number of scalar multiplications.
                    # So M1 is one multiplication: (A11+A22) * (B11+B22).
                    # So we need to represent this multiplication in the tensor form.
                    # U column: [1 at A11, 1 at A22, 0 else]
                    # V column: [1 at B11, 1 at B22, 0 else]
                    # W column: [1 at C11, -1 at C12? No, M1 contributes to C11 and C22 with +1]
                    # C11 = M1 + ...
                    # C22 = M1 - ...
                    # So W column for M1 has +1 at C11 and +1 at C22.
                    
                    # Let's verify the W contributions.
                    # C11 = M1 + M4 - M5 + M7
                    # C12 = M3 + M5
                    # C21 = M2 + M4
                    # C22 = M1 - M2 + M3 + M6
                    
                    # So for each M_m, we add a column to U, V, W.
                    
                    def add_strassen_term(u_terms, v_terms, w_terms):
                        """
                        u_terms: list of (global_u_idx, coeff)
                        v_terms: list of (global_v_idx, coeff)
                        w_terms: list of (global_w_idx, coeff)
                        """
                        u_col = [0] * (n * m)
                        v_col = [0] * (m * p)
                        w_col = [0] * (p * n)
                        
                        for idx, val in u_terms:
                            u_col[idx] += val
                        for idx, val in v_terms:
                            v_col[idx] += val
                        for idx, val in w_terms:
                            w_col[idx] += val
                        
                        U_cols.append(u_col)
                        V_cols.append(v_col)
                        W_cols.append(w_col)
                    
                    # Indices
                    # A11: (i_start+0, k_start+0)
                    # A12: (i_start+0, k_start+1)
                    # A21: (i_start+1, k_start+0)
                    # A22: (i_start+1, k_start+1)
                    
                    # B11: (k_start+0, j_start+0)
                    # B12: (k_start+0, j_start+1)
                    # B21: (k_start+1, j_start+0)
                    # B22: (k_start+1, j_start+1)
                    
                    # C11: (j_start+0, i_start+0) -> W idx (j_start+0)*n + (i_start+0)
                    # C12: (j_start+1, i_start+0) -> W idx (j_start+1)*n + (i_start+0)
                    # C21: (j_start+0, i_start+1) -> W idx (j_start+0)*n + (i_start+1)
                    # C22: (j_start+1, i_start+1) -> W idx (j_start+1)*n + (i_start+1)
                    
                    u11 = (i_start+0)*m + (k_start+0)
                    u12 = (i_start+0)*m + (k_start+1)
                    u21 = (i_start+1)*m + (k_start+0)
                    u22 = (i_start+1)*m + (k_start+1)
                    
                    v11 = (k_start+0)*p + (j_start+0)
                    v12 = (k_start+0)*p + (j_start+1)
                    v21 = (k_start+1)*p + (j_start+0)
                    v22 = (k_start+1)*p + (j_start+1)
                    
                    w11 = (j_start+0)*n + (i_start+0)
                    w12 = (j_start+1)*n + (i_start+0)
                    w21 = (j_start+0)*n + (i_start+1)
                    w22 = (j_start+1)*n + (i_start+1)
                    
                    # M1
                    add_strassen_term(
                        [(u11, 1), (u22, 1)],
                        [(v11, 1), (v22, 1)],
                        [(w11, 1), (w22, 1)]
                    )
                    
                    # M2
                    add_strassen_term(
                        [(u21, 1), (u22, 1)],
                        [(v11, 1)],
                        [(w21, 1), (w22, -1)]
                    )
                    
                    # M3
                    add_strassen_term(
                        [(u11, 1)],
                        [(v21, 1), (v22, -1)],
                        [(w12, 1), (w22, 1)]
                    )
                    
                    # M4
                    add_strassen_term(
                        [(u22, 1)],
                        [(v12, 1), (v11, -1)],
                        [(w11, 1), (w21, 1)]
                    )
                    
                    # M5
                    add_strassen_term(
                        [(u11, 1), (u12, 1)],
                        [(v22, 1)],
                        [(w11, -1), (w12, 1)]
                    )
                    
                    # M6
                    add_strassen_term(
                        [(u21, 1), (u11, -1)],
                        [(v11, 1), (v12, 1)],
                        [(w22, 1)]
                    )
                    
                    # M7
                    add_strassen_term(
                        [(u12, 1), (u22, -1)],
                        [(v21, 1), (v22, 1)],
                        [(w11, 1)]
                    )
                    
                else:
                    # Schoolbook for this block
                    # Size i_size x k_size for A
                    # Size k_size x j_size for B
                    # We need i_size * k_size * j_size rank terms
                    for di in range(i_size):
                        for dk in range(k_size):
                            for dj in range(j_size):
                                u_idx = (i_start + di) * m + (k_start + dk)
                                v_idx = (k_start + dk) * p + (j_start + dj)
                                w_idx = (j_start + dj) * n + (i_start + di)
                                
                                u_col = [0] * (n * m)
                                v_col = [0] * (m * p)
                                w_col = [0] * (p * n)
                                
                                u_col[u_idx] = 1
                                v_col[v_idx] = 1
                                w_col[w_idx] = 1
                                
                                U_cols.append(u_col)
                                V_cols.append(v_col)
                                W_cols.append(w_col)
    
    rank = len(U_cols)
    return {"U": U_cols, "V": V_cols, "W": W_cols}
# EVOLVE-BLOCK-END

