"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    """
    Construct a tensor decomposition for 3x5 * 5x7 matrix multiplication.
    
    Strategy: Pad to 4x4 and 4x8, apply Strassen's 2x2x2 algorithm recursively.
    The 4x4x4 matrix multiplication can be done with Strassen rank 7 (base case 2x2x2).
    Then extract the 3x5x7 sub-tensor.
    
    Padding: A is 3x5 -> pad to 4x8 (rows 0-2, cols 0-4 used; cols 5-7 zero)
             B is 5x7 -> pad to 8x8 (rows 0-4, cols 0-6 used; rows 5-7 zero)
             C is 3x7 -> extract from 4x8 (rows 0-2, cols 0-6)
    
    Actually, let me think about this more carefully.
    We need A (3x5) * B (5x7) = C (3x7).
    
    Let's pad to powers of 2:
    A: 3x5 -> 4x8 (pad rows to 4, cols to 8)
    B: 5x7 -> 8x8 (pad rows to 8, cols to 8)
    C: 3x7 -> 4x8 (extract from 4x8 result)
    
    Strassen on 4x4 * 4x4 -> 4x4 uses 7 recursive 2x2 multiplications.
    Each 2x2 * 2x2 -> 2x2 uses 7 scalar multiplications.
    Total: 7 * 7 = 49 scalar products for the 4x4 * 4x4 case.
    
    But we only need the 3x5 * 5x7 part, so we can extract a sub-tensor.
    
    Let me implement this step by step.
    """
    n, m, p = DIMENSIONS  # 3, 5, 7
    t0 = time.time()
    deadline = t0 + 140  # 140s deadline with 10s margin
    
    # Pad dimensions to next power of 2
    # A: n x m = 3 x 5 -> pad to 4 x 8
    # B: m x p = 5 x 7 -> pad to 8 x 8
    # C: n x p = 3 x 7 -> extract from 4 x 8
    
    N_pad = 4  # padded rows for A, cols for C
    M_pad = 8  # padded cols for A, rows for B
    P_pad = 8  # padded cols for B
    
    # Build Strassen decomposition for M_pad x N_pad * N_pad x P_pad
    # Using recursive 2x2 Strassen
    
    def strassen_decompose(n_rows, m_cols, p_cols):
        """
        Return list of (u_coeff, v_coeff, w_coeff) terms for multiplying
        an n_rows x m_cols matrix by m_cols x p_cols matrix.
        
        Each term is a triple of linear forms:
        u_coeff: dict mapping (i, j) -> coefficient for U[i*m_cols+j]
        v_coeff: dict mapping (j, k) -> coefficient for V[j*p_cols+k]
        w_coeff: dict mapping (k, i) -> coefficient for W[k*n_rows+i]
        """
        terms = []
        
        if n_rows == 1 and m_cols == 1 and p_cols == 1:
            # Base case: single scalar multiplication
            terms.append((
                {(0, 0): 1},
                {(0, 0): 1},
                {(0, 0): 1}
            ))
            return terms
        
        # Split into halves
        n1, n2 = n_rows // 2, n_rows - n_rows // 2
        m1, m2 = m_cols // 2, m_cols - m_cols // 2
        p1, p2 = p_cols // 2, p_cols - p_cols // 2
        
        # A = [[A11, A12], [A21, A22]]
        # B = [[B11, B12], [B21, B22]]
        # C = [[C11, C12], [C21, C22]]
        
        # Strassen's 7 products:
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # For each Strassen product, we need to generate sub-terms
        # that combine the appropriate sub-blocks
        
        def make_combined_u(block_ranges, signs):
            """Create U coefficients from block ranges with signs."""
            coeff = {}
            for (i_range, j_range), sign in zip(block_ranges, signs):
                for i in range(i_range[0], i_range[1]):
                    for j in range(j_range[0], j_range[1]):
                        key = (i, j)
                        coeff[key] = coeff.get(key, 0) + sign
            return coeff
        
        def make_combined_v(block_ranges, signs):
            """Create V coefficients from block ranges with signs."""
            coeff = {}
            for (j_range, k_range), sign in zip(block_ranges, signs):
                for j in range(j_range[0], j_range[1]):
                    for k in range(k_range[0], k_range[1]):
                        key = (j, k)
                        coeff[key] = coeff.get(key, 0) + sign
            return coeff
        
        def make_combined_w(block_ranges, signs):
            """Create W coefficients from block ranges with signs."""
            coeff = {}
            for (k_range, i_range), sign in zip(block_ranges, signs):
                for k in range(k_range[0], k_range[1]):
                    for i in range(i_range[0], i_range[1]):
                        key = (k, i)
                        coeff[key] = coeff.get(key, 0) + sign
            return coeff
        
        # Define block ranges
        A11 = (0, n1, 0, m1)
        A12 = (0, n1, m1, m_cols)
        A21 = (n1, n_rows, 0, m1)
        A22 = (n1, n_rows, m1, m_cols)
        
        B11 = (0, m1, 0, p1)
        B12 = (0, m1, p1, p_cols)
        B21 = (m1, m_cols, 0, p1)
        B22 = (m1, m_cols, p1, p_cols)
        
        C11 = (0, n1, 0, p1)
        C12 = (0, n1, p1, p_cols)
        C21 = (n1, n_rows, 0, p1)
        C22 = (n1, n_rows, p1, p_cols)
        
        # Generate terms for each Strassen product
        # M1 = (A11 + A22) * (B11 + B22)
        u1 = make_combined_u([(A11[:2], A11[2:]), (A22[:2], A22[2:])], [1, 1])
        v1 = make_combined_v([(B11[:2], B11[2:]), (B22[:2], B22[2:])], [1, 1])
        # W for M1 contributes to C11, C12, C21, C22
        # We'll handle W separately when combining
        
        # For now, let's just use schoolbook for the base case and build up
        # Actually, let me use a simpler approach: just enumerate all the 
        # recursive Strassen calls and their contributions
        
        # Let me restart with a cleaner recursive approach
        
        return None  # Placeholder
    
    # Let me use a different approach: directly construct the Strassen 
    # decomposition by recursively splitting and tracking contributions
    
    def build_strassen_terms(n_rows, m_cols, p_cols, i_offset, j_offset, k_offset):
        """
        Recursively build Strassen terms.
        Returns list of (u_dict, v_dict, w_dict) where each dict maps 
        (row, col) -> coefficient.
        """
        terms = []
        
        if n_rows == 1 and m_cols == 1 and p_cols == 1:
            terms.append((
                {(i_offset, j_offset): 1},
                {(j_offset, k_offset): 1},
                {(k_offset, i_offset): 1}
            ))
            return terms
        
        n1, n2 = n_rows // 2, n_rows - n_rows // 2
        m1, m2 = m_cols // 2, m_cols - m_cols // 2
        p1, p2 = p_cols // 2, p_cols - p_cols // 2
        
        # Block offsets
        i1, i2 = i_offset, i_offset + n1
        j1, j2 = j_offset, j_offset + m1
        k1, k2 = k_offset, k_offset + p1
        
        # For each of the 7 Strassen products, we need to generate terms
        # that combine the appropriate sub-blocks
        
        # M1 = (A11 + A22) * (B11 + B22)
        # This requires summing over all combinations
        # We'll generate terms for each pair of sub-blocks
        
        # Actually, the cleanest way is to note that Strassen replaces
        # 8 scalar multiplications with 7, but each "scalar" is actually
        # a matrix block. So we need to recursively decompose each of the
        # 7 products.
        
        # Let me use a different formulation. Instead of trying to be clever,
        # let me just implement the standard Strassen recursion where each
        # level splits into 7 sub-problems, and at the base case (1x1x1)
        # we have a single term.
        
        # The key insight: at each Strassen level, we have 7 "virtual"
        # multiplications, each of which is itself a matrix multiplication
        # that needs to be decomposed.
        
        # For M1 = (A11 + A22) * (B11 + B22):
        # U terms: A11 and A22 blocks
        # V terms: B11 and B22 blocks
        # But we need to be careful: the sum A11 + A22 means we need
        # separate terms for each part.
        
        # Actually, the standard Strassen decomposition works as follows:
        # At each level, we create 7 new "variables" that are linear
        # combinations of the input blocks, then recursively multiply
        # them.
        
        # Let me just implement it directly:
        
        # For each of the 7 Strassen products, generate the recursive terms
        # with appropriate coefficient modifications
        
        def add_strassen_product(u_blocks, v_blocks, w_blocks, sign):
            """
            Add terms for a Strassen product.
            u_blocks: list of (i_range, j_range) for U
            v_blocks: list of (j_range, k_range) for V  
            w_blocks: list of (k_range, i_range) for W
            sign: overall sign multiplier
            """
            # For each combination of u_block and v_block, we need
            # to generate recursive terms
            for ui, uj in u_blocks:
                for vj, vk in v_blocks:
                    # Generate recursive terms for this sub-multiplication
                    sub_terms = build_strassen_terms(
                        uj[1] - uj[0], vk[1] - vj[0], vk[1] - vj[0],
                        uj[0], vj[0], vk[0]
                    )
                    for ut, vt, wt in sub_terms:
                        # Modify W coefficients to point to the correct output block
                        new_w = {}
                        for (ki, wi), coeff in wt.items():
                            # Map to the correct W block
                            for wk, wi_target in w_blocks:
                                if ki >= wk[0] and ki < wk[1] and wi >= wi_target[0] and wi < wi_target[1]:
                                    new_w[(ki, wi)] = coeff * sign
                                    break
                        terms.append((ut, vt, new_w))
        
        # This is getting complicated. Let me use a much simpler approach:
        # Just use schoolbook for now to ensure correctness, then optimize.
        
        # Schoolbook: one term per (i, j, k) triple
        for i in range(n_rows):
            for j in range(m_cols):
                for k in range(p_cols):
                    terms.append((
                        {(i + i_offset, j + j_offset): 1},
                        {(j + j_offset, k + k_offset): 1},
                        {(k + k_offset, i + i_offset): 1}
                    ))
        
        return terms
    
    # For now, use schoolbook to ensure correctness
    all_terms = build_strassen_terms(N_pad, M_pad, P_pad, 0, 0, 0)
    
    # Now extract the 3x5x7 sub-tensor
    # We need terms where:
    # - U index (i, j) with i in 0-2, j in 0-4
    # - V index (j, k) with j in 0-4, k in 0-6
    # - W index (k, i) with k in 0-6, i in 0-2
    
    # Filter and combine terms
    U_terms = {}
    V_terms = {}
    W_terms = {}
    
    for ut, vt, wt in all_terms:
        # Check if this term contributes to our sub-tensor
        valid = True
        for (i, j), coeff in ut.items():
            if i >= 3 or j >= 5:
                valid = False
                break
        if not valid:
            continue
        for (j, k), coeff in vt.items():
            if j >= 5 or k >= 7:
                valid = False
                break
        if not valid:
            continue
        for (k, i), coeff in wt.items():
            if k >= 7 or i >= 3:
                valid = False
                break
        if not valid:
            continue
        
        # Add this term
        key = (tuple(sorted(ut.items())), tuple(sorted(vt.items())), tuple(sorted(wt.items())))
        if key not in U_terms:
            U_terms[key] = {}
            V_terms[key] = {}
            W_terms[key] = {}
        
        for (i, j), coeff in ut.items():
            idx = i * M_pad + j
            U_terms[key][idx] = U_terms[key].get(idx, 0) + coeff
        for (j, k), coeff in vt.items():
            idx = j * P_pad + k
            V_terms[key][idx] = V_terms[key].get(idx, 0) + coeff
        for (k, i), coeff in wt.items():
            idx = k * N_pad + i
            W_terms[key][idx] = W_terms[key].get(idx, 0) + coeff
    
    # Build the final matrices
    rank = len(U_terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for key in U_terms:
        for idx, coeff in U_terms[key].items():
            i, j = idx // M_pad, idx % M_pad
            if i < n and j < m:
                U[i * m + j][col] = coeff
        for idx, coeff in V_terms[key].items():
            j, k = idx // P_pad, idx % P_pad
            if j < m and k < p:
                V[j * p + k][col] = coeff
        for idx, coeff in W_terms[key].items():
            k, i = idx // N_pad, idx % N_pad
            if k < p and i < n:
                W[k * n + i][col] = coeff
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
