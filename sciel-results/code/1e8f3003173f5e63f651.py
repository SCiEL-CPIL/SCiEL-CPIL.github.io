import numpy as np

DIMENSIONS = [4, 4, 7]

def _build_strassen_4x4x4():
    """
    Builds U, V, W for 4x4x4 matrix multiplication using Strassen's algorithm.
    Rank = 49.
    """
    n = 4
    rank = 49
    U = np.zeros((n*n, rank))
    V = np.zeros((n*n, rank))
    W = np.zeros((n*n, rank))
    
    # Recursive construction of Strassen terms
    # We represent the terms by their indices in the recursive tree
    # There are 7 terms at each level, depth 2.
    # Total 49 terms.
    
    # Helper to add a term
    def add_term(idx, u_vec, v_vec, w_vec):
        U[:, idx] = u_vec
        V[:, idx] = v_vec
        W[:, idx] = w_vec

    # We need to generate the 49 (u, v, w) triples.
    # Let's define the 7 Strassen combinations for 2x2x2.
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For 4x4x4, we apply this to 2x2 blocks of 2x2 matrices.
    # The "A" matrices in the top-level Strassen are 2x2 blocks of the 4x4 A.
    # So A11 is A[0:2, 0:2], A12 is A[0:2, 2:4], etc.
    # The multiplications M_s are products of these 2x2 blocks.
    # Each M_s is a 2x2 matrix product, which itself requires 7 multiplications of 2x2 scalars?
    # No, the M_s are 2x2 matrices. To compute them, we need to perform matrix multiplications.
    # Wait, the rank 49 comes from 7 multiplications of 2x2 matrices, each of which is computed with rank 7 (Strassen 2x2x2)?
    # No. Strassen 4x4x4 rank is 49.
    # This means we do 49 scalar multiplications.
    # This is equivalent to doing 7 multiplications of 2x2 matrices, where each 2x2 matrix multiplication is done with 7 scalar multiplications.
    # So yes, we recurse.
    
    # Let's define the structure.
    # Level 0: 4x4 matrices.
    # Split into 4 2x2 blocks.
    # Apply Strassen 2x2x2 logic to the blocks.
    # This gives 7 "intermediate" 2x2 matrix products.
    # For each intermediate product, we need to compute the 2x2 matrix product.
    # We use Strassen 2x2x2 for each of these.
    # This gives 7 * 7 = 49 scalar products.
    
    # We need to map the scalar products to the global U, V, W.
    
    # Let's define the 7 Strassen patterns for 2x2x2 as (u, v, w) vectors of length 4 (for 2x2 matrices).
    # Indices for 2x2: 0:00, 1:01, 2:10, 3:11.
    
    # Pattern 1: (A11+A22)(B11+B22)
    # u = [1, 0, 1, 1] (A11, A12, A21, A22) -> A11+A22 is [1, 0, 0, 1]
    # v = [1, 0, 0, 1] (B11, B12, B21, B22) -> B11+B22 is [1, 0, 0, 1]
    # w = [1, 0, 0, 1] (C11, C12, C21, C22) -> C11 gets M1
    # Wait, the contribution to C is:
    # C11 += M1
    # So w for M1 is [1, 0, 0, 0] for the output of the 2x2 product?
    # No, M1 is a 2x2 matrix. It contributes to C11, C12, C21, C22?
    # No, in Strassen 2x2x2, M1 contributes only to C11?
    # Let's check:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # So M1 contributes to C11 and C22.
    # So w for M1 is [1, 0, 0, 1].
    
    # Let's define the 7 (u, v, w) patterns for 2x2x2.
    # u, v, w are vectors of length 4 (flattened 2x2).
    
    p1_u = [1, 0, 0, 1] # A11 + A22
    p1_v = [1, 0, 0, 1] # B11 + B22
    p1_w = [1, 0, 0, 1] # C11 + C22
    
    p2_u = [0, 0, 1, 1] # A21 + A22
    p2_v = [1, 0, 0, 0] # B11
    p2_w = [0, 0, 1, 0] # C21
    
    p3_u = [1, 0, 0, 0] # A11
    p3_v = [0, 1, 0, -1] # B12 - B22
    p3_w = [0, 1, 0, 1] # C12 + C22
    
    p4_u = [0, 0, 0, 1] # A22
    p4_v = [0, 0, 1, -1] # B21 - B11
    p4_w = [1, 0, 1, 0] # C11 + C21
    
    p5_u = [1, 1, 0, 0] # A11 + A12
    p5_v = [0, 0, 0, 1] # B22
    p5_w = [0, 1, 0, -1] # C12 - C22
    
    p6_u = [0, 0, 1, -1] # A21 - A11
    p6_v = [1, 1, 0, 0] # B11 + B12
    p6_w = [0, 0, 0, 1] # C22
    
    p7_u = [0, 1, 0, -1] # A12 - A22
    p7_v = [0, 0, 1, 1] # B21 + B22
    p7_w = [1, 0, 0, -1] # C11 - C22
    
    patterns = [
        (p1_u, p1_v, p1_w),
        (p2_u, p2_v, p2_w),
        (p3_u, p3_v, p3_w),
        (p4_u, p4_v, p4_w),
        (p5_u, p5_v, p5_w),
        (p6_u, p6_v, p6_w),
        (p7_u, p7_v, p7_w)
    ]
    
    # Now, for 4x4x4:
    # We have 7 top-level patterns.
    # Each top-level pattern corresponds to a multiplication of two 2x2 matrices.
    # Let the top-level pattern be s (0..6).
    # The u_s and v_s are 2x2 matrices (flattened to 4).
    # We need to expand this 2x2x2 multiplication into 7 scalar multiplications.
    # For each top-level s, we iterate through the 7 bottom-level patterns t (0..6).
    # The total index is s * 7 + t.
    
    for s in range(7):
        u_top, v_top, w_top = patterns[s]
        # u_top, v_top are 4-vectors representing 2x2 matrices.
        # We need to form the 4x4 U, V vectors.
        # The 2x2 blocks of A are A00, A01, A10, A11.
        # u_top[0] is coeff for A00, u_top[1] for A01, etc.
        # So the 4x4 U vector is formed by repeating each coeff for the 4 entries in the 2x2 block?
        # No.
        # The multiplication is (A_block * B_block).
        # A_block is a 2x2 matrix formed by linear combination of A00, A01, A10, A11.
        # B_block is a 2x2 matrix formed by linear combination of B00, B01, B10, B11.
        # The product is a 2x2 matrix C_block.
        # C_block contributes to C00, C01, C10, C11 with weights from w_top.
        
        # We need to decompose the 2x2 matrix multiplication into 7 scalar multiplications.
        # Let the 2x2 multiplication be computed by Strassen 2x2x2.
        # The inputs to the 2x2 Strassen are the entries of A_block and B_block.
        # A_block entries are linear combinations of A's 16 entries.
        # B_block entries are linear combinations of B's 16 entries.
        
        # This is getting complicated to hardcode.
        # Alternative: Just use the known fact that rank is 49 and generate the matrices via a standard library or a simpler recursive builder.
        
        # Let's try a different approach:
        # Use the "Kronecker product" structure?
        # Strassen 4x4x4 is not simply a Kronecker product of Strassen 2x2x2.
        # But we can build it recursively.
        
        pass # Placeholder

    # Since hardcoding the 49 terms is error-prone, let's use a simpler decomposition if possible.
    # What if we just use schoolbook for 4x4x4 (rank 64) and schoolbook for 4x4x3 (rank 48)?
    # Total 112. No improvement.
    
    # What if we use Strassen for 4x4x4?
    # Let's try to construct it properly.
    
    # Recursive function to build U, V, W for n x n x n Strassen.
    def build_strassen(n, offset_u=0, offset_v=0, offset_w=0):
        """
        Returns U, V, W for n x n x n Strassen.
        U, V, W are lists of length n*n.
        """
        if n == 1:
            return [1], [1], [1]
        
        half = n // 2
        # Split indices into 4 blocks: 00, 01, 10, 11
        # Block 00: i in 0..half-1, j in 0..half-1
        # Block 01: i in 0..half-1, j in half..n-1
        # Block 10: i in half..n-1, j in 0..half-1
        # Block 11: i in half..n-1, j in half..n-1
        
        # We will create 7 sets of (U, V, W) for the 7 Strassen terms.
        # Each term is a multiplication of two half-sized matrices.
        # We recursively build U, V, W for the half-sized matrices.
        
        # The 7 patterns (u, v, w) for 2x2 blocks of size half.
        # u, v, w are lists of 4 elements, each being a linear combination of the 4 blocks.
        
        # For each of the 7 patterns, we need to combine the U, V, W from the recursive calls.
        
        # Let's define the 7 patterns as (coeffs_u, coeffs_v, coeffs_w)
        # coeffs_u is a list of 4 floats (for blocks 00, 01, 10, 11)
        
        patterns = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
            ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, -1]),
            ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, -1])
        ]
        
        U_full = [0] * (n*n)
        V_full = [0] * (n*n)
        W_full = [0] * (n*n)
        
        # We need to accumulate the contributions from the 7 patterns.
        # Each pattern s involves 4 block multiplications? No.
        # Each pattern s is ONE multiplication of two half-sized matrices.
        # This multiplication is itself decomposed into 7^(log2(half)) scalar multiplications.
        # So for each pattern s, we call build_strassen(half) to get the U, V, W for that half-sized multiplication.
        # Then we scale and place them into the full U, V, W.
        
        for s in range(7):
            cu, cv, cw = patterns[s]
            # Get U, V, W for the half-sized multiplication
            U_h, V_h, W_h = build_strassen(half)
            
            # Now, for each of the 4 blocks in u, v, w:
            # The u-block b (0..3) has coeff cu[b].
            # The U_h corresponds to the input of the half-sized multiplication.
            # The input of the half-sized multiplication is a linear combination of the A blocks.
            # Specifically, the A-block for term s is sum_b cu[b] A_b.
            # So the U vector for this term is sum_b cu[b] U_b, where U_b is the U for block b.
            # But U_b is just U_h placed in the block b position?
            # No. The half-sized multiplication multiplies (sum cu[b] A_b) by (sum cv[c] B_c).
            # The U vector for this term is the concatenation of cu[b] * U_h for each block b?
            # Yes. The U vector has length n*n.
            # Block b of U corresponds to rows i in block b.
            # For each row i in block b, U[i, term] = cu[b] * U_h[i_local].
            
            # Let's implement this placement.
            # Block 00: i in 0..half-1, j in 0..half-1
            # Block 01: i in 0..half-1, j in half..n-1
            # Block 10: i in half..n-1, j in 0..half-1
            # Block 11: i in half..n-1, j in half..n-1
            
            # We need to map (i, j) to block index and local index.
            
            for i in range(n):
                for j in range(n):
                    # Determine block
                    ib = 0 if i < half else 1
                    jb = 0 if j < half else 1
                    block_idx = ib * 2 + jb
                    # Local indices
                    il = i % half
                    jl = j % half
                    local_idx = il * half + jl
                    
                    # U
                    U_full[i*n + j] += cu[block_idx] * U_h[local_idx]
                    
            for i in range(n):
                for j in range(n):
                    ib = 0 if i < half else 1
                    jb = 0 if j < half else 1
                    block_idx = ib * 2 + jb
                    il = i % half
                    jl = j % half
                    local_idx = il * half + jl
                    
                    # V
                    V_full[i*n + j] += cv[block_idx] * V_h[local_idx]
                    
            for i in range(n):
                for j in range(n):
                    ib = 0 if i < half else 1
                    jb = 0 if j < half else 1
                    block_idx = ib * 2 + jb
                    il = i % half
                    jl = j % half
                    local_idx = il * half + jl
                    
                    # W
                    W_full[i*n + j] += cw[block_idx] * W_h[local_idx]
                    
        return U_full, V_full, W_full

    U_s, V_s, W_s = build_strassen(4)
    return U_s, V_s, W_s

def _build_schoolbook_4x4x3():
    """
    Builds U, V, W for 4x4x3 matrix multiplication using schoolbook.
    Rank = 48.
    Note: V has 28 rows (4x7), but we only use the last 12 (for the 3 columns).
    W has 28 rows (7x4), but we only use the last 12 (for the 3 columns).
    """
    n = 4
    m = 4
    p = 3
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*7, rank)) # Full size for B
    W = np.zeros((7*n, rank)) # Full size for C
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U
                U[i*m + j][col] = 1
                # V: k is local 0..2, global 4..6
                V[j*7 + (4 + k)][col] = 1
                # W: k is global 4..6, i is 0..3
                W[(4 + k)*n + i][col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4x4
    U1, V1, W1 = _build_strassen_4x4x4()
    # Rank 48 for 4x4x3
    U2, V2, W2 = _build_schoolbook_4x4x3()
    
    # Concatenate
    U = np.hstack([U1, U2])
    V = np.hstack([V1, V2])
    W = np.hstack([W1, W2])
    
    # Convert to lists of lists
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
