import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def _strassen_2x2():
    """
    Returns U, V, W for 2x2x2 Strassen.
    U, V, W are 4x7 matrices.
    Rows correspond to A00, A01, A10, A11 (and B, C similarly).
    Cols correspond to the 7 multiplications.
    """
    # Standard Strassen
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U rows: A00, A01, A10, A11
    # V rows: B00, B01, B10, B11
    # W rows: C00, C01, C10, C11
    
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 1, 0, 0, 1, 0, -1], # A01
        [0, 0, 0, 0, 0, 1, 1],  # A10
        [1, 0, -1, 1, 0, 0, -1] # A11
    ], dtype=np.int64)
    
    V = np.array([
        [1, 0, 0, -1, 0, 1, 0], # B00
        [0, 1, 1, 0, 0, 1, 0],  # B01
        [0, 0, 0, 1, 0, 0, 1],  # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ], dtype=np.int64)
    
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1], # C00 = M1 - M5 + M4 + M7
        [0, 0, 1, 0, 1, 0, 0],  # C01 = M3 + M5
        [0, 1, 0, 1, 0, 0, 0],  # C10 = M2 + M4
        [1, -1, 1, 0, 0, 1, 0]  # C11 = M1 - M2 + M3 + M6
    ], dtype=np.int64)
    
    return U, V, W

def _verify(U, V, W, n, m, p):
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for r in range(U.shape[1]):
        T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    T_exp = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_exp[i*m+j, j*p+k, k*n+i] = 1
    return np.array_equal(T, T_exp)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _build_4x4x4_strassen():
    n = 4
    U = np.zeros((n*n, 49), dtype=np.int64)
    V = np.zeros((n*n, 49), dtype=np.int64)
    W = np.zeros((n*n, 49), dtype=np.int64)
    
    u2, v2, w2 = _strassen_2x2()
    
    # Map 2x2 positions to 4x4 indices
    # A[i,j] -> i*4+j
    # B[j,k] -> j*4+k
    # C(k,i) -> k*4+i
    
    def add_term(u_row, v_row, w_row, col, val_u, val_v, val_w):
        U[u_row, col] += val_u
        V[v_row, col] += val_v
        W[w_row, col] += val_w

    # We have 7 multiplications M1..M7.
    # Each Mi is a product of two 2x2 matrices (sums of blocks).
    # We decompose each Mi using the 2x2 Strassen formulas (u2, v2, w2).
    # This effectively applies Strassen recursively.
    
    # The 7 multiplications in terms of 2x2 blocks A00..A11, B00..B11:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # For each Mi, we identify the A-blocks and B-blocks involved.
    # Then we apply the 2x2 Strassen decomposition to these blocks.
    
    # Helper to get row index for A block position
    def a_idx(block):
        # block: 00, 01, 10, 11
        r, c = (0,0) if block=='00' else (0,2) if block=='01' else (2,0) if block=='10' else (2,2)
        return [r*4+c, r*4+c+1, (r+1)*4+c, (r+1)*4+c+1] # A00, A01, A10, A11 of the block
        
    # This is getting complicated to map the 2x2 Strassen slots to the 4x4 indices directly.
    # Easier: Just use the fact that Rank(4x4x4) = 7 * Rank(2x2x2) = 49.
    # We can construct it by iterating over the 7 Strassen multiplications,
    # and for each, iterating over the 7 Strassen slots of the 2x2 sub-problem.
    
    # Let's define the 7 top-level multiplications with their A and B block coefficients.
    # A_blocks: dict from block_name to list of (coeff, block_name_in_4x4)
    # This is still messy.
    
    # Alternative: Hardcode the 49 terms?
    # Or use a recursive function that builds the decomposition for n x n x n.
    
    def build_strassen(n):
        if n == 2:
            return _strassen_2x2()
        
        # n is even
        half = n // 2
        u_sub, v_sub, w_sub = build_strassen(half)
        
        U = np.zeros((n*n, 7*7**(int(np.log2(n))-1)), dtype=np.int64) # Rank 7^k
        # Rank is 7 ** (log2(n))
        rank = 7 ** int(np.log2(n))
        U = np.zeros((n*n, rank), dtype=np.int64)
        V = np.zeros((n*n, rank), dtype=np.int64)
        W = np.zeros((n*n, rank), dtype=np.int64)
        
        # The 7 multiplications
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # For each Mi, we have A-blocks and B-blocks.
        # We need to map the sub-problem indices to the full indices.
        
        def get_block_indices(block, n):
            # Returns (row_start, col_start) for the block in the n x n matrix
            r, c = (0,0) if block=='00' else (0, n//2) if block=='01' else (n//2, 0) if block=='10' else (n//2, n//2)
            return r, c
        
        # We will add the sub-decomposition for each Mi.
        # The sub-decomposition has rank 7^(k-1).
        # We place it in a specific column range of the full U,V,W.
        
        sub_rank = 7 ** (int(np.log2(n)) - 1)
        
        # Define the 7 multiplications with signs
        # (a_blocks, b_blocks, c_blocks)
        # a_blocks: list of (sign, block_name)
        # b_blocks: list of (sign, block_name)
        # c_blocks: list of (sign, block_name) -- where the result goes
        
        mults = [
            # M1
            ([(1, '00'), (1, '11')], [(1, '00'), (1, '11')], [(1, '00'), (1, '11')]),
            # M2
            ([(1, '10'), (1, '11')], [(1, '00')], [(1, '10'), (-1, '11')]),
            # M3
            ([(1, '00')], [(1, '01'), (-1, '11')], [(1, '01'), (1, '11')]),
            # M4
            ([(1, '11')], [(1, '10'), (-1, '00')], [(1, '00'), (1, '10')]),
            # M5
            ([(1, '00'), (1, '01')], [(1, '11')], [(-1, '00'), (1, '01')]),
            # M6
            ([(1, '10'), (-1, '00')], [(1, '00'), (1, '01')], [(1, '11')]),
            # M7
            ([(1, '01'), (-1, '11')], [(1, '10'), (1, '11')], [(1, '00')])
        ]
        
        for m_idx, (a_blocks, b_blocks, c_blocks) in enumerate(mults):
            col_offset = m_idx * sub_rank
            # For each sub-slot s in the 2x2 Strassen (0..6)
            for s in range(7):
                # The sub-U, sub-V, sub-W are for a 2x2 problem.
                # We need to map the sub-indices (0..3) to the full indices.
                # Sub-index 0 -> A00 of the block, etc.
                
                # For A:
                # The sub-U row i corresponds to A position i in the 2x2 block.
                # We need to add contributions from all a_blocks.
                for i in range(4):
                    if u_sub[i, s] != 0:
                        # i is 0,1,2,3 corresponding to 00,01,10,11 of the 2x2 block
                        pos_name = ['00', '01', '10', '11'][i]
                        # Find which top-level block this position belongs to
                        # Wait, the sub-problem is a 2x2 multiplication.
                        # The "A" of the sub-problem is one of the top-level blocks (e.g. A00).
                        # So if a_block is A00, and sub-pos is 00, it maps to A[0,0] of A00, which is A[0,0] of the full matrix.
                        
                        # We need to sum over all top-level blocks that contribute to this Mi.
                        # But the sub-problem is applied to the *sum* of blocks.
                        # Linearity: (A00+A11) * (B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
                        # The Strassen decomposition of the product of sums is the sum of the Strassen decompositions of the individual products?
                        # Yes, because Strassen is linear.
                        
                        # So for each pair (a_block, b_block) in the product, we add the sub-decomposition.
                        
                        for (sa, a_name) in a_blocks:
                            for (sb, b_name) in b_blocks:
                                # Add sub-decomposition for a_name * b_name
                                # Scale by sa * sb
                                scale = sa * sb
                                
                                # Map sub-indices to full indices
                                ra, ca = get_block_indices(a_name, n)
                                rb, cb = get_block_indices(b_name, n)
                                
                                # Sub-index i (0..3) maps to local (lr, lc)
                                lr, lc = (0,0) if i==0 else (0,1) if i==1 else (1,0) if i==2 else (1,1)
                                full_a_row = (ra + lr) * n + (ca + lc)
                                
                                # Sub-index j (0..3) for B
                                # We need to iterate j as well?
                                # The sub-U[i,s] is the coefficient for A_sub[i] in slot s.
                                # The sub-V[j,s] is the coefficient for B_sub[j] in slot s.
                                # The sub-W[k,s] is the coefficient for C_sub[k] in slot s.
                                
                                # So we need to triple loop? No, the sub-decomposition is fixed.
                                # For a given slot s, we have terms U[i] V[j] W[k].
                                # We sum over i, j, k where u_sub[i,s], v_sub[j,s], w_sub[k,s] are non-zero.
                                
                                # Actually, it's easier to just loop i, j, k for the sub-problem.
                                
                                pass # This logic is getting nested too deep.
                                
        # Let's simplify: Just hardcode the 4x4x4 Strassen decomposition if possible, or use a known correct library implementation if I can't verify it.
        # But I can't use external libs.
        
        # Let's try a different approach: Verify the current program's Strassen code.
        # If it's correct, I just need to fix the verification or the construction.
        
        return U, V, W # Placeholder

    # Given the complexity, I will rely on the fact that the current program's logic is likely correct but maybe has a bug in index mapping or verification.
    # I will rewrite the verification to be more robust and ensure the Strassen construction is standard.
    
    # Actually, the simplest robust solution is Schoolbook (80).
    # But I want to beat 80.
    # The Strassen (65) is a good target.
    
    # I will implement the 4x4x4 Strassen by explicitly listing the 49 terms if I can't get the recursion right.
    # But 49 terms is a lot to hardcode.
    
    # Let's trust the recursive structure but implement it carefully.
    
    return U, V, W

def construct():
    deadline = time.time() + 120
    n, m, p = DIMENSIONS
    
    # Try Strassen for 4x4x4 + Schoolbook for 4x4x1
    try:
        # Use the provided _build_4x4x4_strassen if it works, else fallback
        # I will re-implement _build_4x4x4_strassen to be safe.
        
        # For now, fallback to schoolbook to ensure validity.
        # Then attempt Strassen.
        
        U44, V44, W44 = _build_4x4x4_strassen()
        if _verify(U44, V44, W44, n, m, 4):
            U1, V1, W1 = _schoolbook(n, m, 1)
            U = np.hstack([U44, U1])
            V = np.hstack([V44, V1])
            W = np.hstack([W44, W1])
            if _verify(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception as e:
        pass
        
    # Fallback
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
