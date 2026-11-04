"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify exact decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    rank = U.shape[1]
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition with rank = n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """
    Return the 7 rank-1 terms of Strassen's algorithm for 2x2x2 matrix multiplication.
    Each term is (u_indices, v_indices, w_indices) where each is a tuple of 4 entries
    corresponding to the 4 block positions (00, 01, 10, 11).
    
    The tensor equation for 2x2x2 matmul: T[i,j,k] = sum over blocks.
    Strassen uses 7 products:
      M1 = (A11-A22)(B21+B22)
      M2 = (A21+A22)B11
      M3 = A11(B11-B22)
      M4 = A22(B22-B11)
      M5 = (A11+A12)B22
      M6 = (A21-A11)(B11+B12)
      M7 = (A12-A22)(B21+B22)
    
    Output: C11 = M1+M4-M5+M7
            C12 = M3+M5
            C21 = M2+M4
            C22 = M1-M2+M3+M6
    
    For each product Mx, we need to express it as sum of rank-1 terms in the U,V,W factors.
    Actually, each Mx is a product of two 2x2 matrices, which itself can be done with rank 4 (schoolbook) 
    or we can use the fact that we're building the full decomposition.
    
    Let me re-think: In the tensor decomposition framework, each "term" r contributes:
      U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 if (i,j,k) is the target, 0 otherwise.
    
    For Strassen at the block level, we partition into 2x2 blocks. The key insight is that
    Strassen replaces 8 schoolbook terms with 7 "intermediate" terms, but each intermediate
    term is itself a matrix product that needs to be decomposed.
    
    Actually, the cleaner approach: Think of the full 3x5x6 problem as embedded in a 
    4x8x8 (or similar) problem where Strassen applies, then restrict.
    
    Let me use a different, cleaner approach:
    
    For 2x2x2, the tensor has 8 components T[i,j,k] for i,j,k in {0,1}.
    Schoolbook: rank 8, with U[i,j,r]=1, V[j,k,r]=1, W[k,i,r]=1 for r=(i,j,k).
    
    Strassen: rank 7. The 7 rank-1 tensors are:
    For each of the 7 M terms, we have a specific pattern.
    
    Actually, let me just construct it directly. The 7 rank-1 terms for 2x2x2 Strassen:
    
    Term 1 (M1 = (A11-A22)(B21+B22)):
      This contributes to C11 and C22.
      In the tensor: we need U[i,j,1]*V[j,k,1]*W[k,i,1] to give the right coefficients.
      
    This is getting complex. Let me use a known explicit construction.
    
    Known rank-7 decomposition for 2x2x2 matmul (Strassen):
    The 7 terms (U, V, W) entries for positions (i,j), (j,k), (k,i) where i,j,k in {0,1}:
    
    r=1: U = [[1, 0], [0, -1]], V = [[0, 0], [1, 1]], W = [[1, 0], [1, 0]]
    r=2: U = [[0, 0], [1, 1]], V = [[1, 0], [0, 0]], W = [[0, 0], [1, 0]]
    r=3: U = [[1, 0], [0, 0]], V = [[1, 0], [0, -1]], W = [[0, 0], [1, 0]]
    r=4: U = [[0, 0], [0, 1]], V = [[0, -1], [0, 1]], W = [[1, 0], [1, 0]]
    r=5: U = [[1, 1], [0, 0]], V = [[0, 0], [0, 1]], W = [[0, -1], [1, 0]]
    r=6: U = [[-1, 0], [1, 0]], V = [[1, 1], [0, 0]], W = [[0, 0], [0, 1]]
    r=7: U = [[0, 1], [0, -1]], V = [[0, 0], [1, 1]], W = [[0, 1], [1, 0]]
    
    Let me verify this is correct by checking the tensor equation.
    """
    # I'll use a verified Strassen decomposition
    # Each entry is for index (row, col) in the respective 2x2 matrix
    # U is 2x2 (i,j), V is 2x2 (j,k), W is 2x2 (k,i)
    
    terms = []
    # Using a standard Strassen rank-7 decomposition
    # I'll construct it carefully
    
    # Let's use the explicit formulas from the literature
    # M1 = (A00-A11)(B10+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B00-B11)
    # M4 = A11(B11-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each Mx, it's a product of two 2x2 matrices, which in the tensor decomposition
    # framework means we need to express each Mx as a sum of rank-1 terms.
    # But actually, in the tensor decomposition, each "rank-1 term" corresponds to one
    # scalar multiplication in the algorithm.
    
    # The correct way: each of the 7 Strassen multiplications is a single scalar product
    # of two linear combinations. In the tensor decomposition, this means:
    # For M1: U vector is the coefficients of (A00-A11), V vector is coefficients of (B10+B11),
    #         and W vector distributes the result to C00 and C11.
    
    # U for M1: [1, 0, 0, -1] (coefficients for A00, A01, A10, A11)
    # V for M1: [0, 0, 1, 1] (coefficients for B00, B01, B10, B11)
    # W for M1: [1, 0, 0, 1] (coefficients for C00, C01, C10, C11) -- wait, this isn't right
    # because W[k,i] has indices k,i, not just position.
    
    # Let me use a different, cleaner formulation. I'll just hardcode a known-good
    # rank-7 decomposition for 2x2x2 and verify it numerically.
    
    # Actually, the simplest approach: use the fact that for our specific dimensions (3,5,6),
    # we can partition into blocks and apply Strassen where it fits.
    
    # For 3x5 x 5x6:
    # Partition A (3x5) into blocks of size 2x2:
    #   Row blocks: [0:2], [2:3] -> 2 row-blocks (second is 1x5)
    #   Col blocks: [0:2], [2:4], [4:5] -> 3 col-blocks (third is 1x1)
    # This doesn't align nicely with 2x2.
    
    # Alternative: Pad to 4x8 x 8x8, apply Strassen, then restrict.
    # Or: Use a hybrid approach where we apply Strassen to the 2x2 sub-blocks that fit.
    
    # Let me try a direct approach: construct the decomposition by considering
    # each (i,j,k) triple and using Strassen's structure to share terms.
    
    # For now, let me return a simple construction that I can verify.
    pass

def _block_strassen(n, m, p):
    """
    Construct a rank-reduced decomposition using block Strassen.
    
    Strategy: Partition the matrices into 2x2 block grids where possible,
    apply Strassen's 7-product algorithm to each 2x2x2 sub-block, and
    handle boundary blocks with schoolbook.
    
    For 3x5 x 5x6:
    - Pad A to 4x6, B to 6x6 (or work with the actual sizes)
    - Actually, let's partition as follows:
      A (3x5): rows [0:2], [2:3]; cols [0:2], [2:4], [4:5]
      B (5x6): rows [0:2], [2:4], [4:5]; cols [0:2], [2:4], [4:6]
    
    This gives a 2x3 grid of A blocks and 3x3 grid of B blocks.
    The product C (3x6) has a 2x3 grid of blocks.
    
    For each 2x2 sub-block of the block grid, we can apply Strassen.
    But the grids don't align to 2x2 everywhere.
    
    Simpler approach: Just use Strassen on the overlapping 2x2 regions.
    
    Actually, let me try a completely different structural approach:
    Use the fact that rank of matmul tensor is known to be less than nmp for
    many dimensions. For 3x5x6, the known upper bound from Strassen-based
    methods is significantly below 90.
    
    Let me try: partition into 2x2 blocks where possible, use Strassen (rank 7)
    for full 2x2x2 blocks, and schoolbook (rank 8) for partial blocks, then
    prune duplicate terms.
    """
    # Build the decomposition term by term
    U_terms = []  # list of (row_idx, value) or full vectors
    V_terms = []
    W_terms = []
    
    # For each (i, j, k) in the output, we need the sum over r of
    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Schoolbook baseline:
    # r = i*m*p + j*p + k: U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    
    # Strassen idea: group terms that can share structure.
    # For a 2x2x2 sub-block with i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}:
    # Instead of 8 terms, use 7.
    
    # Let's identify all 2x2x2 sub-blocks in our 3x5x6 grid.
    # i ranges 0..2, j ranges 0..4, k ranges 0..5
    # 2x2x2 sub-blocks: i0 in {0,1}, j0 in {0,1,2,3}, k0 in {0,1,2,3,4}
    # Total: 2 * 4 * 5 = 40 potential sub-blocks, but they overlap.
    
    # This approach of applying Strassen to overlapping blocks is tricky.
    # Let me use a cleaner method: direct construction with term sharing.
    
    # Alternative: Use the "laser method" or just try to find a good rank
    # by exploiting the rectangular structure.
    
    # For now, let's use a pragmatic approach:
    # 1. Start with schoolbook (rank 90)
    # 2. Try to merge terms that have the same (U row, V row, W row) pattern
    # 3. Apply Strassen to non-overlapping 2x2x2 blocks
    
    # Non-overlapping 2x2x2 blocks in 3x5x6:
    # i: [0,1], [2] (only one full 2-row block)
    # j: [0,1], [2,3], [4] (two full 2-col blocks)
    # k: [0,1], [2,3], [4,5] (three full 2-col blocks)
    # So we can have 1 * 2 * 3 = 6 non-overlapping 2x2x2 blocks.
    # Each uses rank 7 instead of 8, saving 1 rank per block.
    # Total savings: 6, giving rank 90 - 6 = 84.
    
    # But wait, the blocks must be aligned. Let's define:
    # Block (bi, bj, bk) covers i in [2*bi, 2*bi+1], j in [2*bj, 2*bj+1], k in [2*bk, 2*bk+1]
    # bi in {0}, bj in {0, 1}, bk in {0, 1, 2}
    # Remaining indices: i=2, j=4, and all combinations not in full blocks.
    
    # Actually, i=2 is a single row, j=4 is a single col, so:
    # Full 2x2x2 blocks: bi=0, bj in {0,1}, bk in {0,1,2} -> 6 blocks
    # Remaining: i=2 (any j,k), j=4 (any i,k), and the "edges"
    # But j=4 is only in the last column, so it's not part of any 2-col block starting at even j.
    # Similarly i=2 is not part of any 2-row block.
    
    # Let me just construct it explicitly.
    
    rank_terms = []  # Each term: (U_vec, V_vec, W_vec) as dicts or arrays
    
    # Helper to add a schoolbook term
    def add_schoolbook(i, j, k):
        U = np.zeros(n * m)
        V = np.zeros(m * p)
        W = np.zeros(p * n)
        U[i * m + j] = 1.0
        V[j * p + k] = 1.0
        W[k * n + i] = 1.0
        rank_terms.append((U, V, W))
    
    # Helper to add a Strassen term for a 2x2x2 block
    def add_strassen_block(i0, j0, k0):
        """Add 7 Strassen terms for the 2x2x2 block starting at (i0, j0, k0)."""
        # The 7 Strassen terms for 2x2x2:
        # Each term is a rank-1 tensor in the 2x2x2 sub-tensor.
        # U[i,j], V[j,k], W[k,i] for i,j,k in {0,1} relative to the block.
        
        # I'll use a verified set of 7 terms.
        # Let me define them explicitly and verify.
        
        strassen_terms = [
            # (U_2x2, V_2x2, W_2x2)
            # U is indexed [i, j], V is [j, k], W is [k, i]
            # Using 0-based: U[0,0], U[0,1], U[1,0], U[1,1]
            
            # M1 = (A00 - A11)(B10 + B11)
            # Contributes to C00 (+1) and C11 (+1)
            # U: [1, 0, 0, -1] (A00=1, A01=0, A10=0, A11=-1)
            # V: [0, 0, 1, 1] (B00=0, B01=0, B10=1, B11=1)
            # W: [1, 0, 1, 0] (C00=1, C01=0, C10=1, C11=0) -- wait, W[k,i]
            # W[0,0]=1 (k=0,i=0 -> C00), W[0,1]=0 (k=0,i=1 -> C10? no, C[i,k]... )
            
            # I'm getting confused with the index conventions. Let me be very careful.
            # In our problem: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # For a 2x2x2 block at (i0, j0, k0):
            #   i = i0 + di, j = j0 + dj, k = k0 + dk, where di,dj,dk in {0,1}
            #   U[(i0+di)*m + (j0+dj), r]
            #   V[(j0+dj)*p + (k0+dk), r]
            #   W[(k0+dk)*n + (i0+di), r]
            
            # The Strassen identity says:
            # C[di, dk] = sum over the 7 products
            # where C[di, dk] is the (di, dk) entry of the 2x2 output block.
            
            # Let me just hardcode a known-correct decomposition and verify it.
        ]
        
        # I'll use a different approach: generate the Strassen terms by
        # solving the linear system for the 2x2x2 case.
        
        # For 2x2x2, we need 7 vectors (u_r, v_r, w_r) each of length 4 (flattened 2x2)
        # such that for all (i,j,k) in {0,1}^3:
        #   sum_r u_r[i*2+j] * v_r[j*2+k] * w_r[k*2+i] = 1
        
        # This is a system of 8 equations in 7*3 = 21 unknowns (with the constraint
        # that each term is rank-1 in the tensor sense).
        
        # Known solution (from Strassen's original paper, adapted to our indexing):
        # I'll use this set and verify numerically.
        
        terms_2x2 = [
            # r=1: M1
            (np.array([1, 0, 0, -1.0]), np.array([0, 0, 1, 1.0]), np.array([1, 0, 0, 1.0])),
            # r=2: M2
            (np.array([0, 0, 1, 1.0]), np.array([1, 0, 0, 0.0]), np.array([0, 0, 1, 0.0])),
            # r=3: M3
            (np.array([1, 0, 0, 0.0]), np.array([1, 0, 0, -1.0]), np.array([0, 0, 1, 0.0])),
            # r=4: M4
            (np.array([0, 0, 0, 1.0]), np.array([0, -1, 0, 1.0]), np.array([1, 0, 1, 0.0])),
            # r=5: M5
            (np.array([1, 1, 0, 0.0]), np.array([0, 0, 0, 1.0]), np.array([0, -1, 0, 1.0])),
            # r=6: M6
            (np.array([-1, 0, 1, 0.0]), np.array([1, 1, 0, 0.0]), np.array([0, 0, 0, 1.0])),
            # r=7: M7
            (np.array([0, 1, 0, -1.0]), np.array([0, 0, 1, 1.0]), np.array([0, 1, 0, 1.0])),
        ]
        
        # Verify this 2x2x2 decomposition
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = 0.0
                    for (u, v, w) in terms_2x2:
                        s += u[i * 2 + j] * v[j * 2 + k] * w[k * 2 + i]
                    if abs(s - 1.0) > 1e-9:
                        # Fallback to schoolbook for this block
                        for di in range(2):
                            for dj in range(2):
                                for dk in range(2):
                                    add_schoolbook(i0 + di, j0 + dj, k0 + dk)
                        return
                        # Don't add Strassen terms, use schoolbook instead
        
        # Add the 7 Strassen terms mapped to global indices
        for (u, v, w) in terms_2x2:
            U = np.zeros(n * m)
            V = np.zeros(m * p)
            W = np.zeros(p * n)
            for di in range(2):
                for dj in range(2):
                    U[(i0 + di) * m + (j0 + dj)] = u[di * 2 + dj]
            for dj in range(2):
                for dk in range(2):
                    V[(j0 + dj) * p + (k0 + dk)] = v[dj * 2 + dk]
            for dk in range(2):
                for di in range(2):
                    W[(k0 + dk) * n + (i0 + di)] = w[dk * 2 + di]
            rank_terms.append((U, V, W))
    
    # Identify which (i,j,k) triples are covered by Strassen blocks
    covered = set()
    strassen_blocks = []
    for i0 in range(0, n - 1, 2):
        for j0 in range(0, m - 1, 2):
            for k0 in range(0, p - 1, 2):
                strassen_blocks.append((i0, j0, k0))
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            covered.add((i0 + di, j0 + dj, k0 + dk))
    
    # Add Strassen terms for each block
    for (i0, j0, k0) in strassen_blocks:
        add_strassen_block(i0, j0, k0)
    
    # Add schoolbook terms for uncovered (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    add_schoolbook(i, j, k)
    
    # Now try to prune: remove terms that are linearly dependent or zero
    # Build the full matrices
    if not rank_terms:
        return None
    
    total_rank = len(rank_terms)
    U_mat = np.zeros((n * m, total_rank))
    V_mat = np.zeros((m * p, total_rank))
    W_mat = np.zeros((p * n, total_rank))
    
    for r, (U, V, W) in enumerate(rank_terms):
        U_mat[:, r] = U
        V_mat[:, r] = V
        W_mat[:, r] = W
    
    # Verify
    if _verify(U_mat, V_mat, W_mat, n, m, p):
        return {"U": U_mat, "V": V_mat, "W": W_mat}
    
    return None

def _prune_decomposition(U, V, W, n, m, p):
    """Try to reduce rank by removing linearly dependent terms."""
    rank = U.shape[1]
    if rank == 0:
        return None
    
    # Try removing each term one at a time and check if the decomposition still holds
    # This is O(rank * n*m*p * rank) which could be slow, but rank is at most 90.
    
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    # Try to remove terms that are all-zero or nearly all-zero
    for r in range(rank - 1, -1, -1):
        if np.allclose(U[:, r], 0) or np.allclose(V[:, r], 0) or np.allclose(W[:, r], 0):
            U = np.delete(U, r, axis=1)
            V = np.delete(V, r, axis=1)
            W = np.delete(W, r, axis=1)
    
    # Try removing terms that are linear combinations of others
    # For each term r, check if removing it still gives a valid decomposition.
    # This is expensive, so only do it for a subset.
    current_rank = U.shape[1]
    removed = 0
    for r in range(current_rank - 1, -1, -1):
        if removed > 10:  # Limit pruning effort
            break
        U_test = np.delete(U, r, axis=1)
        V_test = np.delete(V, r, axis=1)
        W_test = np.delete(W, r, axis=1)
        if _verify(U_test, V_test, W_test, n, m, p):
            U = U_test
            V = V_test
            W = W_test
            removed += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Try block Strassen first
    result = _block_strassen(n, m, p)
    if result is not None:
        # Try to prune
        pruned = _prune_decomposition(result["U"], result["V"], result["W"], n, m, p)
        if pruned is not None:
            result = pruned
        if _verify(result["U"], result["V"], result["W"], n, m, p):
            return {"U": result["U"].tolist(), "V": result["V"].tolist(), "W": result["W"].tolist()}
    
    # Fallback to schoolbook
    result = _schoolbook(n, m, p)
    return {"U": result["U"].tolist(), "V": result["V"].tolist(), "W": result["W"].tolist()}
# EVOLVE-BLOCK-END
