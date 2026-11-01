import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=7

    start_time = time.time()
    deadline = start_time + 140.0

    def verify(U, V, W, tol=1e-9):
        r = U.shape[1]
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        s = 0.0
                        for c in range(r):
                            s += U[i*m + j, c] * V[jp*p + k, c] * W[k*n + i, c]
                        if abs(s - (1.0 if j == jp else 0.0)) > tol:
                            return False
        return True

    # Strategy: Use Strassen for 2x2x2 blocks, and for the remaining 2x2x1
    # blocks, use a rank-3 decomposition instead of rank-4.
    #
    # For 2x2 * 2x1: C = A*B where A is 2x2, B is 2x1
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    #
    # Standard: 4 multiplications
    # Strassen-like: 
    #   M1 = (A00+A11) * (B0+B1)
    #   M2 = (A10-A00) * B0
    #   M3 = (A01-A11) * B1
    #   C0 = M1 - M3  ... wait, that's not right
    #
    # Actually for 2x2 * 2x1, the minimum rank is 4 (it's a full-rank 2x2 matrix
    # times 2x1, which is equivalent to 2 independent 1x2*2x1, each of rank 2).
    # So rank-4 is minimal for that sub-problem.
    #
    # Alternative: use a different block structure.
    # Instead of (2x2 x 2x2) + (2x2 x 2x1), try overlapping blocks.
    #
    # Let's try: 2x4 * 4x7 = use Strassen 2x2x2 for the first 6 columns,
    # then handle column 6 specially with a 2x2 * 2x1 using 4 terms.
    # Total: 6*7 + 2*4 = 42 + 8 = 50. That's what we have.
    #
    # Can we do better? Let's try a different partition.
    #
    # Idea: Use 2x3 blocks. 2x3 * 3x3 has rank at most 2*3*3=18 (schoolbook)
    # but with Strassen-like methods could be less. Actually for 2x3*3x3,
    # we can do it in 2*(rank of 2x3*3x1) + ... hmm, this gets complicated.
    #
    # Let me try: partition 4 into 2+2, and 7 into 3+3+1.
    # For 2x2 * 2x3: rank at most 2*2*3=12 schoolbook, or use Strassen
    # on 2x2*2x2 (7 terms) + 2x2*2x1 (4 terms) = 11 terms.
    # Total: 3 blocks * 11 = 33 for the 3+3 part, plus 2x2*2x1 = 4 for last col.
    # Total: 33 + 4 = 37? That seems too good. Let me check.
    #
    # Wait, 2x2 * 2x3: A is 2x2, B is 2x3. C = A*B is 2x3.
    # C[:,0:2] = A * B[:,0:2] (2x2 * 2x2, Strassen rank 7)
    # C[:,2] = A * B[:,2] (2x2 * 2x1, rank 4)
    # So 2x2 * 2x3 has rank 7+4=11. Yes!
    #
    # So partition 7 as 3+3+1:
    # Three 2x2*2x3 blocks: each rank 11, total 33
    # One 2x2*2x1 block: rank 4
    # Total: 37!
    
    rank = 37
    U = np.zeros((n*m, rank), dtype=np.int32)
    V = np.zeros((m*p, rank), dtype=np.int32)
    W = np.zeros((p*n, rank), dtype=np.int32)
    col = 0

    def add_strassen_222(i0, j0, k0):
        """Add Strassen 2x2x2 block for rows i0,i0+1; cols j0,j0+1; output k0,k0+1"""
        nonlocal col
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)
        e = j0 * p + k0
        f = j0 * p + (k0 + 1)
        g = (j0 + 1) * p + k0
        h = (j0 + 1) * p + (k0 + 1)
        c00 = k0 * n + i0
        c01 = (k0 + 1) * n + i0
        c10 = k0 * n + (i0 + 1)
        c11 = (k0 + 1) * n + (i0 + 1)

        # M1 = (A00+A11)*(B00+B11) -> C00+=M1, C11+=M1
        U[a, col] += 1; U[d, col] += 1
        V[e, col] += 1; V[h, col] += 1
        W[c00, col] += 1; W[c11, col] += 1
        col += 1
        # M2 = (A10+A11)*B00 -> C10+=M2, C11-=M2
        U[c, col] += 1; U[d, col] += 1
        V[e, col] += 1
        W[c10, col] += 1; W[c11, col] -= 1
        col += 1
        # M3 = A00*(B10-B00) -> C01+=M3, C11+=M3
        U[a, col] += 1
        V[f, col] += 1; V[h, col] -= 1
        W[c01, col] += 1; W[c11, col] += 1
        col += 1
        # M4 = A11*(B01-B10) -> C00+=M4, C10+=M4
        U[d, col] += 1
        V[g, col] += 1; V[e, col] -= 1
        W[c00, col] += 1; W[c10, col] += 1
        col += 1
        # M5 = (A00+A01)*B11 -> C00-=M5, C01+=M5
        U[a, col] += 1; U[b, col] += 1
        V[h, col] += 1
        W[c00, col] -= 1; W[c01, col] += 1
        col += 1
        # M6 = (A10-A00)*(B00+B01) -> C11+=M6
        U[c, col] += 1; U[a, col] -= 1
        V[e, col] += 1; V[f, col] += 1
        W[c11, col] += 1
        col += 1
        # M7 = (A01-A11)*(B10+B11) -> C00+=M7
        U[b, col] += 1; U[d, col] -= 1
        V[g, col] += 1; V[h, col] += 1
        W[c00, col] += 1
        col += 1

    def add_schoolbook_221(i0, j0, k0):
        """Add 2x2 * 2x1 schoolbook block (rank 4)"""
        nonlocal col
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)
        e = j0 * p + k0
        g = (j0 + 1) * p + k0
        c00 = k0 * n + i0
        c10 = k0 * n + (i0 + 1)

        U[a, col] += 1; V[e, col] += 1; W[c00, col] += 1; col += 1
        U[b, col] += 1; V[g, col] += 1; W[c00, col] += 1; col += 1
        U[c, col] += 1; V[e, col] += 1; W[c10, col] += 1; col += 1
        U[d, col] += 1; V[g, col] += 1; W[c10, col] += 1; col += 1

    # Partition p=7 into 3+3+1
    # For each 3-wide k-block: use Strassen 2x2x2 for first 2 cols + schoolbook 2x2x1 for last col
    # For each 3-wide j-block: similar structure
    #
    # Actually, let's be more careful. We have n=2, m=4, p=7.
    # Partition m=4 as 2+2, and p=7 as 3+3+1.
    # For each (j_block, k_block) pair:
    #   j_block is 2 wide (j0, j0+1)
    #   k_block is 3 wide (k0, k0+1, k0+2)
    #   Use Strassen 2x2x2 for (j0,j0+1) x (k0,k0+1) -> rank 7
    #   Use schoolbook 2x2x1 for (j0,j0+1) x (k0+2) -> rank 4
    #   Total per block: 11
    #
    # We have 2 j-blocks and 3 k-blocks, so 6 blocks * 11 = 66? That's worse!
    #
    # Wait, I need to be smarter. The Strassen 2x2x2 handles a 2x2 A block times 2x2 B block.
    # Let me reconsider.
    #
    # Better approach: partition p=7 as 2+2+2+1.
    # For each 2-wide k-block: Strassen 2x2x2 (rank 7) for each 2-wide j-block
    # For the 1-wide k-block: schoolbook 2x2x1 (rank 4) for each 2-wide j-block
    # Total: 3 k-blocks * 2 j-blocks * 7 + 1 k-block * 2 j-blocks * 4 = 42 + 8 = 50. Same as before.
    #
    # The issue is that we need to handle both m and p partitions.
    #
    # Let me try: partition m=4 as 2+2, p=7 as 3+3+1.
    # For a 2-wide j-block and 3-wide k-block:
    #   A is 2x2, B is 2x3, C is 2x3
    #   C[:,0:2] = A * B[:,0:2]: Strassen 2x2x2, rank 7
    #   C[:,2] = A * B[:,2]: schoolbook 2x2x1, rank 4
    #   Total: 11
    # Number of such blocks: 2 j-blocks * 3 k-blocks = 6 blocks * 11 = 66. Worse!
    #
    # The problem is we're double-counting. We need to be more careful.
    #
    # Let me go back to basics. The tensor is T_{i,j,k} = delta_{i,j} (for the A part)
    # times delta_{j,k} (for the B part)... no, that's not right.
    #
    # The matrix multiplication tensor for (n x m) * (m x p) has entries:
    # T_{(i,j),(j',k),(i',k')} = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    # where the three indices are (row of A, col of A), (row of B, col of B), (row of C, col of C)
    # and we need i' = i, j' = j, k' = k for the product to contribute.
    #
    # Actually, the standard decomposition is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor has rank at most n*m*p (schoolbook).
    #
    # For our case: n=2, m=4, p=7, schoolbook rank = 56.
    #
    # The current solution achieves 50. Let me verify this is correct and try to improve.
    
    # Let me try a different approach: use the fact that we can combine the two
    # i-rows more cleverly.
    #
    # For each fixed j and k, we have two output entries: C[0,k] += A[0,j]*B[j,k] and
    # C[1,k] += A[1,j]*B[j,k].
    #
    # The key realization: for a fixed (j,k) pair, the two outputs share the same B[j,k].
    # We can factor this:
    # C[0,k] += A[0,j]*B[j,k]
    # C[1,k] += A[1,j]*B[j,k]
    # This is 2 multiplications in schoolbook.
    #
    # Can we do it in 1? Only if A[0,j] and A[1,j] are linearly dependent, which they're not
    # in general. So 2 is minimal for each (j,k) pair, giving 4*7*2 = 56 total.
    #
    # But Strassen shows we can do 2x2*2x2 in 7 instead of 8, saving 1 per 2x2x2 block.
    #
    # Let me try a completely different partition: m=4 as 1+1+1+1, and use
    # Strassen across the i dimension and k dimension.
    #
    # Actually, let me just try to get rank 49 by finding one more saving.
    #
    # The 2x2x1 blocks have rank 4. Can we reduce any of them to rank 3?
    # 2x2 * 2x1: C0 = A00*B0 + A01*B1, C1 = A10*B0 + A11*B1
    # This is a 2x2 matrix (the coefficients of B0 and B1) which is full rank,
    # so the tensor rank is 4 (it's the tensor product of a 2x1 vector with a 2x2 matrix).
    # So rank 4 is minimal for 2x2*2x1.
    #
    # What if we use a different block structure that avoids 2x2x1 blocks?
    #
    # Try: partition p=7 as 4+3.
    # 2x4 * 4x4: use 2x2x2 Strassen blocks (4 of them, rank 4*7=28) + cross terms?
    # No, 2x4 * 4x4 can be done as 2x(2x4) * (4x4), where 2x4*4x4 is...
    #
    # Let me try: 2x4 * 4x7.
    # Partition m=4 as 2+2, p=7 as 4+3.
    # Block (0,0): 2x2 * 2x4. This is A(2x2) * B(2x4).
    #   B(2x4) can be split into B1(2x2) + B2(2x2).
    #   C1 = A*B1: 2x2*2x2, Strassen rank 7
    #   C2 = A*B2: 2x2*2x2, Strassen rank 7
    #   Total: 14
    # Block (0,1): 2x2 * 2x3.
    #   B(2x3) split into B1(2x2) + B2(2x1).
    #   C1 = A*B1: 2x2*2x2, Strassen rank 7
    #   C2 = A*B2: 2x2*2x1, rank 4
    #   Total: 11
    # Block (1,0): 2x2 * 2x4, same as (0,0): 14
    # Block (1,1): 2x2 * 2x3, same as (0,1): 11
    # Total: 14 + 11 + 14 + 11 = 50. Still 50!
    #
    # I'm stuck at 50. Let me verify the current solution works and try a numerical
    # approach to see if rank 49 is achievable.
    
    # Let me try rank 49 with a numerical optimization approach.
    # Use a random initialization and gradient descent on the factor matrices.
    
    def try_rank_r(r, max_iter=5000, lr=0.01):
        """Try to find rank-r decomposition using numerical optimization."""
        if time.time() > deadline - 10:
            return None
        
        # Initialize with random values
        np.random.seed(42)
        U = np.random.randn(n*m, r) * 0.1
        V = np.random.randn(m*p, r) * 0.1
        W = np.random.randn(p*n, r) * 0.1
        
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Compute loss
            loss = 0.0
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for jp in range(m):
                            s = 0.0
                            for c in range(r):
                                s += U[i*m + j, c] * V[jp*p + k, c] * W[k*n + i, c]
                            target = 1.0 if j == jp else 0.0
                            loss += (s - target) ** 2
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_loss < 1e-6:
                break
            
            # Compute gradients (this is expensive, so do it efficiently)
            # dL/dU[i*m+j, c] = sum over (i,k,jp) where j is involved
            # Actually, let's use a simpler approach: compute gradients via finite differences
            # for a few random indices to save time.
            
            # This is getting too slow. Let me skip the numerical approach.
            # Instead, let me try a more structured approach.
            break
        
        if best_loss < 1e-6:
            return best_U, best_V, best_W
        return None

    # Build the rank-50 solution as before, but let me try to be smarter.
    #
    # Actually, let me try a different block structure.
    # What if we use Strassen on the (i,k) plane for each j?
    #
    # For fixed j, we have:
    # C[0,k] += A[0,j] * B[j,k] for all k
    # C[1,k] += A[1,j] * B[j,k] for all k
    #
    # This is like a 2x1 (A column) times 1x7 (B row) = 2x7 output.
    # The tensor for this is: (2x1) ⊗ (1x7) = 2x7, which has rank 2*7=14 (schoolbook).
    # Can we do it in fewer? The 2x7 matrix has rank at most 2, so the tensor
    # (which is a rank-2 matrix in the (i,k) plane for each j) has tensor rank
    # at most 2 (since it's a sum of 2 rank-1 tensors: A[0,j]*B[j,:] and A[1,j]*B[j,:]).
    # Wait, that's not quite right.
    #
    # The tensor T_j for fixed j has entries T_j[i,k] = A[i,j] * B[j,k].
    # This is a 2x7 matrix with rank at most 1 (it's outer product of A[:,j] and B[j,:]).
    # So T_j = A[:,j] ⊗ B[j,:], which is a rank-1 tensor in the (i,k) space.
    # So for each j, we need 1 "multiplication" to compute A[:,j] * B[j,:].
    # Total: 4 multiplications. But that's not right either, because we need to
    # actually compute the products.
    #
    # I think I'm overcomplicating this. Let me just go with the rank-50 solution
    # and try to verify it works.
    
    # Build rank-50 solution
    rank = 50
    U = np.zeros((n*m, rank), dtype=np.int32)
    V = np.zeros((m*p, rank), dtype=np.int32)
    W = np.zeros((p*n, rank), dtype=np.int32)
    col = 0

    def add_strassen_222(i0, j0, k0):
        nonlocal col
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)
        e = j0 * p + k0
        f = j0 * p + (k0 + 1)
        g = (j0 + 1) * p + k0
        h = (j0 + 1) * p + (k0 + 1)
        c00 = k0 * n + i0
        c01 = (k0 + 1) * n + i0
        c10 = k0 * n + (i0 + 1)
        c11 = (k0 + 1) * n + (i0 + 1)

        U[a, col] += 1; U[d, col] += 1
        V[e, col] += 1; V[h, col] += 1
        W[c00, col] += 1; W[c11, col] += 1
        col += 1

        U[c, col] += 1; U[d, col] += 1
        V[e, col] += 1
        W[c10, col] += 1; W[c11, col] -= 1
        col += 1

        U[a, col] += 1
        V[f, col] += 1; V[h, col] -= 1
        W[c01, col] += 1; W[c11, col] += 1
        col += 1

        U[d, col] += 1
        V[g, col] += 1; V[e, col] -= 1
        W[c00, col] += 1; W[c10, col] += 1
        col += 1

        U[a, col] += 1; U[b, col] += 1
        V[h, col] += 1
        W[c00, col] -= 1; W[c01, col] += 1
        col += 1

        U[c, col] += 1; U[a, col] -= 1
        V[e, col] += 1; V[f, col] += 1
        W[c11, col] += 1
        col += 1

        U[b, col] += 1; U[d, col] -= 1
        V[g, col] += 1; V[h, col] += 1
        W[c00, col] += 1
        col += 1

    def add_schoolbook_221(i0, j0, k0):
        nonlocal col
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)
        e = j0 * p + k0
        g = (j0 + 1) * p + k0
        c00 = k0 * n + i0
        c10 = k0 * n + (i0 + 1)

        U[a, col] += 1; V[e, col] += 1; W[c00, col] += 1; col += 1
        U[b, col] += 1; V[g, col] += 1; W[c00, col] += 1; col += 1
        U[c, col] += 1; V[e, col] += 1; W[c10, col] += 1; col += 1
        U[d, col] += 1; V[g, col] += 1; W[c10, col] += 1; col += 1

    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            add_strassen_222(0, j_start, k_start)
        add_schoolbook_221(0, j_start, 6)

    assert col == rank, f"Expected rank {rank}, got {col}"

    if verify(U, V, W):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }

    # Fallback to schoolbook
    U = np.zeros((n*m, n*m*p))
    V = np.zeros((m*p, n*m*p))
    W = np.zeros((p*n, n*m*p))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, idx] = 1.0
                V[j*p + k, idx] = 1.0
                W[k*n + i, idx] = 1.0
                idx += 1
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
