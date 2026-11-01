"""Matrix multiplication 2x5x5: rank-minimal exact tensor decomposition."""
import time
import itertools
import random
import numpy as np

DIMENSIONS = [2, 5, 5]
N, M, P = DIMENSIONS
RANK_TARGET = N * M * P  # 50 schoolbook


def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x5 by 5x5 matrix multiplication.
    Returns {'U': (n*m, rank), 'V': (m*p, rank), 'W': (p*n, rank)}.
    """
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    n, m, p = N, M, P
    nm, mp, pn = n * m, m * p, p * n

    # Build the matrix multiplication tensor T[i,j,k] = 1 if (i,j,k) is a valid index
    # T is a 3D array of shape (n, m, p) where T[i,j,k] = 1
    # The tensor to decompose: for each (i,j,k), we need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # And for all other combinations (which don't exist in the tensor), they're 0.
    # Actually, the matrix multiplication tensor has entries:
    # T_{(i,j),(j,k),(k,i)} = 1 for all valid i,j,k
    # All other entries are 0.
    #
    # The decomposition requires:
    # For all i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    #
    # We use Strassen-like block decomposition.
    #
    # Strategy: Use the known Strassen algorithm structure for 2x2 blocks.
    # For 2x5 x 5x5, we can split the 5x5 into blocks.
    #
    # Let's use a direct approach: enumerate all possible rank-r decompositions
    # using integer/half-integer coefficients, trying to minimize rank.
    #
    # Actually, let's use a smarter approach based on known results.
    # The tensor rank of 2x5x5 matrix multiplication:
    # - Schoolbook: 50
    # - Using Strassen on 2x2 blocks: can save some rank
    #
    # Let me think about this differently. We'll use a randomized/local search approach
    # to find a valid decomposition at various ranks, starting from low ranks.

    # First, let's try to construct a decomposition using block Strassen.
    # Split the 5x5 matrix into blocks:
    # A is 2x5, B is 5x5.
    # We can think of B as having rows split and columns split.
    #
    # Alternative: Use the fact that we can apply Strassen's 7-multiplication algorithm
    # to 2x2 sub-blocks.
    #
    # Let's try a more systematic approach: use integer linear programming or
    # exhaustive search over small coefficient sets.
    #
    # For a practical approach within time limits, let's use the following:
    # 1. Start with the schoolbook decomposition (rank 50).
    # 2. Try to merge terms using Strassen-like identities.
    # 3. Use local search to reduce rank.

    # Build schoolbook decomposition
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W, rank

    # Verify a decomposition
    def verify(U, V, W):
        rank = len(U[0]) if U else 0
        # Check all (i,j,k) triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # Convert to numpy for faster verification
    def verify_np(U, V, W):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        rank = U_np.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_np[i * m + j]
                    v_val = V_np[j * p + k]
                    w_val = W_np[k * n + i]
                    s = np.dot(np.multiply(u_val, v_val), w_val)
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Try to find a decomposition using Strassen-inspired block structure.
    # 
    # Key insight: For 2x2x2 matrix multiplication, Strassen uses rank 7 instead of 8.
    # We can apply this to sub-blocks of our problem.
    #
    # Let's try a structured approach:
    # Split the 5 columns of A (and rows of B) into groups.
    # Split the 5 rows of B (and cols of B) into groups.
    #
    # Actually, let's use a more direct algebraic approach.
    #
    # The tensor T has 50 nonzero entries (one for each (i,j,k) with i<2, j<5, k<5).
    # We want to write T = sum_{r=1}^{R} u_r ⊗ v_r ⊗ w_r
    # where u_r is in R^{nm}, v_r in R^{mp}, w_r in R^{pn}.
    #
    # Each term u_r ⊗ v_r ⊗ w_r contributes to entry (i,j,k) the value
    # u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i].
    #
    # For the decomposition to be exact, for each (i,j,k):
    # sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] = 1
    #
    # Let's try rank values from low to high, using a greedy/local search.

    # Use a randomized approach to find low-rank decompositions.
    # For each candidate rank R, we try to find U, V, W with entries in {-1, -0.5, 0, 0.5, 1, 2, ...}
    # such that the tensor equation is satisfied.

    # Better approach: Use the structure of Strassen's algorithm.
    # 
    # For 2x2 matrices, Strassen computes:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    #
    # This gives rank 7 for 2x2x2.
    #
    # For our 2x5x5 problem, let's think of it as:
    # A is 2x5, B is 5x5, C is 2x5.
    #
    # We can partition the 5 columns of A into: first 2 and last 3.
    # Similarly partition the 5 rows/cols of B.
    #
    # Let's try a different approach: use the fact that we can express
    # the multiplication using fewer than 50 scalar multiplications.
    #
    # Known result: The tensor rank of <2,5,5> is at most 42 (using block Strassen).
    # Let me try to construct this explicitly.
    #
    # Partition: A = [A1 A2] where A1 is 2x2, A2 is 2x3.
    # B = [[B11, B12], [B21, B22]] where B11 is 2x2, B12 is 2x3, B21 is 3x2, B22 is 3x3.
    #
    # C1 = A1*B11 + A2*B21 (2x2)
    # C2 = A1*B12 + A2*B22 (2x3)
    #
    # For C1 (2x2 = 2x2 * 2x2): use Strassen, rank 7.
    # For C2 (2x3 = 2x2 * 2x3 + 2x3 * 3x3):
    #   A1*B12: 2x2 * 2x3, schoolbook rank 12, or use Strassen-like
    #   A2*B22: 2x3 * 3x3, schoolbook rank 18
    #
    # This is getting complex. Let me just try a computational search.

    # Computational approach: Try to find a decomposition at various ranks.
    # Use the fact that entries can be small integers or half-integers.
    #
    # For efficiency, let's use a greedy construction followed by local search.

    # Actually, let's use a well-known result: 
    # The rank of <2,5,5> matrix multiplication tensor.
    # 
    # Let me try a direct construction using the "laser method" or known bounds.
    #
    # For practical purposes within the time limit, let me try:
    # 1. Rank 42 (known upper bound from block Strassen)
    # 2. Try to verify and if not, increase rank.

    # Let me try to construct a rank-42 decomposition explicitly.
    # 
    # Use the following block structure:
    # A (2x5) = [A_11 (2x2) | A_12 (2x3)]
    # B (5x5) = [B_11 (2x2)  B_12 (2x3); B_21 (3x2)  B_22 (3x3)]
    # C (2x5) = [C_11 (2x2) | C_12 (2x3)]
    #
    # C_11 = A_11 * B_11 + A_12 * B_21  (2x2)
    # C_12 = A_11 * B_12 + A_12 * B_22  (2x3)
    #
    # For C_11: This is a sum of two 2x2 matrix products.
    # A_11*B_11: rank 8 schoolbook, rank 7 Strassen
    # A_12*B_21: 2x3 * 3x2, rank 12 schoolbook
    #
    # This partitioning doesn't directly help. Let me try a different approach.
    #
    # Let me just do a computational search with numpy for speed.

    # Try ranks from 40 upward, using a randomized local search.
    # The search space is huge, so we need a good initialization.

    # Better: use the structure of the problem.
    # Each rank-1 term (u, v, w) contributes to the tensor.
    # We need the sum of contributions to equal 1 at each (i,j,k) position.
    #
    # Let's use a greedy approach:
    # 1. Start with 0 terms.
    # 2. At each step, find the term (u,v,w) that best reduces the residual.
    # 3. Continue until the residual is zero (or close enough).

    # But this is NP-hard in general. Let's use a different strategy.
    #
    # Strategy: Use the known Strassen decomposition for 2x2 blocks and
    # combine with schoolbook for the rest.
    #
    # Let me try to be more systematic. I'll use the following construction:
    #
    # Think of the 5x5 matrix B as having its 5 rows and 5 columns.
    # The 2x5 matrix A has 2 rows and 5 columns.
    #
    # We can group the 5 columns into: {0,1} and {2,3,4}.
    # Similarly for the 5 rows of B.
    #
    # Block decomposition:
    # A = [A00 (2x2)  A01 (2x3)]
    # B = [B00 (2x2)  B01 (2x3); B10 (3x2)  B11 (3x3)]
    #
    # C00 = A00*B00 + A01*B10  (2x2)
    # C01 = A00*B01 + A01*B11  (2x3)
    #
    # Now, A00*B00 is a 2x2*2x2 product: use Strassen (rank 7).
    # A01*B10 is a 2x3*3x2 product: schoolbook rank 12.
    # A00*B01 is a 2x2*2x3 product: schoolbook rank 12.
    # A01*B11 is a 2x3*3x3 product: schoolbook rank 18.
    #
    # Total: 7 + 12 + 12 + 18 = 49. Not better than 50.
    #
    # Let me try a different grouping.
    #
    # Group columns as: {0,1,2} and {3,4}.
    # A = [A00 (2x3)  A01 (2x2)]
    # B = [B00 (3x3)  B01 (3x2); B10 (2x3)  B11 (2x2)]
    #
    # C00 = A00*B00 + A01*B10  (2x3)
    # C01 = A00*B01 + A01*B11  (2x2)
    #
    # A00*B00: 2x3*3x3, rank 18
    # A01*B10: 2x2*2x3, rank 12
    # A00*B01: 2x3*3x2, rank 12
    # A01*B11: 2x2*2x2, rank 7 (Strassen)
    #
    # Total: 18+12+12+7 = 49. Still not better.
    #
    # The issue is that we're not exploiting the structure well enough.
    #
    # Let me try yet another approach: use the fact that for the 2x2 case,
    # Strassen saves 1 rank. If we can apply Strassen to multiple 2x2 sub-problems
    # simultaneously, we can save more.
    #
    # Actually, the known result for <2,5,5> might be rank 42 or similar.
    # Let me just try to find it computationally.
    #
    # Given the time constraints, let me use a practical approach:
    # 1. Generate a random initialization for U, V, W at a given rank.
    # 2. Use gradient-based optimization to minimize the error.
    # 3. Snap to nearest half-integers.
    # 4. Verify.
    # 5. If successful, try lower ranks.

    # This is a research-level problem. Let me use a simpler but effective approach:
    # Use the schoolbook decomposition and try to eliminate terms using
    # linear dependencies.

    # Actually, for the time limit, let me just return the schoolbook decomposition
    # which is guaranteed to be correct, and then try to improve it if time permits.

    # Let me try a more clever approach: use the "laser" method / Strassen's method.
    #
    # The key idea: if we can find a set of rank-1 tensors whose sum is the
    # matrix multiplication tensor, we're done.
    #
    # For <2,5,5>, let me try to use the following construction:
    #
    # We know that <2,2,2> has rank 7 (Strassen).
    # We can embed this into our problem.
    #
    # Let me try a specific construction:
    #
    # Label the 50 entries of the tensor by (i,j,k) where i∈{0,1}, j∈{0,...,4}, k∈{0,...,4}.
    #
    # I'll use a computer search approach with the following strategy:
    # - Try rank R from 40 to 50.
    # - For each R, use a randomized approach to find U, V, W.
    # - Use the fact that the system of equations is:
    #   For each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    #
    # This is a system of 50 equations in 3*R*50 variables (but with structure).
    #
    # Given the complexity, let me just implement a practical search.

    # Practical implementation:
    # Use numpy for fast tensor operations.
    # Try to find a decomposition at various ranks using local search.

    def try_rank(R, max_iters=5000, seed=42):
        """Try to find a rank-R decomposition using local search."""
        if time.time() > deadline - 10:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Initialize with random values in {-1, 0, 1}
        U = rng.randint(-1, 2, size=(nm, R)).astype(float)
        V = rng.randint(-1, 2, size=(mp, R)).astype(float)
        W = rng.randint(-1, 2, size=(pn, R)).astype(float)
        
        # Compute the tensor product sum
        def compute_residual(U, V, W):
            # For each (i,j,k), compute sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
            residual = np.zeros((n, m, p), dtype=float)
            for r in range(R):
                u_col = U[:, r]  # shape (nm,)
                v_col = V[:, r]  # shape (mp,)
                w_col = W[:, r]  # shape (pn,)
                for i in range(n):
                    for j in range(m):
                        u_val = u_col[i * m + j]
                        if u_val == 0:
                            continue
                        for k in range(p):
                            v_val = v_col[j * p + k]
                            w_val = w_col[k * n + i]
                            residual[i, j, k] += u_val * v_val * w_val
            return residual - 1.0  # We want this to be 0
        
        # Local search: randomly perturb one entry and accept if it reduces error
        best_error = None
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Compute current error
            res = compute_residual(U, V, W)
            error = np.sum(res ** 2)
            
            if best_error is None or error < best_error:
                best_error = error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if error < 1e-12:
                break
            
            # Random perturbation
            mat = rng.randint(0, 3)  # 0=U, 1=V, 2=W
            if mat == 0:
                row = rng.randint(0, nm)
                col = rng.randint(0, R)
                delta = rng.choice([-1, 1])
                U[row, col] += delta
            elif mat == 1:
                row = rng.randint(0, mp)
                col = rng.randint(0, R)
                delta = rng.choice([-1, 1])
                V[row, col] += delta
            else:
                row = rng.randint(0, pn)
                col = rng.randint(0, R)
                delta = rng.choice([-1, 1])
                W[row, col] += delta
        
        # Check if we found a solution
        if best_error < 1e-9:
            # Convert to lists
            return (best_U.tolist(), best_V.tolist(), best_W.tolist(), R)
        return None

    # Try to find a decomposition at various ranks
    best_result = None
    
    # Try ranks from 40 to 50
    for R in range(40, 51):
        if time.time() > deadline - 20:
            break
        
        # Try multiple random seeds
        for seed in range(5):
            if time.time() > deadline - 20:
                break
            
            result = try_rank(R, max_iters=3000, seed=seed + R * 100)
            if result is not None:
                U, V, W, rank = result
                # Verify
                if verify_np(U, V, W):
                    best_result = (U, V, W, rank)
                    break
        
        if best_result is not None:
            break
    
    # If we didn't find anything better, return schoolbook
    if best_result is None:
        U, V, W, rank = schoolbook()
    else:
        U, V, W, rank = best_result
    
    return {"U": U, "V": V, "W": W}
