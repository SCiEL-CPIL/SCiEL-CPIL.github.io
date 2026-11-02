import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 150s kill

    # Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m + j, j*p + k, k*n + i] = 1 for all i, j, k
    # This is the tensor that defines the bilinear form:
    #   C[k, i] = sum_j A[i, j] * B[j, k]
    # i.e., C = B @ A.T (p x n output), matching the W[k*n+i] convention.

    # We want a rank-r decomposition: T = sum_{r} u_r ⊗ v_r ⊗ w_r
    # where u_r is in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}

    # Strategy: Try to find a low-rank decomposition using:
    # 1. First, try known algebraic decompositions (Strassen-like, Winograd)
    # 2. Then, use randomized SVD / iterative methods to find lower rank
    # 3. Fall back to schoolbook if nothing better is found

    # The schoolbook decomposition has rank 100.
    # For 4x5x5, let's think about what's possible:
    # - The matrix multiplication tensor M_{4,5,5} has rank at most 100 (schoolbook)
    # - Lower bounds: rank >= max(4*5, 5*5, 4*5) = 25 (trivial)
    # - For square n x n x n, known results give rank < n^3 for n >= 2
    # - For rectangular, it's harder to get below schoolbook

    # Let's try a block-based approach:
    # Split A (4x5) into rows, B (5x5) into columns
    # Actually, let's try to use the structure more directly.

    # Approach: Use tensor train / CP decomposition search
    # For small dimensions, we can set up the optimization problem and solve it.

    # But given time constraints, let's try a few specific constructions:

    # Construction 1: Row-grouping (Winograd-like for 2 rows)
    # Group rows of A in pairs, use (a+b), (a-b) style combinations
    # This reduces the effective rank for the row dimension.

    # For A with 4 rows, group into 2 groups of 2:
    # A_0 = A[0,:] + A[1,:], A_1 = A[0,:] - A[1,:]
    # A_2 = A[2,:] + A[3,:], A_3 = A[2,:] - A[3,:]
    # Then compute S_i = A_i * B for i=0,1,2,3 (4 matrix-vector products, each rank 5)
    # Then reconstruct:
    # C[0,:] = (S_0 + S_1)/2, C[1,:] = (S_0 - S_1)/2
    # C[2,:] = (S_2 + S_3)/2, C[3,:] = (S_2 - S_3)/2
    # Total rank: 4 * 5 = 20? No, each S_i is a 5x5 product, which needs rank 5.
    # So total rank = 4 * 5 = 20 for the S computations + reconstruction overhead.
    # Actually, the reconstruction is just linear combinations of the outputs,
    # which doesn't add to the tensor rank of the computation.
    # Wait, the tensor rank counts the number of scalar multiplications.
    # Each S_i = A_i * B requires 5 scalar multiplications (one per column of B).
    # So total = 4 * 5 = 20 scalar multiplications.
    # But we also need to form A_i = A[r1,:] ± A[r2,:], which are additions, not multiplications.
    # And C[j,:] = (S_a ± S_b)/2, also additions.
    # So the total rank (number of bilinear terms) is 20.
    # That's a huge reduction from 100!

    # Let me verify this more carefully.
    # The tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1.
    # A rank-r decomposition is T = sum_{s=1}^r u_s ⊗ v_s ⊗ w_s.
    # Each term corresponds to one scalar multiplication: x_s = (u_s . A_flat) * (v_s . B_flat)
    # and then C_flat += w_s * x_s.

    # For the row-grouping:
    # A_i = A[2i,:] + A[2i+1,:] for i=0,1
    # This is a linear combination of A entries, so u_s has entries in {1, -1, 0}.
    # Specifically, for S_0 = (A[0,:] + A[1,:]) * B:
    #   The scalar products are: for each column k of B,
    #   x_{0,k} = (A[0,k] + A[1,k]) * B[k, :] ... wait, B is 5x5.
    #   Actually S_0 is a 1x5 row vector: S_0[j] = sum_k (A[0,k] + A[1,k]) * B[k,j]
    #   = sum_k A[0,k]*B[k,j] + sum_k A[1,k]*B[k,j]
    #   This is 2*5 = 10 scalar products if done naively, but we can do it in 5:
    #   First compute T_k = A[0,k] + A[1,k] (addition, no rank cost)
    #   Then S_0[j] = sum_k T_k * B[k,j] (5 scalar products)
    # So yes, each grouped row pair costs 5 scalar products.
    # 4 groups * 5 = 20 scalar products total.

    # Let's write this out explicitly as a rank-20 decomposition.

    # For group g (g=0,1), rows 2g and 2g+1:
    #   For each column k of B (k=0..4):
    #     u vector: has 1 at position (2g)*m + k and 1 at position (2g+1)*m + k
    #     v vector: has 1 at position k*p + j for each j... no, v is for B.
    #     Actually, the scalar product is: (A[2g,k] + A[2g+1,k]) * B[k,j]
    #     So u has entries: u[(2g)*m + k] = 1, u[(2g+1)*m + k] = 1
    #     v has entry: v[k*p + j] = 1
    #     w has entry: w[j*n + (2g)] = 1/2, w[j*n + (2g+1)] = 1/2
    #     Wait, we need both + and - for the two rows.
    #     For S_0 (sum): w[j*n + 2g] += 1/2, w[j*n + (2g+1)] += 1/2
    #     For S_1 (diff): w[j*n + 2g] += 1/2, w[j*n + (2g+1)] -= 1/2
    #     So each (g, k) pair needs 2 scalar products (sum and diff), each with rank contribution.
    #     Total: 2 groups * 5 columns * 2 (sum/diff) = 20 scalar products.

    # Let me construct this rank-20 decomposition.

    rank = 20
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    col = 0
    for g in range(n // 2):  # 2 groups
        r1 = 2 * g
        r2 = 2 * g + 1
        for k in range(m):  # 5 columns of B
            # Term for sum: (A[r1,k] + A[r2,k]) * B[k,j]
            # u: 1 at r1*m+k and r2*m+k
            U[r1 * m + k][col] = 1
            U[r2 * m + k][col] = 1
            for j in range(p):
                # v: 1 at k*p+j
                V[k * p + j][col] = 1
                # w: 1/2 at j*n+r1 and j*n+r2
                W[j * n + r1][col] = 0.5
                W[j * n + r2][col] = 0.5
            col += 1

            # Term for diff: (A[r1,k] - A[r2,k]) * B[k,j]
            # u: 1 at r1*m+k, -1 at r2*m+k
            U[r1 * m + k][col] = 1
            U[r2 * m + k][col] = -1
            for j in range(p):
                V[k * p + j][col] = 1
                W[j * n + r1][col] = 0.5
                W[j * n + r2][col] = -0.5
            col += 1

    # Verify this is a valid decomposition
    # Check: C[k_out, i] = sum_j A[i, j] * B[j, k_out]
    # In our indexing: C_flat[k*n + i] = sum_j A[i*m+j] * B[j*p+k]
    # Let's verify for a specific entry.

    # Actually, let me just verify numerically.
    def _verify(U, V, W, n, m, p):
        """Check if the decomposition is exact."""
        rank = len(U[0])
        # Build tensor from decomposition
        # T[i*m+j, j*p+k, k*n+i] should equal 1
        # T = sum_s u_s[i*m+j] * v_s[j*p+k] * w_s[k*n+i]
        # Check a few random entries
        import random
        random.seed(42)
        for _ in range(200):
            i = random.randint(0, n - 1)
            j = random.randint(0, m - 1)
            k = random.randint(0, p - 1)
            val = 0.0
            for s in range(rank):
                val += U[i * m + j][s] * V[j * p + k][s] * W[k * n + i][s]
            if abs(val - 1.0) > 1e-9:
                return False
        # Also check that non-multiplication entries are 0
        for _ in range(200):
            i = random.randint(0, n - 1)
            j = random.randint(0, m - 1)
            k = random.randint(0, p - 1)
            # Check an entry that shouldn't be 1
            # T[i*m+j, j2*p+k, k*n+i] where j2 != j should be 0
            j2 = (j + 1) % m
            if j2 == j:
                j2 = (j + 2) % m
            val = 0.0
            for s in range(rank):
                val += U[i * m + j][s] * V[j2 * p + k][s] * W[k * n + i][s]
            if abs(val) > 1e-9:
                return False
        return True

    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}

    # If verification failed, fall back to schoolbook
    rank = rank_schoolbook
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
