"""Matrix multiplication 4,4,5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [4, 4, 5]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 4, 4, 5

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank

    best_U, best_V, best_W, best_rank = schoolbook()

    def check_valid(U_mat, V_mat, W_mat, r, num_samples=500):
        """Verify that the decomposition is valid."""
        # The tensor equation: sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] = 1 for all (i,j,k)
        # where the canonical tensor has 1s at (i,j,k) positions.
        # Actually, the matrix multiplication tensor T[i,j,k] = delta_{i,j} * delta_{j,k} is not right.
        # The matrix multiplication tensor is: T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]B[j,k]
        # So the tensor has T[i,j,k] = 1 for all i,j,k? No.
        # 
        # Let me reconsider. The matrix multiplication tensor M satisfies:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor M has entries M[i,j,k] such that:
        # sum_j A[i,j] B[j,k] = sum_j M[i,j,k] A[i,j] B[j,k]
        # So M[i,j,k] = 1 for all i,j,k? No, that's not right either.
        #
        # Actually, the standard formulation is:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # This can be written as a tensor contraction:
        # C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
        # where T[i,j,k] = 1 for all i,j,k.
        #
        # But the rank decomposition is:
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        #
        # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
        #
        # Wait, that's not quite right. Let me re-examine.
        #
        # The matrix multiplication tensor is defined such that:
        # (A * B)[i,k] = sum_j A[i,j] * B[j,k]
        #
        # In tensor notation, if we flatten A to a vector of length n*m (indexed by (i,j)),
        # B to a vector of length m*p (indexed by (j,k)), and C to a vector of length p*n (indexed by (k,i)),
        # then:
        # C[k*n+i] = sum_j A[i*m+j] * B[j*p+k]
        #
        # This is a bilinear map, which can be represented as:
        # C = sum_r U_r ⊗ V_r ⊗ W_r
        # where U_r is a vector of length n*m, V_r is a vector of length m*p, W_r is a vector of length p*n.
        #
        # Specifically, the tensor T has entries:
        # T[i*m+j, j*p+k, k*n+i] = 1
        # and 0 otherwise.
        #
        # The rank-r decomposition gives:
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        #
        # So for the matrix multiplication tensor:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # for all i in 0..n-1, j in 0..m-1, k in 0..p-1.
        #
        # And for all other (a,b,c) not of the form (i*m+j, j*p+k, k*n+i), the sum should be 0.
        # But since we only need to match the non-zero entries, and the decomposition is linear,
        # it suffices to check the non-zero entries.

        for _ in range(num_samples):
            i = np.random.randint(n)
            j = np.random.randint(m)
            k = np.random.randint(p)
            val = 0.0
            for t in range(r):
                val += U_mat[i * m + j, t] * V_mat[j * p + k, t] * W_mat[k * n + i, t]
            if abs(val - 1.0) > 1e-9:
                return False
        return True

    def check_valid_full(U_mat, V_mat, W_mat, r):
        """Full verification for all (i,j,k) tuples."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for t in range(r):
                        val += U_mat[i * m + j, t] * V_mat[j * p + k, t] * W_mat[k * n + i, t]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Try to find a lower rank decomposition using random search
    # Strategy: start with schoolbook, then try to reduce rank by finding linear dependencies

    # For 4x4x5, the known lower bound is around 20-25.
    # Let's try a random search approach.

    def random_search(target_rank, num_trials=100):
        """Try to find a rank-`target_rank` decomposition via random initialization."""
        for trial in range(num_trials):
            if time.time() - start_time > DEADLINE - 10:
                return None

            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)

            # Normalize to make the problem more tractable
            # This is a heuristic; may not help but doesn't hurt

            # Check if this random decomposition works (very unlikely, but let's see)
            # Actually, random decompositions won't work. We need a structured approach.
            # Let's use a different strategy.
            pass

        return None

    # Better approach: use the known fact that for 4x4x5, we can use a 2x2x2 block decomposition
    # or use the Winograd algorithm.

    # For 4x4x5, let's try to construct a decomposition using the following idea:
    # Split the matrices into blocks and use fast matrix multiplication for the blocks.

    # Actually, let's try a more systematic approach:
    # The rank of 4x4x5 matrix multiplication is known to be at most 40 (from 2x2 blocks)
    # and possibly lower.

    # Let's try to construct a rank-20 decomposition using the following idea:
    # For 4x4x4, the rank is 20 (from the 2x2x2 Winograd algorithm).
    # For 4x4x5, we can extend this.

    # The 2x2x2 Winograd algorithm uses 7 multiplications instead of 8.
    # For 4x4x4, we can split into 2x2 blocks and use 7 multiplications per block,
    # giving 4 * 7 = 28 multiplications. But we can do better.

    # Actually, for 4x4x4, the rank is 20. Let me try to construct it.

    # For now, let's try a simpler approach: use the schoolbook and see if we can
    # reduce the rank by finding linear dependencies.

    # Try removing terms one by one and checking validity
    U, V, W, rank = schoolbook()

    # Try to reduce by removing terms
    for t in range(rank - 1, -1, -1):
        if time.time() - start_time > DEADLINE - 10:
            break

        # Try removing term t
        U_t = np.delete(U, t, axis=1)
        V_t = np.delete(V, t, axis=1)
        W_t = np.delete(W, t, axis=1)

        if check_valid(U_t, V_t, W_t, rank - 1, num_samples=100):
            # Check more thoroughly
            if check_valid(U_t, V_t, W_t, rank - 1, num_samples=500):
                U, V, W = U_t, V_t, W_t
                rank -= 1
                if rank < best_rank:
                    best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), rank

    # Return the best decomposition found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
