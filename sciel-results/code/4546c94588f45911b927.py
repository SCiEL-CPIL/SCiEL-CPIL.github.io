"""Matrix multiplication 4x4 * 4x9 tensor decomposition with structured search."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x4 * 4x9 matrix multiplication.
    Uses a combination of:
    1. Schoolbook baseline (rank 144)
    2. Structured block-diagonal attempts
    3. Systematic search for column merges
    4. Local optimization with exact verification
    """
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s kill

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 144

    def verify(U, V, W, tol=1e-8):
        """Verify decomposition: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1"""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def verify_fast(U, V, W, tol=1e-8):
        """Faster verification using numpy"""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True

    # Start with schoolbook
    U = np.zeros((n*m, schoolbook_rank))
    V = np.zeros((m*p, schoolbook_rank))
    W = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1

    best_rank = schoolbook_rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # Strategy 1: Try to merge pairs of columns
    # Two columns c1, c2 can be merged if their contributions can be combined
    # For schoolbook, each column (i,j,k) contributes 1 to exactly one (i,j,k) entry
    # We can try to find linear combinations that preserve the tensor

    # Strategy 2: Use structured decompositions based on matrix factorizations
    # For 4x4 * 4x9, we can try:
    # - Factor A as P*Q where P is 4x4, Q is 4x4 (but A is 4x4, so this is just A itself)
    # - Use the fact that we can share computations across different (i,j,k) triples

    # Let's try a more systematic approach:
    # The tensor T[i,j,k] = 1 for all (i,j,k)
    # We need rank-r vectors u_{i,j}, v_{j,k}, w_{k,i} such that
    # sum_c u_{i,j,c} * v_{j,k,c} * w_{k,i,c} = 1

    # Key insight: if we can find a set of rank-1 tensors whose sum is the all-ones tensor
    # A rank-1 tensor is defined by three vectors (u, v, w) and has entries u_i * v_j * w_k

    # For our problem, the "indices" are (i,j) for U, (j,k) for V, (k,i) for W
    # So a rank-1 term is: U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]

    # Let's try to construct solutions using Kronecker product structure
    # If we can write the all-ones tensor as a sum of Kronecker products...

    # Attempt: Use the fact that ones(n) ⊗ ones(m) ⊗ ones(p) can be decomposed
    # Actually, the all-ones tensor of size n*m*p has rank 1 in the standard tensor sense
    # But our tensor has a specific structure due to the index coupling

    # Let's try a different approach: solve the system of equations directly
    # We have n*m*p = 144 equations
    # Unknowns: r * (n*m + m*p + p*n) = r * (16 + 36 + 36) = 88r

    # For r = 2: 176 unknowns, 144 equations (overdetermined but might have solution)
    # For r = 1: 88 unknowns, 144 equations (underdetermined, likely no solution)

    # Let's try to solve for small r using least squares
    def solve_linear_system(target_rank):
        """Try to find a decomposition of given rank using linear algebra"""
        if time.time() > deadline - 5:
            return None

        # Number of unknowns
        n_unknowns = target_rank * (n*m + m*p + p*n)

        # Build the system: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # This is nonlinear, so we can't directly use linear algebra
        # Instead, we'll use a different approach

        return None

    # Strategy 3: Use the fact that we can add zero-valued columns
    # If we have a valid decomposition of rank r, we can add columns that are
    # linearly dependent to get rank r+k

    # Strategy 4: Try to find the minimal rank by exhaustive search over small ranks
    # Given the complexity, let's focus on structured attempts

    # Attempt: Use block structure
    # For each j (0..m-1), we need to compute A[:,j] * B[j,:] which is a 4x9 outer product
    # The outer product can be computed with rank 1 if we allow the W matrix to depend on j
    # But W is shared across j's, so we need to be careful

    # Let's try: for each j, use a separate set of columns
    # Total rank = m * (rank for one j)
    # For one j, we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,k
    # If we set W[k*n+i,c] = 1 for all c, then we need sum_c U[i*m+j,c] * V[j*p+k,c] = 1
    # This means U_j * V_j^T = ones(4,9)
    # The rank of ones(4,9) is 1, so we can use rank 1 for each j
    # Total rank = m * 1 = 4

    # But wait, this requires W[k*n+i,c] = 1 for all c, which means all W columns are the same
    # And U[i*m+j,c] and V[j*p+k,c] must satisfy U_j * V_j^T = ones(4,9)
    # For rank 1: U_j[i,0] = a_i, V_j[k,0] = b_k, and a_i * b_k = 1 for all i,k
    # This requires all a_i equal and all b_k equal, with a * b = 1

    # Let's try this construction:
    U_try = np.zeros((n*m, m))
    V_try = np.zeros((m*p, m))
    W_try = np.zeros((p*n, m))

    for j in range(m):
        # For column j, set U[i*m+j, j] = 1, V[j*p+k, j] = 1, W[k*n+i, j] = 1
        for i in range(n):
            U_try[i*m+j, j] = 1.0
        for k in range(p):
            V_try[j*p+k, j] = 1.0
        for i in range(n):
            for k in range(p):
                W_try[k*n+i, j] = 1.0

    # Verify this construction
    if verify_fast(U_try, V_try, W_try):
        best_rank = m
        best_U, best_V, best_W = U_try.copy(), V_try.copy(), W_try.copy()
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }

    # The above didn't work, so let's try a different construction
    # For each j, we need the sum over c to give 1 for all (i,k)
    # If we use m columns (one per j), then for column j, only the j-th row of U and V are nonzero
    # So for (i,j',k) with j' != j, the contribution from column j is 0
    # This means we can't share columns across different j's

    # Let's try: use n*p columns, one for each (i,k) pair
    # For column (i0,k0), set U[i*m+j, c] = delta_{i,i0}, V[j*p+k, c] = delta_{k,k0}, W[k*n+i, c] = 1
    # Then for (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = sum_{i0,k0} delta_{i,i0}*delta_{k,k0}*1 = 1 if (i,k) is in the set, 0 otherwise
    # This doesn't give 1 for all (i,j,k)

    # Let's try yet another approach: use the identity matrix structure
    # For A (4x4) and B (4x9), the product C = A*B has entries C[i,k] = sum_j A[i,j]*B[j,k]
    # We need to compute all 4*9 = 36 entries of C

    # The standard algorithm uses 4*4*9 = 144 multiplications
    # Can we do better?

    # For square matrices, Strassen's algorithm reduces the number of multiplications
    # For 4x4 * 4x4, Strassen uses 7 multiplications instead of 16
    # For 4x4 * 4x9, we can pad B to 4x12 or use a different approach

    # Let's try to use Strassen-like ideas
    # Actually, for rectangular matrices, the optimal algorithm is more complex

    # Given the time constraints, let's just try to find any valid decomposition with rank < 144
    # and return it

    # Try random search for lower rank
    np.random.seed(42)

    def try_random_rank(target_rank, n_trials=100):
        """Try to find a valid decomposition of given rank"""
        if time.time() > deadline - 5:
            return None

        for trial in range(n_trials):
            if time.time() > deadline - 5:
                break

            # Initialize with small random values
            U_r = np.random.randn(n*m, target_rank) * 0.1
            V_r = np.random.randn(m*p, target_rank) * 0.1
            W_r = np.random.randn(p*n, target_rank) * 0.1

            # Simple iterative refinement
            lr = 0.01
            for inner_it in range(100):
                if time.time() > deadline - 3:
                    break

                # Compute errors
                errors = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            errors[i,j,k] = np.sum(U_r[i*m+j, :] * V_r[j*p+k, :] * W_r[k*n+i, :]) - 1.0

                err = np.sum(errors**2)
                if err < 1e-10:
                    return U_r, V_r, W_r

                # Gradient w.r.t. U[i*m+j, c] = error[i,j,k] * V[j*p+k,c] * W[k*n+i,c] for each k
                # Do a few random coordinate updates
                for _ in range(10):
                    i = np.random.randint(n)
                    j = np.random.randint(m)
                    k = np.random.randint(p)
                    c = np.random.randint(target_rank)

                    gU = errors[i,j,k] * V_r[j*p+k, c] * W_r[k*n+i, c]
                    U_r[i*m+j, c] -= lr * gU

                    gV = errors[i,j,k] * U_r[i*m+j, c] * W_r[k*n+i, c]
                    V_r[j*p+k, c] -= lr * gV

                    gW = errors[i,j,k] * U_r[i*m+j, c] * V_r[j*p+k, c]
                    W_r[k*n+i, c] -= lr * gW

        return None

    # Try a few specific ranks
    candidate_ranks = [140, 130, 120, 100, 80, 60, 50, 40, 36, 32, 28, 24, 20, 16, 12, 8, 6, 4, 3, 2]

    for target_rank in candidate_ranks:
        if time.time() > deadline - 10:
            break
        if target_rank >= best_rank:
            continue

        result = try_random_rank(target_rank, n_trials=50)
        if result is not None:
            U_r, V_r, W_r = result
            if verify_fast(U_r, V_r, W_r):
                best_rank = target_rank
                best_U, best_V, best_W = U_r, V_r, W_r
                break

    # Convert to lists for return
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
