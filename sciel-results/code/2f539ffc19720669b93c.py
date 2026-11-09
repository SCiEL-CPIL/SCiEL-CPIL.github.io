"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # 110s limit for construct()

    def make_seed():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def validate(U, V, W):
        """Check if U, V, W form a valid tensor decomposition."""
        r = len(U[0])
        Ua = np.array(U, dtype=np.float64)
        Va = np.array(V, dtype=np.float64)
        Wa = np.array(W, dtype=np.float64)

        # Check that entries are integer or half-integer
        Ua2 = Ua * 2
        Va2 = Va * 2
        Wa2 = Wa * 2
        if not np.allclose(Ua2, np.round(Ua2), atol=1e-6):
            return False
        if not np.allclose(Va2, np.round(Va2), atol=1e-6):
            return False
        if not np.allclose(Wa2, np.round(Wa2), atol=1e-6):
            return False

        Ur = Ua.reshape(n, m, r)
        Vr = Va.reshape(m, p, r)
        Wr = Wa.reshape(p, n, r)
        T = np.einsum('ijl,jkl,kin->ijk', Ur, Vr, Wr)

        if not np.allclose(T, np.ones((n, m, p)), atol=1e-4):
            return False
        return True

    def check_half_integer(a, tol=1e-6):
        a2 = a * 2
        return np.allclose(a2, np.round(a2), atol=tol)

    def compute_T(U, V, W):
        Ur = U.reshape(n, m, -1)
        Vr = V.reshape(m, p, -1)
        Wr = W.reshape(p, n, -1)
        return np.einsum('ijl,jkl,kin->ijk', Ur, Vr, Wr)

    # Start with the trivial rank-90 decomposition
    seed = make_seed()
    best_result = seed
    best_rank = n * m * p

    # Strategy: Try to find a rank-89 decomposition by modifying the trivial one.
    # The trivial decomposition has U[i*m+j, i*m*p+j*p+k] = 1, etc.
    # We need to find a linear dependency that allows us to remove one term.
    # 
    # Alternative: Use a structured search. For rank r, we have:
    # sum_{l=1}^{r} U[i*m+j,l] * V[j*p+k,l] * W[k*n+i,l] = 1 for all (i,j,k)
    #
    # Let's try a local search: start from rank 90, try to "merge" two terms
    # into one by finding coefficients that preserve the decomposition.
    #
    # More practically: try random rank-89 decompositions with a smarter initialization.
    # Initialize most entries to 0, and set up a sparse structure.

    values = [-1.0, -0.5, 0.0, 0.5, 1.0]
    rng = np.random.default_rng(42)

    # Try rank 89 with various strategies
    for rank in [89, 88]:
        if time.time() > deadline - 30:
            break

        found = False

        # Strategy 1: Random sparse initialization
        # Each row has a few non-zero entries
        for attempt in range(500):
            if time.time() > deadline - 15:
                break

            # Create sparse random matrices
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))

            # For each rank column, assign non-zero values sparsely
            for l in range(rank):
                # Pick a few rows in U, V, W to be non-zero
                u_rows = rng.choice(n * m, size=min(3, n * m), replace=False)
                v_rows = rng.choice(m * p, size=min(3, m * p), replace=False)
                w_rows = rng.choice(p * n, size=min(3, p * n), replace=False)

                for r_idx in u_rows:
                    U[r_idx, l] = rng.choice(values)
                for r_idx in v_rows:
                    V[r_idx, l] = rng.choice(values)
                for r_idx in w_rows:
                    W[r_idx, l] = rng.choice(values)

            T = compute_T(U, V, W)
            if np.allclose(T, np.ones((n, m, p)), atol=1e-4):
                if check_half_integer(U) and check_half_integer(V) and check_half_integer(W):
                    U_list = [[float(x) for x in row] for row in U]
                    V_list = [[float(x) for x in row] for row in V]
                    W_list = [[float(x) for x in row] for row in W]
                    if validate(U_list, V_list, W_list):
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = rank
                        found = True
                        break

        if found:
            break

        # Strategy 2: Try to modify the trivial decomposition
        # Remove one term and try to compensate with adjustments to other terms
        if not found and rank == 89:
            # The trivial decomposition: term (i,j,k) has U[i*m+j, idx]=1, V[j*p+k, idx]=1, W[k*n+i, idx]=1
            # where idx = i*m*p + j*p + k
            # Try removing term (i0,j0,k0) and adjusting others
            for i0 in range(n):
                if time.time() > deadline - 20:
                    break
                for j0 in range(m):
                    if time.time() > deadline - 20:
                        break
                    for k0 in range(p):
                        if time.time() > deadline - 20:
                            break

                        # Start with trivial rank-90, remove one term
                        U = np.zeros((n * m, rank))
                        V = np.zeros((m * p, rank))
                        W = np.zeros((p * n, rank))

                        col = 0
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    if i == i0 and j == j0 and k == k0:
                                        continue  # skip this term
                                    U[i * m + j, col] = 1
                                    V[j * p + k, col] = 1
                                    W[k * n + i, col] = 1
                                    col += 1

                        # Now we have rank-89 with one term missing.
                        # The missing term contributes 1 to T[i0,j0,k0] and 0 elsewhere.
                        # We need to adjust the remaining terms to compensate.
                        # This is hard in general, but let's try small perturbations.

                        # Check current state
                        T = compute_T(U, V, W)
                        # T[i0,j0,k0] should be 0 (missing), others should be 1
                        # We need to add 1 to T[i0,j0,k0] without changing other entries.

                        # Try adding half-integer adjustments to some terms
                        # This is a complex constraint satisfaction problem.
                        # For now, just check if by chance it works (it won't, but the structure is set up)
                        pass

    return best_result
# EVOLVE-BLOCK-END
