"""Program for M_4,4,6 tensor rank; attempts structured factorisation with verification."""
DIMENSIONS = [4, 4, 6]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """Return a valid exact tensor decomposition with minimal rank found within budget."""
    start_wall = time.time()
    DEADLINE = 150.0
    MARGIN = 15.0  # safety margin seconds

    n, m, p = 4, 4, 6
    U_dim, V_dim, W_dim = n * m, m * p, p * n  # 16, 24, 24

    # Schoolbook rank (baseline)
    R_schoolbook = n * m * p  # 96

    def make_schoolbook(r=None):
        r = r if r is not None else R_schoolbook
        U = [[0]*r for _ in range(U_dim)]
        V = [[0]*r for _ in range(V_dim)]
        W = [[0]*r for _ in range(W_dim)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][c] = 1
                    V[j*p + k][c] = 1
                    W[k*n + i][c] = 1
                    c += 1
        return U, V, W

    # Verify that the decomposition is exact:
    # For all (a,b,c) in Z: (A x B)_{a,b} = sum_r U_{a, r} * V_{b, r} should be encoded in W
    # Actually, the tensor M[i,j,k] = 1 if and only if the decomposition reconstructs it.
    # The contraction: sum_r U[i*m+j, r] * V[j*p+k, r] should equal W[k*n+i, r] summed over r?
    # No, the standard tensor decomposition: T[i,j,k] = sum_r U[i, r] * V[j, r] * W[k, r]
    # Here our indexing: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # So the "tensor" at indices (i,j,k) is sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This must equal 1 for the matrix multiplication tensor: T[i,j,k] = 1 always? 
    # Wait, the matrix multiplication tensor T[i,j,k] = 1 for all valid (i,j,k).
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k.

    def verify(U, V, W):
        """Check if the decomposition is exact."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        r = U.shape[1]
        # Build tensor: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # T has shape (n, m, p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for rr in range(r):
                        val += U[i*m + j, rr] * V[j*p + k, rr] * W[k*n + i, rr]
                    if abs(val - 1.0) > 1e-6:
                        return False
        return True

    def verify_fast(U, V, W):
        """Faster verification using numpy."""
        Ua = np.array(U)
        Va = np.array(V)
        Wa = np.array(W)
        r = Ua.shape[1]
        # Create index arrays
        idx_U = np.array([[i*m+j for j in range(m)] for i in range(n)]).flatten()  # shape (n*m,)
        idx_V = np.array([[j*p+k for k in range(p)] for j in range(m)]).flatten()  # shape (m*p,)
        idx_W = np.array([[k*n+i for i in range(n)] for k in range(p)]).flatten()  # shape (p*n,)
        # For each (i,j,k), check sum_r U[idx_U[i*m+j], r] * V[idx_V[j*p+k], r] * W[idx_W[k*n+i], r] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(Ua[idx_U[i*m+j], :] * Va[idx_V[j*p+k], :], Wa[idx_W[k*n+i], :])
                    if abs(val - 1.0) > 1e-6:
                        return False
        return True

    # Try to find a lower-rank decomposition using structured/random approaches
    best_rank = R_schoolbook
    best_decomp = make_schoolbook(R_schoolbook)

    # Attempt 1: Try to find a decomposition with rank < 96 using random initialisation
    # and local optimisation (this is a non-convex problem, but we can try).
    # For n=m=p, we know rank 4*4*4 has rank at most 72 (or less with Strassen-like).
    # For 4x4 by 4x6, the schoolbook is 96. Can we do better?
    # Let's try a random dense decomposition and see if we can reduce rank.

    # Actually, a more systematic approach: 
    # The tensor is T[i,j,k] = 1. We need sum_r u_{i,j}^{(r)} * v_{j,k}^{(r)} * w_{k,i}^{(r)} = 1.
    # This is a system of n*m*p = 96 equations with 3*R unknowns (R rank).
    # We want to find the minimum R such that integer/half-integer solutions exist.

    # Let's try small ranks. The theoretical minimum is not known exactly for M_4,4,6.
    # Try ranks from 64 to 95 (skip the trivial 96).

    def try_random_rank(r, n_trials=50, seed=42):
        """Try to find a rank-r decomposition with random initialisation."""
        rng = random.Random(seed)
        np_rng = np.random.RandomState(seed)
        for trial in range(n_trials):
            if time.time() - start_wall > DEADLINE - MARGIN:
                return None
            # Random initialisation with values from {-1, 0, 1}
            U = np_rng.choice([-1, 0, 1], size=(U_dim, r))
            V = np_rng.choice([-1, 0, 1], size=(V_dim, r))
            W = np_rng.choice([-1, 0, 1], size=(W_dim, r))
            if verify_fast(U, V, W):
                # Convert to lists
                return [list(row) for row in U.tolist()], [list(row) for row in V.tolist()], [list(row) for row in W.tolist()]
        return None

    # Try random rank search for R < 96
    # Focus on ranks that are multiples of 4 or 8, as they may have structure
    for r in [64, 68, 72, 76, 80, 84, 88, 92]:
        if time.time() - start_wall > DEADLINE - MARGIN:
            break
        result = try_random_rank(r, n_trials=100, seed=r*1000)
        if result is not None:
            best_rank = r
            best_decomp = result
            break  # Found a lower rank, use it

    # If nothing found, use schoolbook
    U, V, W = best_decomp

    # Final verification
    if not verify_fast(U, V, W):
        # Fall back to schoolbook
        U, V, W = make_schoolbook(R_schoolbook)
        best_rank = R_schoolbook

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
