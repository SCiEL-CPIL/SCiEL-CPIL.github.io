"""Program for 3x3x3 matrix multiplication tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication.
    Tries to construct/verify a rank-23 decomposition using a finite-field sparse
    linear-system formulation, then falls back to the known rank-26 construction.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    n, m, p = DIMENSIONS

    def idx_u(i, j):
        return i * m + j

    def idx_v(j, k):
        return j * p + k

    def idx_w(k, i):
        return k * n + i

    def make_rank26():
        R = 26
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))

        # Strassen part for the 2x2 core
        r = 0
        U[idx_u(0, 0), r] = 1
        U[idx_u(1, 1), r] = 1
        V[idx_v(0, 0), r] = 1
        V[idx_v(1, 1), r] = 1
        W[idx_w(0, 0), r] = 1
        W[idx_w(1, 1), r] = 1

        r = 1
        U[idx_u(1, 0), r] = 1
        U[idx_u(1, 1), r] = 1
        V[idx_v(0, 0), r] = 1
        W[idx_w(0, 1), r] = 1
        W[idx_w(1, 1), r] = -1

        r = 2
        U[idx_u(0, 0), r] = 1
        V[idx_v(0, 1), r] = 1
        V[idx_v(1, 1), r] = -1
        W[idx_w(1, 0), r] = 1
        W[idx_w(1, 1), r] = 1

        r = 3
        U[idx_u(1, 1), r] = 1
        V[idx_v(1, 0), r] = 1
        V[idx_v(0, 0), r] = -1
        W[idx_w(0, 0), r] = 1
        W[idx_w(0, 1), r] = 1

        r = 4
        U[idx_u(0, 0), r] = 1
        U[idx_u(0, 1), r] = 1
        V[idx_v(1, 1), r] = 1
        W[idx_w(0, 0), r] = -1
        W[idx_w(1, 0), r] = 1

        r = 5
        U[idx_u(1, 0), r] = 1
        U[idx_u(0, 0), r] = -1
        V[idx_v(0, 0), r] = 1
        V[idx_v(0, 1), r] = 1
        W[idx_w(1, 1), r] = 1

        r = 6
        U[idx_u(0, 1), r] = 1
        U[idx_u(1, 1), r] = -1
        V[idx_v(1, 0), r] = 1
        V[idx_v(1, 1), r] = 1
        W[idx_w(0, 0), r] = 1

        # Border terms with at least one index equal to 2
        col = 7
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if i < 2 and j < 2 and k < 2:
                        continue
                    U[idx_u(i, j), col] = 1
                    V[idx_v(j, k), col] = 1
                    W[idx_w(k, i), col] = 1
                    col += 1

        U_reshaped = U.reshape((n, m, R))
        V_reshaped = V.reshape((m, p, R))
        W_reshaped = W.reshape((p, n, R))
        T_recon = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        T_target = np.ones((n, m, p))

        if not np.allclose(T_recon, T_target, atol=1e-9):
            # Schoolbook fallback
            R = n * m * p
            U = np.zeros((n * m, R))
            V = np.zeros((m * p, R))
            W = np.zeros((p * n, R))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[idx_u(i, j), col] = 1
                        V[idx_v(j, k), col] = 1
                        W[idx_w(k, i), col] = 1
                        col += 1

        return U, V, W

    def verify_real(U, V, W, R):
        U_reshaped = U.reshape((n, m, R))
        V_reshaped = V.reshape((m, p, R))
        W_reshaped = W.reshape((p, n, R))
        T_recon = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        T_target = np.ones((n, m, p))
        return np.allclose(T_recon, T_target, atol=1e-9)

    # Attempt a rank-23 search over GF(5).
    # We model a sparse decomposition with rank R and solve the resulting
    # quadratic constraints by linearization: fix a support pattern and solve
    # for coefficients.  The support pattern is chosen from the rank-26
    # construction by merging some border terms into Strassen-like linear
    # combinations.  This is a heuristic within the same construction family.
    def try_rank23_gf5():
        R = 23
        q = 5

        # Start from rank-26 pattern and try to remove 3 border terms by
        # allowing their contributions to be absorbed by nearby Strassen terms.
        # We create a candidate support by taking the first 20 terms of the
        # rank-26 construction (7 Strassen + 13 border) and then adding 3
        # additional structured terms that may cover the remaining border
        # contributions.
        #
        # This is intentionally conservative: we only try a small number of
        # structured patterns and verify over GF(5) using polynomial identity
        # testing with random evaluations.

        # Base rank-26 support (first 20 terms)
        base_terms = []
        # Strassen terms
        base_terms.append([(0,0),(0,0),(0,0),1,1,1])
        base_terms.append([(1,0),(0,0),(1,0),1,1,1])
        base_terms.append([(0,0),(0,1),(1,0),1,1,1])
        base_terms.append([(1,1),(1,0),(0,0),1,1,1])
        base_terms.append([(0,0),(1,1),(0,0),1,1,-1])
        base_terms.append([(1,0),(0,0),(1,1),1,1,1])
        base_terms.append([(0,1),(1,1),(0,0),1,1,1])

        # Border terms in deterministic order; take 13 of them.
        border_order = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if i < 2 and j < 2 and k < 2:
                        continue
                    border_order.append((i,j,k))
        # border_order has 19 entries; take first 13
        for (i,j,k) in border_order[:13]:
            base_terms.append([(i,j),(j,k),(k,i),1,1,1])

        # Add 3 structured terms that may absorb remaining border contributions.
        # These are heuristic combinations involving index 2.
        extra_terms = [
            [(0,2),(2,0),(2,0),1,1,1],
            [(1,2),(2,1),(2,1),1,1,1],
            [(2,0),(0,2),(2,0),1,1,1],
        ]
        pattern = base_terms + extra_terms
        if len(pattern) != R:
            return None

        # Build a linear system over GF(5) for coefficients.
        # For each (i,j,k) in {0,1,2}^3, the equation is:
        #   sum_r u_{i,j,r} v_{j,k,r} w_{k,i,r} = 1
        # We linearize by fixing the support pattern (which entries are nonzero)
        # and solving for the nonzero coefficients.  Because the system is
        # quadratic, we instead use a randomized identity test: assign random
        # values to the free coefficients and check if the polynomial identity
        # holds for random matrix entries.  This is not a full solve, but it
        # can verify a candidate if we find one by local search.

        # For this heuristic, we initialize coefficients to 1 and then perform
        # a few local perturbations over GF(5) to try to satisfy the identity.
        # This is a very limited search, but it stays within the same family
        # and uses finite-field arithmetic to avoid numerical instability.

        import random
        rng = random.Random(42)

        # Initialize coefficients
        coeffs = []
        for term in pattern:
            coeffs.append([1, 1, 1])  # u, v, w coefficients

        # Local search: perturb coefficients and check identity over GF(5)
        # using random matrix evaluations.
        def check_identity(coeffs, num_trials=5):
            for _ in range(num_trials):
                # Random matrices over GF(5)
                A = np.array([[rng.randint(0, q-1) for _ in range(m)] for _ in range(n)], dtype=int)
                B = np.array([[rng.randint(0, q-1) for _ in range(p)] for _ in range(m)], dtype=int)
                C = np.array([[rng.randint(0, q-1) for _ in range(p)] for _ in range(n)], dtype=int)
                # Compute C_target = A @ B over GF(5)
                C_target = (A @ B) % q
                # Compute C_recon from decomposition
                C_recon = np.zeros((n, p), dtype=int)
                for r in range(R):
                    u_coeff, v_coeff, w_coeff = coeffs[r]
                    (i,j), (jk), (ki) = pattern[r][0], pattern[r][1], pattern[r][2]
                    # This is a simplified check: we only support terms where
                    # the pattern is a single (i,j,k) triple.  For structured
                    # terms, we would need to expand, but this heuristic is
                    # intentionally limited.
                    if pattern[r][3] == 1 and pattern[r][4] == 1 and pattern[r][5] == 1:
                        # Single term
                        val = (u_coeff * A[i,j] * v_coeff * B[j,k] * w_coeff) % q
                        C_recon[i, k] = (C_recon[i, k] + val) % q
                    else:
                        # Structured term: skip (not fully implemented)
                        continue
                if not np.all((C_recon == C_target) % q):
                    return False
            return True

        # Try a few local perturbations
        for _ in range(100):
            if check_identity(coeffs, num_trials=3):
                # If verified, convert to real-valued matrices
                U = np.zeros((n * m, R))
                V = np.zeros((m * p, R))
                W = np.zeros((p * n, R))
                for r in range(R):
                    (i,j), (jk), (ki) = pattern[r][0], pattern[r][1], pattern[r][2]
                    u_coeff, v_coeff, w_coeff = coeffs[r]
                    U[idx_u(i, j), r] = u_coeff
                    V[idx_v(jk[0], jk[1]), r] = v_coeff
                    W[idx_w(ki[1], ki[0]), r] = w_coeff
                if verify_real(U, V, W, R):
                    return U, V, W
            # Perturb a random coefficient
            r = rng.randint(0, R-1)
            c = rng.randint(0, 2)
            coeffs[r][c] = (coeffs[r][c] + rng.randint(1, q-1)) % q

        return None

    # Try rank-23 first
    result = try_rank23_gf5()
    if result is not None:
        U, V, W = result
        if verify_real(U, V, W, 23):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to rank-26
    U, V, W = make_rank26()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
