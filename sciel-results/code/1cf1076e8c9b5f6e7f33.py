"""Matrix multiplication 3,3,5 – rank-minimal exact tensor decomposition."""
import time
import random
import itertools

import numpy as np

DIMENSIONS = [3, 3, 5]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _verify(U, V, W, n, m, p):
    """Check that the factorisation exactly reproduces the multiplication tensor.

    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    Convention: output C[i,k] = sum_j A[i,j] B[j,k]
    The multiplication tensor T has entries T[i*m+j, j*p+k, k*n+i] ...
    Actually the standard 3-way tensor for matrix multiplication is:
    T[i, j, k] where we contract over j.
    In the flattened 2-matrix + 1-index convention used here:
      U row index: i*m + j  (A factor)
      V row index: j*p + k  (B factor)
      W row index: k*n + i  (C factor, output index)
    The tensor element is T[(i*m+j), (j*p+k), (k*n+i)] = 1 when the indices
    are consistent (same j for U and V, same k for V and W, same i for W and U).
    More precisely, the multiplication identity is:
      sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]  ... no.

    Let me re-read the task:
    "U', 'V', 'W'}: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    "W[k*n+i,r]"

    The tensor decomposition identity is:
      For all i in [0,n), j in [0,m), k in [0,p):
        delta[i,m,*,j,p,*,k,n,*,i] ... 

    Actually the standard way: the matrix multiplication tensor has a trilinear form:
      sum_{i,j,k} A[i,j] B[j,k] C[k,i]  ... no, C is the output.

    Let me think again. The multiplication map is:
      (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k]

    As a 3-way tensor with dimensions (n*m, m*p, p*n):
      T[a, b, c] where a = i*m+j, b = j'*p+k, c = k'*n+i'
      T[a,b,c] = 1 iff j == j' AND k == k' AND i == i'
      i.e., T[i*m+j, j*p+k, k*n+i] = 1, and 0 otherwise.

    A rank-r decomposition: T[a,b,c] = sum_{r=0}^{R-1} U[a,r] * V[b,r] * W[c,r]

    So we need: sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r] = delta_{j,j'} delta_{k,k'} delta_{i,i'}

    For the schoolbook: each r corresponds to a unique (i,j,k) triple:
      U[i*m+j, r_{i,j,k}] = 1
      V[j*p+k, r_{i,j,k}] = 1
      W[k*n+i, r_{i,j,k}] = 1
    This gives exactly the delta.
    """
    n, m, p = n, m, p
    r = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    for i in range(n):
        for j in range(m):
            for jp in range(m):  # j'
                for k in range(p):
                    for kn in range(p):  # k'
                        for i2 in range(n):  # i'
                            lhs = np.dot(
                                np.dot(U_arr[i * m + j, :], V_arr[jp * p + k, :]),
                                0  # placeholder
                            )
                            # Actually compute T[i*m+j, jp*p+k, kn*n+i2]
                            # = sum_r U[i*m+j, r] * V[jp*p+k, r] * W[kn*n+i2, r]
                            prod = U_arr[i * m + j, :] * V_arr[jp * p + k, :] * W_arr[kn * n + i2, :]
                            val = np.sum(prod)
                            expected = 1.0 if (j == jp and k == kn and i == i2) else 0.0
                            if abs(val - expected) > 1e-9:
                                return False
    return True


def _rank_from_factors(U, V, W):
    """Return the number of columns (rank) of the factor matrices."""
    if U is None or len(U) == 0 or len(U[0]) == 0:
        return 0
    return len(U[0])


# ---------------------------------------------------------------------------
# Construct known low-rank decompositions for small matrix multiplication
# ---------------------------------------------------------------------------

def _construct_strassen_2x2x2_scaled(n, m, p):
    """
    Build a rank-7 decomposition for 2x2 x 2x2 using Strassen.
    This is just for reference/verification.
    """
    pass  # not needed for 3,3,5


def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x5 matrix multiplication.
    Returns {'U': U, 'V': V, 'W': W} with shapes (9, rank), (15, rank), (15, rank).
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = start_time + 120  # seconds

    # --- Schoolbook baseline (always valid) ---
    rank_school = n * m * p  # 45
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school

    def try_rank(R):
        """Attempt to find a rank-R exact decomposition. Returns (U,V,W) or None."""
        if time.time() > DEADLINE - 5:
            return None

        # Strategy: try several random initializations and use Newton refinement
        # on the tensor factorization problem.
        # The tensor is T with T[i*m+j, j*p+k, k*n+i] = 1.

        rng = random.Random(42 + R * 1000)
        np_rng = np.random.default_rng(42 + R * 1000)

        # Number of random restarts
        n_restarts = 3
        for restart in range(n_restarts):
            if time.time() > DEADLINE - 5:
                break

            # Initialize factors randomly
            U = np_rng.standard_normal((n * m, R)) * 0.5
            V = np_rng.standard_normal((m * p, R)) * 0.5
            W = np_rng.standard_normal((p * n, R)) * 0.5

            # Newton's method for tensor factorization:
            # We want sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c] for all (a,b,c) in support.
            # The support size is n*m*p = 45 equations, unknowns = R*(n*m + m*p + p*n) = R*39.
            # For R < 45/39 ≈ 1.15, we have more unknowns than equations? No:
            # Unknowns = R*39, equations = 45. So for R >= 2, 78 unknowns > 45 eqs.
            # This is an underdetermined nonlinear system.

            # Newton iteration:
            # F(U,V,W)_{abc} = sum_r U[a,r] V[b,r] W[c,r] - T[a,b,c]
            # We iterate: at each step, solve for small delta_U, delta_V, delta_W
            # such that linearized F + J * delta = 0.

            # Simpler approach: use the "alternating" method or leverage the sparsity.
            # For exact integer/half-integer solutions, try a structured approach.

            # Alternative: try to find solutions with entries in {-1, 0, 1} or half-integers.
            # This is a combinatorial search, but for small R it's feasible.

            # For R=15: unknowns = 15*39 = 585, equations = 45. Very underdetermined.
            # The solution space is huge. Random Newton should converge quickly.

            # Let's do simple fixed-point / Gauss-Seidel style update.
            for iteration in range(200):
                if time.time() > DEADLINE - 5:
                    break

                # Update U: for each a, U[a,:] = (sum over b,c with T[a,b,c]!=0) ... 
                # Actually, from T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]:
                # Fix V,W, solve for U:
                # For each a: U[a,:] * (V ⊙ W)_[b,c,:]^T ... this is messy.

                # Let me use a different approach: direct linear solve per row.
                # For fixed V, W, the equation for each (a,b,c) is:
                # sum_r U[a,r] * (V[b,r] * W[c,r]) = T[a,b,c]
                # This is linear in U[a,:]. But it's the same U[a,:] for all b,c.
                # We can set up: for each a, we have equations for all (b,c) where the
                # coefficient vector (V[b,r]*W[c,r])_r is defined.
                # There are m*p * p*n = 15*15 = 225 equations for U[a,:] (which has R components).
                # For R <= 225, we can least-squares solve.

                # Actually T is sparse: T[a,b,c] = 1 only for specific (b,c) given a.
                # For a = i*m+j, T[a, jp*p+k, kn*n+i2] = 1 iff jp=j, kn=k, i2=i.
                # So for fixed a=(i,j), T[a,b,c]=1 only when b=j*p+k (any k) and c=k*n+i.
                # That's p*1 = 5 nonzero entries per a? No: for each k, b=j*p+k and c=k*n+i.
                # So 5 entries.

                # This means for each a, we only have 5 equations (with R unknowns in U[a,:]).
                # For R>5, the system is underdetermined per a. We need to use the
                # coupling across a's.

            # Given the complexity, let me try a direct approach for small R using
            # the fact that we can parametrize and solve.

            break  # Skip Newton for now; use combinatorial/structured search below

        return None

    # --- Structured search for low-rank decompositions ---
    # Known results:
    # 3x3 * 3x3: rank 21 is conjectured minimal (Bini et al.), but not proven.
    # 3x3 * 3x5: likely rank around 23-25.
    # Let me try to construct a valid decomposition with rank < 45 using known techniques.

    # Approach: start from schoolbook and try to merge/eliminate terms.
    # Or: use the "tensor contraction" approach with random projections and
    # verify with exact rational arithmetic.

    # For now, let me try a practical approach:
    # 1. Try rank values from 20 to 44.
    # 2. For each, use a randomized algorithm:
    #    - Initialize U, V, W with small random rational values.
    #    - Use iterative refinement to satisfy the 45 tensor equations.
    #    - Check for exactness.

    # Given time constraints, let me implement a focused search for R in a small range.

    def _attempt_decomposition(R, seed):
        """Try to find a rank-R decomposition using randomized Newton on the tensor."""
        if time.time() > DEADLINE - 5:
            return None

        np_rng = np.random.default_rng(seed)

        # Initialize
        U = np_rng.standard_normal((n * m, R)) * 0.5
        V = np_rng.standard_normal((m * p, R)) * 0.5
        W = np_rng.standard_normal((p * n, R)) * 0.5

        # Build the sparse tensor structure
        # T[a, b, c] = 1 for a=i*m+j, b=j*p+k, c=k*n+i
        # List of (a, b, c) indices
        support = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    support.append((a, b, c))

        # Newton's method on the system F(U,V,W) = 0 where
        # F[a,b,c] = sum_r U[a,r] V[b,r] W[c,r] - 1 for (a,b,c) in support

        for it in range(100):
            if time.time() > DEADLINE - 3:
                break

            # Compute residual
            res = 0.0
            for (a, b, c) in support:
                val = np.dot(U[a, :] * V[b, :], W[c, :])
                res += (val - 1.0) ** 2

            if res < 1e-12:
                return (U.tolist(), V.tolist(), W.tolist())

            # Gradient / Jacobian computation is complex for 3-way tensor.
            # Use simple coordinate descent instead.
            # For each factor, update to minimize residual.

            # Update U
            for a in range(n * m):
                # Solve for U[a,:] given V, W
                # Minimize sum_{(a,b,c) in support} (sum_r U[a,r] V[b,r] W[c,r] - 1)^2
                # This is linear in U[a,:]:
                # Let w_bc = V[b,:] * W[c,:]  (element-wise, shape R)
                # Then we need U[a,:] · w_bc ≈ 1 for each (b,c) with (a,b,c) in support
                # Collect equations:
                A_eq = []
                b_eq = []
                for (a2, b, c) in support:
                    if a2 == a:
                        w_bc = V[b, :] * W[c, :]
                        A_eq.append(w_bc)
                        b_eq.append(1.0)
                if A_eq:
                    A_mat = np.array(A_eq)  # (num_eq, R)
                    b_vec = np.array(b_eq)
                    # Least squares
                    U[a, :], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)

            # Update V
            for b in range(m * p):
                A_eq = []
                b_eq = []
                for (a, b2, c) in support:
                    if b2 == b:
                        w_ac = U[a, :] * W[c, :]
                        A_eq.append(w_ac)
                        b_eq.append(1.0)
                if A_eq:
                    A_mat = np.array(A_eq)
                    b_vec = np.array(b_eq)
                    V[b, :], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)

            # Update W
            for c in range(p * n):
                A_eq = []
                b_eq = []
                for (a, b, c2) in support:
                    if c2 == c:
                        w_ab = U[a, :] * V[b, :]
                        A_eq.append(w_ab)
                        b_eq.append(1.0)
                if A_eq:
                    A_mat = np.array(A_eq)
                    b_vec = np.array(b_eq)
                    W[c, :], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)

            # Check convergence
            res = 0.0
            for (a, b, c) in support:
                val = np.dot(U[a, :] * V[b, :], W[c, :])
                res += (val - 1.0) ** 2
            if res < 1e-10:
                return (U.tolist(), V.tolist(), W.tolist())

        return None

    # Try decreasing ranks
    for R in range(20, rank_school):
        if time.time() > DEADLINE - 10:
            break
        for seed in [R * 7 + 1, R * 13 + 2, R * 23 + 3]:
            result = _attempt_decomposition(R, seed)
            if result is not None:
                U_res, V_res, W_res = result
                # Verify exactness
                if _verify(U_res, V_res, W_res, n, m, p):
                    best_U, best_V, best_W = U_res, V_res, W_res
                    best_rank = R
                    return {"U": best_U, "V": best_V, "W": best_W}

    # If no better found, return schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {_rank_from_factors(result['U'], result['V'], result['W'])}")
