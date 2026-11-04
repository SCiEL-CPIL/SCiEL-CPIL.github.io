import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 5
    deadline = time.time() + 140

    # Tensor dimensions
    dim_A = n * m  # 6
    dim_B = m * p  # 15
    dim_C = p * n  # 10

    # Target tensor T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1

    # Precompute index mappings
    def a_idx(i, j): return i * m + j
    def b_idx(j, k): return j * p + k
    def c_idx(k, i): return k * n + i

    # Build the list of all (i, j, k) triples
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    num_equations = len(triples)  # 30

    best_result = None
    best_rank = n * m * p  # schoolbook = 30

    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        for (i, j, k) in triples:
            val = 0.0
            for r in range(U.shape[1]):
                val += U[a_idx(i, j), r] * V[b_idx(j, k), r] * W[c_idx(k, i), r]
            if abs(val - 1.0) > 1e-9:
                return False
        return True

    def is_valid_entries(M):
        """Check entries are integers or half-integers."""
        for val in M.flatten():
            if abs(val * 2 - round(val * 2)) > 1e-9:
                return False
        return True

    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((dim_A, rank))
        V = np.zeros((dim_B, rank))
        W = np.zeros((dim_C, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[a_idx(i, j), col] = 1
                    V[b_idx(j, k), col] = 1
                    W[c_idx(k, i), col] = 1
                    col += 1
        return U, V, W

    # Strategy: For each rank r, try to find a decomposition by:
    # 1. Randomly assign sparsity patterns to U, V, W
    # 2. Set up the linear system
    # 3. Solve and check for valid (half-integer) solutions
    
    # The key insight: if we fix U and V, then for each equation (i,j,k):
    # sum_r U[a_idx(i,j),r] * V[b_idx(j,k),r] * W[c_idx(k,i),r] = 1
    # This is linear in W. So we can:
    # 1. Randomly generate U, V with sparse ±1 patterns
    # 2. Build the linear system for W
    # 3. Solve and check validity

    # Alternative: fix W patterns and solve for U, V jointly is harder.
    # Let's fix U and V, solve for W.

    # For a given U (dim_A x r) and V (dim_B x r), the equation for W becomes:
    # For each (i,j,k): sum_r (U[a_idx(i,j),r] * V[b_idx(j,k),r]) * W[c_idx(k,i),r] = 1
    # Let M_{(i,j,k), r} = U[a_idx(i,j),r] * V[b_idx(j,k),r]
    # Then M @ W_vec = 1 (vector of ones)
    # But W is dim_C x r, so we have num_equations equations in dim_C*r unknowns.
    # Actually, W[c_idx(k,i), r] appears in equation (i,j,k) with coefficient M_{(i,j,k),r}.
    # So the system is: for each (i,j,k), sum_r M_{(i,j,k),r} * W[c_idx(k,i),r] = 1
    # This is a system of num_equations equations in dim_C*r unknowns.
    # If r is large enough, this is underdetermined and we can find a solution.
    # But we need the solution to have half-integer entries.

    # Better approach: fix ALL three matrices' sparsity patterns, then solve.
    # But that's 3 * r * dim unknowns. Too many.

    # Alternative: Use the fact that we can set some entries of W to 1 (or -1)
    # and solve for U, V. Or use a rank-1 outer product approach.

    # Let's try a different structural approach:
    # Use the known fact that rank(2x3x5) <= 27 (from block decomposition).
    # But let's try to find rank 26 or lower via search.

    # Actually, let's try a completely different approach:
    # Use the "border rank" or "approximate" rank and then round.
    # No, we need exact.

    # Let's try: for each rank r from 20 to 30, try random sparse patterns.
    # For efficiency, fix U and V to have specific structures.

    # Key structural idea: Use the fact that the tensor is all-ones.
    # If we use U with entries in {0,1,-1} and V with entries in {0,1,-1},
    # then M_{(i,j,k),r} is in {0,1,-1}.
    # We need W such that for each (i,j,k): sum_r M_{(i,j,k),r} * W[c_idx(k,i),r] = 1.

    # Let's try: set W to have a specific structure.
    # For example, W[c_idx(k,i), r] = 1 for some subset of (k,i,r).

    # Actually, let's use a more systematic approach:
    # Generate random U, V with entries in {0, 1, -1} (sparse),
    # then solve the linear system for W, and check if W has valid entries.

    rng = np.random.default_rng(42)

    for r in range(20, min(31, n*m*p + 1)):
        if time.time() > deadline - 10:
            break

        for attempt in range(200):
            if time.time() > deadline - 5:
                break

            # Generate sparse U: each row has ~2 non-zero entries
            U = np.zeros((dim_A, r))
            for i in range(dim_A):
                nz = rng.integers(1, 4)  # 1 to 3 non-zero entries
                cols = rng.choice(r, size=nz, replace=False)
                U[i, cols] = rng.choice([-1, 1], size=nz)

            # Generate sparse V
            V = np.zeros((dim_B, r))
            for i in range(dim_B):
                nz = rng.integers(1, 4)
                cols = rng.choice(r, size=nz, replace=False)
                V[i, cols] = rng.choice([-1, 1], size=nz)

            # Build the linear system for W
            # Equation (i,j,k): sum_r U[a_idx(i,j),r] * V[b_idx(j,k),r] * W[c_idx(k,i),r] = 1
            # Unknowns: W[c_idx(k,i), r] for all k, i, r
            # Let's vectorize W as W_vec of length dim_C * r
            # W_vec[c_idx(k,i) * r + r_idx] = W[c_idx(k,i), r_idx]

            A_mat = np.zeros((num_equations, dim_C * r))
            b_vec = np.ones(num_equations)

            eq_idx = 0
            for (i, j, k) in triples:
                ai = a_idx(i, j)
                bi = b_idx(j, k)
                ci = c_idx(k, i)
                for r_idx in range(r):
                    coeff = U[ai, r_idx] * V[bi, r_idx]
                    if coeff != 0:
                        A_mat[eq_idx, ci * r + r_idx] = coeff
                eq_idx += 1

            # Solve the underdetermined system A_mat @ W_vec = b_vec
            # Use least squares
            try:
                W_vec, residuals, rank_solver, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            except:
                continue

            # Check if the solution is exact
            if residuals.size > 0 and residuals[0] > 1e-6:
                continue

            W = W_vec.reshape(dim_C, r)

            # Check if W has valid entries (integers or half-integers)
            # Round to nearest half-integer
            W_rounded = np.round(W * 2) / 2

            # Verify the rounded solution
            valid = True
            for (i, j, k) in triples:
                val = 0.0
                for r_idx in range(r):
                    val += U[a_idx(i, j), r_idx] * V[b_idx(j, k), r_idx] * W_rounded[c_idx(k, i), r_idx]
                if abs(val - 1.0) > 1e-9:
                    valid = False
                    break

            if valid:
                best_rank = r
                best_result = (U, V, W_rounded)
                break  # Found a valid solution for this rank

        if best_result is not None and best_rank == r:
            break  # Found the minimum rank

    # If we found a better result, use it; otherwise use schoolbook
    if best_result is not None:
        U, V, W = best_result
    else:
        U, V, W = schoolbook()
        best_rank = n * m * p

    # Final verification
    if not verify(U, V, W):
        U, V, W = schoolbook()

    if not is_valid_entries(U) or not is_valid_entries(V) or not is_valid_entries(W):
        U, V, W = schoolbook()

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
