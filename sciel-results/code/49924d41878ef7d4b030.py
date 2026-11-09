"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 130.0  # safety margin below 150s

    # Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    nm = n * m
    mp = m * p
    pn = p * n
    T = np.zeros((nm, mp, pn), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1

    def verify_decomposition(U, V, W):
        """Check if U, V, W (as arrays) exactly decompose T."""
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_rec, T, atol=1e-9)

    # Start with schoolbook
    rank_school = n * m * p
    U_sb = np.zeros((nm, rank_school))
    V_sb = np.zeros((mp, rank_school))
    W_sb = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1

    best_rank = rank_school
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    # Strategy: Use structured low-rank decompositions based on
    # the algebraic structure of matrix multiplication.
    #
    # Key insight: For an n x m matrix A and m x p matrix B, the (i,k) entry
    # of AB is sum_j A[i,j] * B[j,k]. We can try to exploit the fact that
    # the tensor has a specific Kronecker-like structure.
    #
    # Approach 1: Try to find decompositions where each rank-1 term corresponds
    # to a subset of the (i,j,k) triples, potentially sharing terms.
    #
    # Approach 2: Use the fact that for small dimensions, we can enumerate
    # possible "patterns" of how rank-1 terms can cover the tensor entries.

    # First, let's try a smarter initialization based on the tensor structure.
    # For each (i,k) pair, we need sum_j A[i,j]*B[j,k] = C[i,k].
    # A rank-1 decomposition term (u, v, w) contributes u[i*m+j] * v[j*p+k] * w[k*n+i]
    # to the (i*m+j, j*p+k, k*n+i) entry.

    # Let's try a different approach: use the known lower bounds and known
    # results for small matrix multiplication.
    #
    # For 3x5 * 5x6, the schoolbook rank is 90.
    # Known lower bound: max(n*m, m*p, p*n) = max(15, 30, 18) = 30.
    # There's also a bound related to the "support" of the tensor.
    #
    # Let's try to find a decomposition of rank < 90 using a more structured approach.

    # Method: Start from schoolbook and try to merge terms.
    # Two schoolbook terms (i1,j1,k1) and (i2,j2,k2) can potentially be merged
    # if their "interaction" can be captured by a single rank-1 term.

    # Actually, let's try a more direct approach: use random search with
    # a better objective function and more structured initialization.

    rng = np.random.RandomState(12345)

    # Try ranks from a reasonable lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = 30 for our case
    lower_bound = max(n*m, m*p, p*n)

    # Also try the "Strassen-like" bound: for square n x n, rank is about n^log2(7)
    # For 3x5x6, let's just try from 30 up
    for r in range(lower_bound, rank_school + 1):
        if time.time() > deadline - 30:
            break

        max_attempts = 5 if r < 50 else 3
        for attempt in range(max_attempts):
            if time.time() > deadline - 20:
                break

            # Initialize with a structured pattern
            # Each rank-1 term will be associated with a "block" of the multiplication
            U = np.zeros((nm, r))
            V = np.zeros((mp, r))
            W = np.zeros((pn, r))

            # Assign each (i,j,k) triple to a rank index
            # Use a hash-like function to distribute
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Simple assignment: use (i*m+j + k) % r or similar
                        # Better: use a more structured assignment
                        idx = (i * m + j) * p + k
                        rank_idx = idx % r
                        U[i*m + j, rank_idx] += 1
                        V[j*p + k, rank_idx] += 1
                        W[k*n + i, rank_idx] += 1

            # Now try to optimize via local search
            obj_prev = np.linalg.norm(np.einsum('ar,br,cr->abc', U, V, W) - T)

            # Local search: perturb random entries
            max_iters = 100
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    break

                # Pick a random matrix, row, and column to perturb
                which = rng.randint(0, 3)
                if which == 0:
                    row_idx = rng.randint(0, nm)
                    col_idx = rng.randint(0, r)
                    delta = rng.randn() * 0.5
                    U_new = U.copy()
                    U_new[row_idx, col_idx] += delta
                    cand = (U_new, V, W)
                elif which == 1:
                    row_idx = rng.randint(0, mp)
                    col_idx = rng.randint(0, r)
                    delta = rng.randn() * 0.5
                    V_new = V.copy()
                    V_new[row_idx, col_idx] += delta
                    cand = (U, V_new, W)
                else:
                    row_idx = rng.randint(0, pn)
                    col_idx = rng.randint(0, r)
                    delta = rng.randn() * 0.5
                    W_new = W.copy()
                    W_new[row_idx, col_idx] += delta
                    cand = (U, V, W_new)

                obj_new = np.linalg.norm(np.einsum('ar,br,cr->abc', *cand) - T)
                if obj_new < obj_prev:
                    U, V, W = cand
                    obj_prev = obj_new

            # Try to snap to half-integers
            U_h = np.round(U * 2) / 2.0
            V_h = np.round(V * 2) / 2.0
            W_h = np.round(W * 2) / 2.0

            if np.allclose(U_h, U, atol=1e-9) and np.allclose(V_h, V, atol=1e-9) and np.allclose(W_h, W, atol=1e-9):
                if verify_decomposition(U_h, V_h, W_h):
                    if r < best_rank:
                        best_rank = r
                        best_U, best_V, best_W = U_h, V_h, W_h
                    break

    # Convert best to list format
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()

    # Ensure entries are exact half-integers
    def clean_half_int(lst):
        out = []
        for row in lst:
            new_row = []
            for v in row:
                v2 = round(v * 2) / 2.0
                if v2 == int(v2):
                    new_row.append(int(v2))
                else:
                    new_row.append(v2)
            out.append(new_row)
        return out

    U_out = clean_half_int(U_out)
    V_out = clean_half_int(V_out)
    W_out = clean_half_int(W_out)

    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
