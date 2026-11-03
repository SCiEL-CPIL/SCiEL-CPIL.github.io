"""Rewritten program: structured sparse linear algebra / binary IP for exact tensor decomposition."""
DIMENSIONS = [3, 3, 8]
import random
random.seed(1)


# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = 3, 3, 8
    nm, mp, pn = n * m, m * p, p * n  # 9, 24, 24

    # Target tensor T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k; else 0.
    # We need T = sum_r U[:,r] * V[:,r] * W[:,r] (outer product per rank).
    # Equivalently: for each (a,b,c), sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c].

    # Build dictionary of candidate rank-1 terms (u, v, w) with small integer entries.
    # For each candidate, compute the vector t_c = u * v * w (elementwise product) of length nm*mp*pn.
    # We need to select a subset (with integer coefficients) such that sum coeff_c * t_c = T.

    # Strategy: Use a dictionary of structured patterns, then solve a linear system
    # where we pick R columns from a large matrix and solve for integer coefficients.

    # Generate candidate patterns:
    candidates = []  # each is (u, v, w) as numpy arrays

    # 1. Schoolbook patterns: single (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = np.zeros(nm); u[i*m+j] = 1
                v = np.zeros(mp); v[j*p+k] = 1
                w = np.zeros(pn); w[k*n+i] = 1
                candidates.append((u, v, w))

    # 2. Strassen-like patterns for 3x3x3 blocks (rank 23 for 3x3x3)
    # For 3x3x3, known rank is 23. Let's add some structured patterns.
    # Pattern: u has support on multiple (i,j), v on multiple (j,k), w on multiple (k,i)
    # Try all subsets of size 2-3 for each vector.

    import itertools

    # Add patterns where u, v, w each have 2 or 3 non-zero entries from {-1, 0, 1}
    # This is expensive, so limit to structured ones.

    # Strassen for 3x3x3: I'll add a few known patterns.
    # Actually, let me generate random sparse patterns with small support.
    rng = np.random.RandomState(42)

    # Add random sparse patterns: each vector has 1-3 non-zero entries from {-1, 1}
    for trial in range(5000):
        # Choose support sizes
        su = rng.randint(1, 4)
        sv = rng.randint(1, 4)
        sw = rng.randint(1, 4)
        if su * sv * sw > 27:  # too large
            continue
        u = np.zeros(nm)
        v = np.zeros(mp)
        w = np.zeros(pn)
        u_idx = rng.choice(nm, su, replace=False)
        v_idx = rng.choice(mp, sv, replace=False)
        w_idx = rng.choice(pn, sw, replace=False)
        u[u_idx] = rng.choice([-1, 1], su)
        v[v_idx] = rng.choice([-1, 1], sv)
        w[w_idx] = rng.choice([-1, 1], sw)
        candidates.append((u, v, w))

    # Build the target tensor vector
    T = np.zeros(nm * mp * pn, dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a * mp * pn + b * pn + c] = 1

    # For each candidate, compute its contribution vector
    # t_c = u ⊗ v ⊗ w flattened
    cand_vecs = []
    for (u, v, w) in candidates:
        tv = np.outer(np.outer(u, v), w).flatten()
        cand_vecs.append(tv)

    cand_matrix = np.column_stack(cand_vecs)  # shape (nm*mp*pn, len(candidates))

    # Now we need to find a subset of columns and integer coefficients such that
    # cand_matrix @ x = T, minimizing sum |x_r| or number of non-zero x_r.
    # This is an integer linear programming problem.

    # Greedy approach: iteratively pick the candidate that best reduces the residual.
    # Since we need exact integer solutions, use a different strategy.

    # Alternative: Use the fact that the tensor is sparse (72 ones out of 5184).
    # Try to find R such that we can solve a system of 72 equations (the 1-entries)
    # with R unknowns, ensuring all other entries are 0.

    # Let's use a randomized local search approach:
    # Start with R candidates (schoolbook + some Strassen-like), assign integer coefficients,
    # and minimize the number of violated constraints.

    # Actually, let's use a more direct approach:
    # The schoolbook gives rank 72. Can we do better?
    # For 3x3x3, rank is 23 (Strassen). For 3x3x8, split into 3x3x3 + 3x3x3 + 3x3x2.
    # 3x3x3 rank 23, 3x3x2 rank at most 18 (schoolbook 3*3*2=18), maybe less.
    # Total: 23 + 23 + 18 = 64. Or 23 + 23 + 12 = 58 if 3x3x2 has rank 12.

    # Let me try to construct a rank-58 or lower decomposition.
    # Split B (3x8) into B1(3x3), B2(3x3), B3(3x2).
    # Use Strassen for A*B1 and A*B2 (rank 23 each), and a good decomposition for A*B3.

    # For 3x3x2, the tensor has 3*3*2=18 ones. Schoolbook rank 18.
    # Can we do better? The border rank of 3x3x2 is at most 18, but exact rank might be less.
    # Let's try rank 14 or 15 for 3x3x2.

    # Actually, let me just use the schoolbook for now as a safe baseline, but try to
    # find a better decomposition via the candidate dictionary approach.

    # Use a simple greedy: start with residual = T, iteratively pick the candidate
    # that has the largest inner product with residual, subtract it, repeat.
    # This won't give exact integer solutions in general, but let's try.

    # Better: use numpy lstsq to find a real solution, then round.

    # Let's try: pick R candidates, solve for real coefficients, check if rounding works.

    # Start with R = 50
    for R in range(50, 73):
        # Pick R candidates: first 72 are schoolbook, rest are random
        # Use a mix
        idx_list = list(range(min(R, 72)))  # schoolbook first
        if R > 72:
            idx_list.extend(range(72, R))
        else:
            # Add some random ones to replace some schoolbook
            random_idx = rng.choice(len(candidates), R - min(R, 72), replace=False)
            for ri in random_idx:
                if ri not in idx_list:
                    idx_list.append(ri)
                    if len(idx_list) >= R:
                        break

        idx_list = idx_list[:R]
        A_mat = cand_matrix[:, idx_list]  # shape (5184, R)

        # Solve A_mat @ x = T using least squares
        x, residuals, rank_val, sv = np.linalg.lstsq(A_mat, T, rcond=None)

        # Check if x is close to integers
        x_rounded = np.round(x)
        error = np.max(np.abs(x_rounded - x))
        if error < 1e-6:
            # Verify exactness
            recon = A_mat @ x_rounded
            if np.array_equal(recon, T):
                # Build U, V, W
                U = np.zeros((nm, R), dtype=int)
                V = np.zeros((mp, R), dtype=int)
                W = np.zeros((pn, R), dtype=int)
                for r_idx, ci in enumerate(idx_list):
                    u, v, w = candidates[ci]
                    U[:, r_idx] = u * x_rounded[r_idx]
                    V[:, r_idx] = v * x_rounded[r_idx]
                    W[:, r_idx] = w * x_rounded[r_idx]
                # Remove zero columns
                nonzero = np.any(U != 0, axis=0) | np.any(V != 0, axis=0) | np.any(W != 0, axis=0)
                U = U[:, nonzero]
                V = V[:, nonzero]
                W = W[:, nonzero]
                actual_rank = U.shape[1]
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook rank 72
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
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
