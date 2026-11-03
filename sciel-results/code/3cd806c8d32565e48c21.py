"""Rewritten program: structured sparse linear algebra / binary IP for exact tensor decomposition."""
DIMENSIONS = [3, 3, 8]
import random
random.seed(1)


# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    from itertools import product

    n, m, p = 3, 3, 8
    nm, mp, pn = n * m, m * p, p * n  # 9, 24, 24

    # The target tensor T[a, b, c] = 1 iff a = i*m+j, b = j*p+k, c = k*n+i for some (i,j,k), else 0.
    # We need: T = sum_r u_r ⊗ v_r ⊗ w_r, i.e., T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]

    # Build a dictionary of candidate rank-1 terms.
    # Each candidate is (u, v, w) with small integer entries.
    # Key insight: for matrix multiplication, natural candidates have structure:
    # u has nonzeros at positions (i,j), v at (j,k), w at (k,i).
    # Strassen-style: u,v,w can have multiple nonzeros that create cross terms canceling out.

    # Strategy: generate candidates with up to 'max_nnz' nonzeros per vector,
    # values in {-1, 0, 1, 2}, then use greedy set-cover on the 72 target equations
    # while ensuring all 5184-72=5112 off-target equations are satisfied.

    # Actually, a cleaner approach: use the bilinear structure.
    # For each candidate term (u, v, w), it contributes to T[a,b,c] += u[a]*v[b]*w[c].
    # We need the sum over selected terms to equal T exactly.

    # Let's build candidates systematically:
    # Type 1: Schoolbook-like - u has 1 nonzero at (i,j), v has 1 at (j,k), w has 1 at (k,i)
    # Type 2: Strassen-like - u has 2 nonzeros, v has 2, w has 2, arranged so cross terms cancel
    # Type 3: Mixed patterns

    # Generate all candidates with small support
    candidates = []

    # Type 1: single-entry (schoolbook)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0]*nm; v = [0]*mp; w = [0]*pn
                u[i*m+j] = 1
                v[j*p+k] = 1
                w[k*n+i] = 1
                candidates.append((tuple(u), tuple(v), tuple(w)))

    # Type 2: Strassen-style for the 3x3x3 sub-blocks
    # For 3x3x3, Strassen uses 7 terms instead of 9.
    # The 7 Strassen terms for multiplying 3x3 by 3x3:
    # Let A = [[a11,a12,a13],[a21,a22,a23],[a31,a32,a33]]
    # B = [[b11,b12,b13],[b21,b22,b23],[b31,b32,b33]]
    #
    # Strassen for 3x3 (one known 7-term decomposition):
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a33(b21+b33)
    # M5 = (a13+a33)b31
    # M6 = (a31-a11)(b11+b12)
    # M7 = (a22-a32)(b33+b31)
    #
    # Wait, the standard Strassen for 3x3 isn't simply 7. Let me use a verified 23-term for 4x4x4
    # and embed 3x3x8 into it.

    # Better approach: Use 4x4x4 Strassen-Laderman (rank 23) embedded in 4x4x8 (rank 46).
    # Pad A from 3x3 to 4x4, B from 3x8 to 4x8, compute, extract 3x3*3x8 result.
    # The 4x4x4 rank-23 decomposition:

    # Strassen-Laderman 4x4x4, rank 23.
    # I'll construct it from the known formula.
    # For 4x4x4, the 23 terms are indexed by (i,j,k) with i+j+k < 4 (excluding certain ones).
    # Actually, let me use a well-known explicit construction.

    # The 23 Strassen-Laderman terms for 4x4x4:
    # Using the formulation where each term corresponds to a "diagonal" pattern.
    # Let me use the explicit matrices from the literature.

    # Alternative: build 3x3x8 directly using a combination of:
    # - Two 3x3x4 blocks, each using a rank-12 decomposition (if exists)
    # - Or use the 4x4x4 embedding

    # Let me try the 4x4x4 embedding approach with a hardcoded rank-23 decomposition.

    # 4x4x4 Strassen-Laderman rank 23:
    # U4: 16x23, V4: 16x23, W4: 16x23
    # Row indices: U4[i*4+j, r], V4[j*4+k, r], W4[k*4+i, r]

    # I'll use the following known 23-term decomposition for 4x4x4:
    # Terms are parameterized by (a,b,c) where the pattern is based on diagonals.

    # Let me use a simpler approach: generate candidates with small integer entries
    # and use a greedy algorithm to cover the tensor.

    # Generate diverse candidates:
    # For each combination of small supports on u, v, w with values in {-2,-1,0,1,2}
    # such that the "diagonal" structure is preserved (nonzero positions are consistent).

    # A term (u,v,w) is "valid" if for every nonzero u[a]*v[b]*w[c], the indices
    # satisfy a=i*m+j, b=j*p+k, c=k*n+i for some (i,j,k). This ensures no off-target contributions.
    # But Strassen terms don't satisfy this - they have off-target contributions that cancel.

    # So I need to allow terms with off-target contributions, as long as they cancel in the sum.

    # Greedy approach:
    # 1. Start with all 72 target equations unsatisfied (need value 1).
    # 2. All 5112 off-target equations need value 0 (start at 0).
    # 3. At each step, pick the candidate term that maximally reduces the total error.
    # 4. Continue until all equations are satisfied or no improvement possible.

    # This is expensive. Let me limit to a smaller candidate set.

    # Generate candidates with support size up to 3 on each vector, values in {-1,0,1}:
    # This gives C(9,<=3)*3^3 * C(24,<=3)*3^3 * C(24,<=3)*3^3 which is too large.

    # Instead, use structured candidates:
    # For each pair of "diagonals" in the 3x3 grid, create Strassen-like terms.

    # Let me use a different strategy: exploit the fact that 3x3x8 = 3x3x3 + 3x3x3 + 3x3x2
    # and find good decompositions for each part, then combine.

    # For 3x3x3: best known rank is 23 (but that's for 4x4x4... no, 3x3x3 rank is 23? No.)
    # Actually, the tensor rank of 3x3x3 matrix multiplication is 23. Yes!
    # Wait, no. The bilinear complexity of 3x3x3 is 23. Yes, that's correct.

    # So 3x3x3 has rank 23. Then 3x3x8 = 3x3x3 + 3x3x3 + 3x3x2.
    # 3x3x2 has rank at most 18 (schoolbook), possibly less.
    # Total: 23 + 23 + 18 = 64. Still better than 72.

    # But I need the actual 23-term decomposition for 3x3x3.

    # The 23-term decomposition of 3x3x3 (Strassen):
    # This is complex. Let me look up or construct it.

    # Actually, let me use a computational approach:
    # Generate a large pool of candidate terms, then use a greedy set-cover
    # with verification.

    # Candidate generation:
    # For each "pattern" of nonzeros on the 3x3 grids:
    # u: subset of {(i,j) for i,j in 0..2} with values in {-1,1}
    # v: subset of {(j,k) for j,k in 0..2} with values in {-1,1}
    # w: subset of {(k,i) for k,i in 0..2} with values in {-1,1}
    # But we also need terms that span multiple k values for the 3x3x8 case.

    # For 3x3x8, the structure is:
    # u: subset of {(i,j) for i,j in 0..2}
    # v: subset of {(j,k) for j in 0..2, k in 0..7}
    # w: subset of {(k,i) for k in 0..7, i in 0..2}

    # Let me generate candidates where each vector has at most 2 nonzeros,
    # values in {-1, 1}, and the nonzero positions are "compatible" in the sense
    # that they form a connected subgraph in the bipartite graph interpretation.

    # This is getting complex. Let me use a pragmatic approach:
    # Use the 4x4x4 Strassen-Laderman decomposition (rank 23) and embed 3x3x8 into 4x4x8.

    # I'll hardcode the 4x4x4 rank-23 decomposition.
    # From the literature, one explicit 23-term decomposition:

    # The 23 terms for 4x4x4 (Strassen-Laderman):
    # I'll use the formulation from "The rank of 4x4x4 matrix multiplication is 23"

    # Let me construct it programmatically using the known pattern:
    # For 4x4x4, the 23 terms correspond to the 23 "minimal" patterns.

    # Actually, let me just use a verified hardcoded set.
    # Here's a known 23-term decomposition for 4x4x4:

    # I'll use the one from the paper by Strassen (1973) / Laderman (1976).
    # The terms are:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A33(B21+B33)
    # M5 = (A13+A33)B31
    # M6 = (A31-A11)(B11+B12)
    # M7 = (A22-A32)(B33+B31)
    # ... and so on for 4x4.

    # This is getting too complex to hardcode correctly from memory.
    # Let me use a different strategy: random search with verification.

    # Generate R random sparse terms, check if they can be adjusted to satisfy the tensor.
    # If not, try different R.

    # Given time constraints, let me use a safe approach:
    # Use schoolbook rank 72 but try to find a smaller rank via greedy elimination.

    # Actually, let me try a focused approach:
    # 1. Generate 5000 random candidate terms with small support and integer values.
    # 2. Use a greedy algorithm to select a subset that exactly satisfies the tensor.
    # 3. If successful, return the selected terms. Otherwise, fallback to schoolbook.

    np.random.seed(42)
    max_candidates = 5000
    candidates = []

    # Generate structured candidates
    # Type A: Single nonzero in each vector (schoolbook-like)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0]*nm; v = [0]*mp; w = [0]*pn
                u[i*m+j] = 1
                v[j*p+k] = 1
                w[k*n+i] = 1
                candidates.append((tuple(u), tuple(v), tuple(w)))

    # Type B: Two nonzeros in each vector, with values +/-1
    # For u: pick two positions (i1,j1) and (i2,j2)
    # For v: pick two positions (j1,k1) and (j2,k2) where j1,j2 match i1,i2 pattern
    # For w: pick two positions (k1,i1) and (k2,i2)
    # This creates Strassen-like cross terms.

    for i1 in range(n):
        for i2 in range(n):
            if i1 == i2:
                continue
            for j1 in range(m):
                for j2 in range(m):
                    # u has nonzeros at (i1,j1) and (i2,j2)
                    for k1 in range(p):
                        for k2 in range(p):
                            # v has nonzeros at (j1,k1) and (j2,k2)
                            # w has nonzeros at (k1,i1) and (k2,i2)
                            for s1 in [-1, 1]:
                                for s2 in [-1, 1]:
                                    for t1 in [-1, 1]:
                                        for t2 in [-1, 1]:
                                            for r1 in [-1, 1]:
                                                for r2 in [-1, 1]:
                                                    u = [0]*nm; v = [0]*mp; w = [0]*pn
                                                    u[i1*m+j1] = s1
                                                    u[i2*m+j2] = s2
                                                    v[j1*p+k1] = t1
                                                    v[j2*p+k2] = t2
                                                    w[k1*n+i1] = r1
                                                    w[k2*n+i2] = r2
                                                    candidates.append((tuple(u), tuple(v), tuple(w)))

    # Type C: Three nonzeros in each vector
    # This is too many. Skip for now.

    # Deduplicate
    candidate_set = set()
    for c in candidates:
        candidate_set.add(c)
    candidates = list(candidate_set)

    # Now use greedy set-cover:
    # We need to satisfy 72 equations (target = 1) and 5112 equations (target = 0).
    # Total 5184 equations.

    # For each candidate, compute its contribution vector (length 5184).
    # This is expensive. Instead, track only the 72 target equations and verify at the end.

    # Greedy: at each step, pick the candidate that maximally reduces the L1 error
    # on the 72 target equations, while not increasing the error on off-target equations too much.

    # Actually, let me use a simpler approach:
    # 1. Start with empty set, current_sum = 0 (all zeros).
    # 2. Target: 72 ones, 5112 zeros.
    # 3. At each step, for each candidate, compute new_error = |current_sum + candidate - target|.
    # 4. Pick the candidate with minimum new_error.
    # 5. Repeat until error = 0 or max iterations.

    # This is O(|candidates| * 5184) per step, which is ~5000 * 5184 = 25.9M per step.
    # With 72 steps, that's ~1.8B operations. Too slow in Python.

    # Let me limit to a smaller candidate pool and use numpy for speed.

    # Use only the first 2000 candidates.
    candidates = candidates[:2000]

    # Convert to numpy arrays for speed
    U_cand = np.array([c[0] for c in candidates], dtype=np.int8)  # (num_cand, nm)
    V_cand = np.array([c[1] for c in candidates], dtype=np.int8)  # (num_cand, mp)
    W_cand = np.array([c[2] for c in candidates], dtype=np.int8)  # (num_cand, pn)

    # Target tensor as a flat array
    target = np.zeros(nm * mp * pn, dtype=np.int8)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a * mp * pn + b * pn + c] = 1

    current_sum = np.zeros(nm * mp * pn, dtype=np.int32)

    selected = []
    max_steps = 80

    for step in range(max_steps):
        if np.all(current_sum == target):
            break

        # For each candidate, compute the new error if we add it
        # new_sum = current_sum + candidate_contribution
        # error = |new_sum - target|

        # Candidate contribution: for candidate r, contribution[a,b,c] = U_cand[r,a] * V_cand[r,b] * W_cand[r,c]
        # This is a 3D tensor. Computing all candidates' contributions is expensive.

        # Instead, compute the error reduction for each candidate:
        # delta_error = |current_sum + contrib - target| - |current_sum - target|

        # Let me compute this incrementally.
        best_idx = -1
        best_error = np.sum(np.abs(current_sum - target))

        for r in range(len(candidates)):
            # Compute contribution of candidate r
            contrib = np.einsum('a,b,c->abc', U_cand[r], V_cand[r], W_cand[r])
            contrib_flat = contrib.flatten()

            new_sum = current_sum + contrib_flat
            new_error = np.sum(np.abs(new_sum - target))

            if new_error < best_error:
                best_error = new_error
                best_idx = r

        if best_idx == -1:
            break

        # Add candidate best_idx
        contrib = np.einsum('a,b,c->abc', U_cand[best_idx], V_cand[best_idx], W_cand[best_idx])
        current_sum += contrib.flatten()
        selected.append(best_idx)

    # Check if we achieved exact decomposition
    if np.all(current_sum == target):
        R = len(selected)
        U = np.zeros((nm, R), dtype=np.int32)
        V = np.zeros((mp, R), dtype=np.int32)
        W = np.zeros((pn, R), dtype=np.int32)
        for col, r in enumerate(selected):
            U[:, col] = U_cand[r]
            V[:, col] = V_cand[r]
            W[:, col] = W_cand[r]
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
