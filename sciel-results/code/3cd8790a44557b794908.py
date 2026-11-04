"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit

    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p

    # Construct Schoolbook Decomposition
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    U_school = np.zeros((n * m, rank_schoolbook), dtype=np.int64)
    V_school = np.zeros((m * p, rank_schoolbook), dtype=np.int64)
    W_school = np.zeros((p * n, rank_schoolbook), dtype=np.int64)

    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, column] = 1
                V_school[j * p + k, column] = 1
                W_school[k * n + i, column] = 1
                column += 1

    # Quick verification
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_school[i*m+j, :] * V_school[j*p+k, :] * W_school[k*n+i, :])
                if val != 1:
                    raise ValueError("Schoolbook decomposition failed verification")

    # Attempt to find a lower-rank decomposition by merging pairs of rank-1 terms.
    # Two schoolbook terms (i1,j1,k1) and (i2,j2,k2) can be replaced by a single
    # rank-1 term if their sum tensor is rank-1.
    #
    # The sum tensor T = e_{i1,j1,k1} + e_{i2,j2,k2} is rank-1 iff:
    # - The supports in each dimension have at most 2 distinct values
    # - And the tensor can be written as u ⊗ v ⊗ w for some vectors u,v,w
    #
    # For the matrix multiplication tensor structure, a sum of two basis tensors
    # e_{i1,j1,k1} + e_{i2,j2,k2} is rank-1 iff:
    #   (i1==i2 and j1==j2) or (i1==i2 and k1==k2) or (j1==j2 and k1==k2)
    # In other words, they must share at least 2 of the 3 indices.
    #
    # If they share exactly 2 indices, the merged term has half-integer coefficients.
    # If they share all 3, they're the same term (can't merge).
    #
    # Let's find all mergeable pairs and use a greedy matching to maximize merges.

    if time.time() > deadline - 10:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

    # For each pair of schoolbook indices, check if they can be merged
    # Schoolbook term c corresponds to (i, j, k) where c = i*m*p + j*p + k
    # Actually, let's re-index: term index t corresponds to (i,j,k) with
    # t = i*(m*p) + j*p + k

    # Find all pairs (t1, t2) with t1 < t2 that can be merged
    mergeable_pairs = []
    for t1 in range(rank_schoolbook):
        i1 = t1 // (m * p)
        rem = t1 % (m * p)
        j1 = rem // p
        k1 = rem % p

        for t2 in range(t1 + 1, rank_schoolbook):
            i2 = t2 // (m * p)
            rem2 = t2 % (m * p)
            j2 = rem2 // p
            k2 = rem2 % p

            # Check if they share at least 2 indices
            shared = 0
            if i1 == i2:
                shared += 1
            if j1 == j2:
                shared += 1
            if k1 == k2:
                shared += 1

            if shared >= 2:
                mergeable_pairs.append((t1, t2))

    # Greedy matching: select non-conflicting mergeable pairs
    # A pair (a,b) conflicts with (c,d) if any of a,b,c,d are shared
    used = set()
    selected_pairs = []
    for t1, t2 in mergeable_pairs:
        if t1 not in used and t2 not in used:
            selected_pairs.append((t1, t2))
            used.add(t1)
            used.add(t2)

    if not selected_pairs:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

    # Build the new decomposition
    # Each selected pair becomes one new term; each unused term stays as is
    new_terms = []  # list of (type, data) where type is 'single' or 'merged'

    for t1, t2 in selected_pairs:
        new_terms.append(('merged', (t1, t2)))
    for t in range(rank_schoolbook):
        if t not in used:
            new_terms.append(('single', t))

    new_rank = len(new_terms)

    # Build new factor matrices
    U_new = np.zeros((n * m, new_rank), dtype=np.float64)
    V_new = np.zeros((m * p, new_rank), dtype=np.float64)
    W_new = np.zeros((p * n, new_rank), dtype=np.float64)

    for c, (typ, data) in enumerate(new_terms):
        if typ == 'single':
            t = data
            i = t // (m * p)
            rem = t % (m * p)
            j = rem // p
            k = rem % p
            U_new[i * m + j, c] = 1.0
            V_new[j * p + k, c] = 1.0
            W_new[k * n + i, c] = 1.0
        else:  # merged
            t1, t2 = data
            i1 = t1 // (m * p)
            rem1 = t1 % (m * p)
            j1 = rem1 // p
            k1 = rem1 % p

            i2 = t2 // (m * p)
            rem2 = t2 % (m * p)
            j2 = rem2 // p
            k2 = rem2 % p

            # The sum tensor is e_{i1,j1,k1} + e_{i2,j2,k2}
            # We need to express this as u ⊗ v ⊗ w
            # Since they share 2 indices, the non-shared index has 2 values
            # The rank-1 decomposition uses half-integer coefficients

            if i1 == i2:
                # Share i, differ in j and k
                # But wait, if they share i but differ in both j and k,
                # that's only sharing 1 index. Let me recheck.
                # Actually if i1==i2 and j1!=j2 and k1!=k2, shared=1, not mergeable.
                # So this case shouldn't occur if shared>=2.
                # If i1==i2 and j1==j2, then k1!=k2 (since t1!=t2)
                # If i1==i2 and k1==k2, then j1!=j2
                if j1 == j2:
                    # Share i and j, differ in k
                    # u has support on i1, v on j1, w on {k1, k2}
                    U_new[i1 * m + j1, c] = 1.0
                    V_new[j1 * p + k1, c] = 0.5
                    V_new[j1 * p + k2, c] = 0.5
                    W_new[k1 * n + i1, c] = 1.0
                    W_new[k2 * n + i1, c] = 1.0
                elif k1 == k2:
                    # Share i and k, differ in j
                    U_new[i1 * m + j1, c] = 0.5
                    U_new[i1 * m + j2, c] = 0.5
                    V_new[j1 * p + k1, c] = 1.0
                    W_new[k1 * n + i1, c] = 1.0
            elif j1 == j2:
                # Share j, differ in i and k
                if i1 == i2:
                    # Already handled
                    pass
                else:
                    # j1==j2, i1!=i2, k1!=k2 -> shared=1, shouldn't be here
                    pass
            elif k1 == k2:
                # Share k, differ in i and j
                if i1 == i2:
                    pass
                elif j1 == j2:
                    pass
                else:
                    # k1==k2, i1!=i2, j1!=j2 -> shared=1, shouldn't be here
                    pass

            # Let me redo this more carefully
            # Clear and rebuild
            U_new[i1 * m + j1, c] = 0
            U_new[i2 * m + j2, c] = 0
            V_new[j1 * p + k1, c] = 0
            V_new[j2 * p + k2, c] = 0
            W_new[k1 * n + i1, c] = 0
            W_new[k2 * n + i2, c] = 0

            if i1 == i2 and j1 == j2:
                # Share i and j, k differs
                U_new[i1 * m + j1, c] = 1.0
                V_new[j1 * p + k1, c] = 0.5
                V_new[j1 * p + k2, c] = 0.5
                W_new[k1 * n + i1, c] = 1.0
                W_new[k2 * n + i1, c] = 1.0
            elif i1 == i2 and k1 == k2:
                # Share i and k, j differs
                U_new[i1 * m + j1, c] = 0.5
                U_new[i1 * m + j2, c] = 0.5
                V_new[j1 * p + k1, c] = 1.0
                W_new[k1 * n + i1, c] = 1.0
            elif j1 == j2 and k1 == k2:
                # Share j and k, i differs
                U_new[i1 * m + j1, c] = 1.0
                V_new[j1 * p + k1, c] = 0.5
                V_new[j2 * p + k2, c] = 0.5
                W_new[k1 * n + i1, c] = 1.0
                W_new[k2 * n + i2, c] = 1.0

    # Verify the new decomposition
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_new[i*m+j, :] * V_new[j*p+k, :] * W_new[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

    # Check time
    if time.time() > deadline - 5:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

    # Convert to lists
    return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
# EVOLVE-BLOCK-END
