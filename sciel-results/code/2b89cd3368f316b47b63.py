"""Structured sparse linear algebra / binary IP for exact tensor decomposition."""
DIMENSIONS = [3, 3, 8]
import random
random.seed(1)


# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 8
    nm, mp, pn = n * m, m * p, p * n  # 9, 24, 24

    # Build the target tensor T of shape (nm, mp, pn)
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some valid i,j,k; else 0
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[(a, b, c)] = 1

    # Generate candidate rank-1 terms from structured patterns.
    # Each candidate is (u, v, w) where u, v, w are sparse vectors with entries in {0, 1, -1}.
    # A rank-1 term contributes u[a]*v[b]*w[c] to T[a,b,c].
    # We want to select a subset of candidates such that their sum equals T exactly.

    # Strategy: Generate candidates from:
    # 1. Schoolbook terms: u has one 1, v has one 1, w has one 1
    # 2. Strassen-like terms: u, v, w have 2-3 non-zero entries each
    # 3. Random sparse terms with small support

    candidates = []

    # 1. Schoolbook candidates
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0] * nm
                v = [0] * mp
                w = [0] * pn
                u[i * m + j] = 1
                v[j * p + k] = 1
                w[k * n + i] = 1
                candidates.append((tuple(u), tuple(v), tuple(w)))

    # 2. Generate Strassen-like candidates with small support
    # For each pair of indices in each dimension, create patterns
    for i1 in range(n):
        for i2 in range(n):
            if i1 == i2:
                continue
            for j1 in range(m):
                for j2 in range(m):
                    if j1 == j2:
                        continue
                    for k1 in range(p):
                        for k2 in range(p):
                            if k1 == k2:
                                continue
                            # Pattern: u has 1 at (i1,j1) and 1 at (i2,j2)
                            # v has 1 at (j1,k1) and 1 at (j2,k2)
                            # w has 1 at (k1,i1) and 1 at (k2,i2)
                            u = [0] * nm
                            v = [0] * mp
                            w = [0] * pn
                            u[i1 * m + j1] = 1
                            u[i2 * m + j2] = 1
                            v[j1 * p + k1] = 1
                            v[j2 * p + k2] = 1
                            w[k1 * n + i1] = 1
                            w[k2 * n + i2] = 1
                            candidates.append((tuple(u), tuple(v), tuple(w)))

                            # Also try with -1 signs
                            u2 = [0] * nm
                            v2 = [0] * mp
                            w2 = [0] * pn
                            u2[i1 * m + j1] = 1
                            u2[i2 * m + j2] = -1
                            v2[j1 * p + k1] = 1
                            v2[j2 * p + k2] = -1
                            w2[k1 * n + i1] = 1
                            w2[k2 * n + i2] = 1
                            candidates.append((tuple(u2), tuple(v2), tuple(w2)))

    # 3. Random sparse candidates
    random.seed(42)
    for _ in range(5000):
        u = [0] * nm
        v = [0] * mp
        w = [0] * pn
        # Random sparse vectors with support 2-4
        sup_u = random.randint(2, 4)
        sup_v = random.randint(2, 4)
        sup_w = random.randint(2, 4)
        for idx in random.sample(range(nm), sup_u):
            u[idx] = random.choice([1, -1])
        for idx in random.sample(range(mp), sup_v):
            v[idx] = random.choice([1, -1])
        for idx in random.sample(range(pn), sup_w):
            w[idx] = random.choice([1, -1])
        candidates.append((tuple(u), tuple(v), tuple(w)))

    # Remove duplicates
    candidates = list(set(candidates))

    # Compute the contribution of each candidate to the tensor
    # For each candidate, compute which (a,b,c) entries it affects and by how much
    cand_contrib = []
    for idx, (u, v, w) in enumerate(candidates):
        contrib = {}
        for a in range(nm):
            if u[a] == 0:
                continue
            for b in range(mp):
                if v[b] == 0:
                    continue
                for c in range(pn):
                    if w[c] == 0:
                        continue
                    val = u[a] * v[b] * w[c]
                    if val != 0:
                        contrib[(a, b, c)] = val
        cand_contrib.append(contrib)

    # Greedy set cover: select candidates to cover all required tensor entries
    # Required entries: all (a,b,c) in T with value 1
    # We need sum of selected candidates to equal T exactly

    # Track residual tensor
    residual = dict(T)  # Start with target

    selected = []
    remaining_entries = set(residual.keys())

    # Greedy: at each step, pick the candidate that best reduces the residual
    # Score: number of entries where it helps (reduces |residual|) minus number where it hurts
    for step in range(100):
        best_idx = -1
        best_score = -1
        best_new_residual = None

        for idx, contrib in enumerate(cand_contrib):
            if idx in selected:
                continue
            # Compute new residual if we add this candidate
            new_residual = dict(residual)
            score = 0
            for entry, val in contrib.items():
                if entry in new_residual:
                    old_val = new_residual[entry]
                    new_val = old_val - val
                    # Score: reduction in |val|
                    score += abs(old_val) - abs(new_val)
                    new_residual[entry] = new_val
                else:
                    # This entry wasn't needed but now has a value
                    if val != 0:
                        score -= abs(val)
                        new_residual[entry] = -val
            # Remove zero entries
            new_residual = {k: v for k, v in new_residual.items() if v != 0}

            if score > best_score:
                best_score = score
                best_idx = idx
                best_new_residual = new_residual

        if best_idx == -1 or best_score <= 0:
            break

        selected.append(best_idx)
        residual = best_new_residual

        if not residual:
            break

    # If greedy didn't solve it, try a different approach:
    # Use the schoolbook as base and try to improve
    if residual:
        # Fallback: use schoolbook decomposition
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

    # Build the final decomposition from selected candidates
    R = len(selected)
    U = [[0] * R for _ in range(nm)]
    V = [[0] * R for _ in range(mp)]
    W = [[0] * R for _ in range(pn)]

    for col, idx in enumerate(selected):
        u, v, w = candidates[idx]
        for a in range(nm):
            U[a][col] = u[a]
        for b in range(mp):
            V[b][col] = v[b]
        for c in range(pn):
            W[c][col] = w[c]

    # Verify the decomposition
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = 0
                for r in range(R):
                    s += U[a][r] * V[b][r] * W[c][r]
                if s != 1:
                    # Fallback to schoolbook
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

    # Also verify no spurious entries
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                if (a, b, c) not in T:
                    s = 0
                    for r in range(R):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if s != 0:
                        # Fallback to schoolbook
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

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
