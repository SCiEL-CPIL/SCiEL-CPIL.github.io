"""Tensor decomposition for matrix multiplication 3x3 * 3x8.
Structural change: instead of Strassen 2x2 blocks + greedy elimination,
use a global linear-algebraic rank-reduction approach.

Key insight: the tensor for 3x3*3x8 matrix multiplication has rank at most
n*m*p = 72 (schoolbook). We build the schoolbook decomposition, then try to
reduce rank by finding integer linear dependencies among the tensor vectors
of the terms. The tensor vector for term (i,j,k) is the vector in R^(n*m*p)
that has 1 at position (i,j,k) and 0 elsewhere. But that's just the standard
basis, so there are no dependencies.

The real structure: each term in a tensor decomposition is a RANK-1 tensor
(u ⊗ v ⊗ w). The matrix multiplication tensor T has entries T[i,j,k] = 1 if
the term A[i,j]*B[j,k] contributes to C[i,k]. For the full multiplication,
T[i,j,k] = delta(i,k) where the multiplication maps (i,j) x (j,k) -> (i,k).

Actually, the multiplication tensor for (n,m,p) has:
T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B multiply to give
the (i,k) entry of C. So T[i,j,k] = 1 for all valid (i,j,k) where j is
shared. Wait, that's not right either.

Let me re-derive. The matrix multiplication tensor is defined as:
T ∈ R^{(n*m) x (m*p) x (p*n)} where
T[(i,j), (j,k), (i,k)] = 1 and all other entries are 0.

So the tensor has exactly n*m*p non-zero entries, each equal to 1, at positions
[(i*m+j), (j*p+k), (k*n+i)] for i in range(n), j in range(m), k in range(p).

A rank-r decomposition is: T = sum_{s=1}^{r} U[:,s] ⊗ V[:,s] ⊗ W[:,s]
where U is (n*m) x r, V is (m*p) x r, W is (p*n) x r.

This means: for all (a, b, c), T[a,b,c] = sum_s U[a,s]*V[b,s]*W[c,s].

The schoolbook decomposition has r = n*m*p with each term being a single
non-zero entry. To reduce rank, we need to find combinations of rank-1 tensors
that sum to T.

New approach: Use the fact that for small dimensions, we can set up the
problem as finding integer solutions. Build the schoolbook, then look for
integer linear relations among the rank-1 tensors.

A rank-1 tensor u ⊗ v ⊗ w contributes to T at position (a,b,c) the value
u[a]*v[b]*w[c]. For the schoolbook term (i,j,k), u has 1 at (i,j), v has 1
at (j,k), w has 1 at (i,k).

To find a lower-rank decomposition, we can try to express the tensor as a sum
of fewer rank-1 tensors. This is the tensor rank problem, which is NP-hard
in general, but for small dimensions we can search.

Practical approach: Start with schoolbook (rank 72). Try to eliminate terms
by finding integer relations. For each term r, check if its tensor vector can
be expressed as an integer combination of others. If so, absorb it.

But the key structural change: instead of trying to eliminate one term at a
time (which is what the current code does), use a SIMULTANEOUS elimination
approach. Find a set of terms that can be simultaneously eliminated by
modifying the remaining terms.

Actually, let me think about this differently. The tensor for 3x3*3x8 is:
T[(i,j),(j,k),(i,k)] = 1

This is a 3-tensor of dimensions 9 x 24 x 24.

Known bounds: For 3x3*3x3, the rank is 19 (Landsberg). For 3x3*3x8, we can
use the 3x3*3x3 result and extend.

Let me try a different structural approach: use the known rank-19
decomposition for 3x3*3x3 (if we can find one) and extend it to 3x3*3x8.

Actually, for 3x3*3x3, the multiplication tensor has rank 19 (this is known).
For 3x3*3x8, we can think of it as 3x3*3x3 + 3x3*3x5. The 3x3*3x5 part has
rank at most 3*3*5 = 45 by schoolbook, but can be reduced.

Better: 3x3*3x8 = (3x3*3x3) + (3x3*3x5). If 3x3*3x3 has rank 19 and
3x3*3x5 has rank at most 45, total is at most 64. But we can do better.

Actually, let me just implement a more aggressive search. The key insight for
the structural change: instead of greedy elimination, use a global optimization
approach where we simultaneously try to reduce multiple terms.

Let me implement:
1. Build schoolbook decomposition (rank 72)
2. Use integer linear algebra to find the minimum rank
3. For each subset of terms to keep, check if the tensor can be represented

This is exponential, so we need heuristics. Let me try:
1. Start with schoolbook
2. Repeatedly find the term that is most "redundant" (most linearly
   dependent on others) and eliminate it
3. Use a better redundancy measure than just checking if one term is an
   integer combo of all others

The key improvement: instead of checking if term r is an integer combination
of ALL other terms (which is a large system), check if it's an integer
combination of a SMALL subset of other terms. This is much more likely to
succeed and much faster.

Let me implement this with a focus on small subsets (size 2-4).
"""

import time
import numpy as np
from itertools import combinations

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = 140.0

    N = n * m  # 9
    M = m * p  # 24
    P = p * n  # 24
    total_dim = N * M * P  # 5184

    def build_schoolbook():
        """Build the schoolbook (rank n*m*p) decomposition."""
        rank = n * m * p
        U = np.zeros((N, rank))
        V = np.zeros((M, rank))
        W = np.zeros((P, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank

    def term_tensor(U, V, W, r):
        """Get the flattened tensor vector for term r."""
        vec = np.zeros(total_dim)
        for a in range(N):
            if U[a, r] == 0:
                continue
            for b in range(M):
                if V[b, r] == 0:
                    continue
                for c in range(P):
                    if W[c, r] == 0:
                        continue
                    val = U[a, r] * V[b, r] * W[c, r]
                    if val != 0:
                        vec[a * M * P + b * P + c] = val
        return vec

    def verify(U, V, W):
        """Verify the decomposition is exact."""
        # T[(i,j),(j,k),(i,k)] should equal 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(U.shape[1]):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        # Also check that other entries are 0
        # This is expensive, so skip for now and rely on the structure
        return True

    def verify_full(U, V, W):
        """Full verification including zero entries."""
        rank = U.shape[1]
        # Build the full tensor from the decomposition
        T = np.zeros((N, M, P))
        for r in range(rank):
            T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        # Check that T[(i,j),(j,k),(i,k)] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if abs(T[i*m+j, j*p+k, k*n+i] - 1.0) > 1e-9:
                        return False
        # Check that all other entries are 0
        # This is O(N*M*P) which is 5184, fast enough
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 0
        if np.any(np.abs(T) > 1e-9):
            return False
        return True

    def try_eliminate_term(U, V, W, r, rank):
        """Try to eliminate term r by finding integer combination of others.
        Returns (new_U, new_V, new_W) or None."""
        others = [s for s in range(rank) if s != r]
        if not others:
            return None

        target = term_tensor(U, V, W, r)
        if not np.any(np.abs(target) > 1e-12):
            # Term is already zero, just remove it
            mask = np.ones(rank, dtype=bool)
            mask[r] = False
            return U[:, mask], V[:, mask], W[:, mask]

        # Try small subsets of others
        # First try: is target = c * term_s for some single s?
        for s in others:
            vs = term_tensor(U, V, W, s)
            if not np.any(np.abs(vs) > 1e-12):
                continue
            # Check if target = c * vs
            nz = np.nonzero(np.abs(vs) > 1e-12)[0]
            if len(nz) == 0:
                continue
            c = target[nz[0]] / vs[nz[0]]
            c2 = round(c * 2) / 2
            if abs(c - c2) < 1e-9:
                # target = c2 * vs
                # We can absorb: new term s has coefficient (1 + c2)
                # Actually, in the sum, term r contributes target, term s
                # contributes vs. If target = c2 * vs, then we can remove
                # term r and add c2 to the "weight" of term s. But in our
                # decomposition, each term has weight 1. So we need to
                # scale term s by (1 + c2) and remove term r.
                # But scaling a term means scaling U[:,s], V[:,s], W[:,s]
                # by (1+c2). Let's check if this gives integer/half-integer
                # entries.
                new_U = U.copy()
                new_V = V.copy()
                new_W = W.copy()
                scale = 1.0 + c2
                new_U[:, s] *= scale
                new_V[:, s] *= scale
                new_W[:, s] *= scale
                # Remove term r
                mask = np.ones(rank, dtype=bool)
                mask[r] = False
                new_U = new_U[:, mask]
                new_V = new_V[:, mask]
                new_W = new_W[:, mask]
                if verify_full(new_U, new_V, new_W):
                    return new_U, new_V, new_W

        # Try pairs: target = c1 * vs1 + c2 * vs2
        for s1, s2 in combinations(others, 2):
            vs1 = term_tensor(U, V, W, s1)
            vs2 = term_tensor(U, V, W, s2)
            # Solve [vs1, vs2] * [c1, c2]^T = target
            # Use least squares
            A = np.column_stack([vs1, vs2])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, target, rcond=None)
            except Exception:
                continue
            coeffs_int = np.round(coeffs * 2) / 2
            if np.all(np.abs(coeffs - coeffs_int) < 1e-9):
                # Check if this gives valid new matrices
                new_U = U.copy()
                new_V = V.copy()
                new_W = W.copy()
                new_U[:, s1] *= (1.0 + coeffs_int[0])
                new_V[:, s1] *= (1.0 + coeffs_int[0])
                new_W[:, s1] *= (1.0 + coeffs_int[0])
                new_U[:, s2] *= (1.0 + coeffs_int[1])
                new_V[:, s2] *= (1.0 + coeffs_int[1])
                new_W[:, s2] *= (1.0 + coeffs_int[1])
                mask = np.ones(rank, dtype=bool)
                mask[r] = False
                new_U = new_U[:, mask]
                new_V = new_V[:, mask]
                new_W = new_W[:, mask]
                if verify_full(new_U, new_V, new_W):
                    return new_U, new_V, new_W

        # Try triples
        if rank > 10:
            for s1, s2, s3 in combinations(others[:15], 3):
                vs1 = term_tensor(U, V, W, s1)
                vs2 = term_tensor(U, V, W, s2)
                vs3 = term_tensor(U, V, W, s3)
                A = np.column_stack([vs1, vs2, vs3])
                try:
                    coeffs, _, _, _ = np.linalg.lstsq(A, target, rcond=None)
                except Exception:
                    continue
                coeffs_int = np.round(coeffs * 2) / 2
                if np.all(np.abs(coeffs - coeffs_int) < 1e-9):
                    new_U = U.copy()
                    new_V = V.copy()
                    new_W = W.copy()
                    for idx, s in enumerate([s1, s2, s3]):
                        new_U[:, s] *= (1.0 + coeffs_int[idx])
                        new_V[:, s] *= (1.0 + coeffs_int[idx])
                        new_W[:, s] *= (1.0 + coeffs_int[idx])
                    mask = np.ones(rank, dtype=bool)
                    mask[r] = False
                    new_U = new_U[:, mask]
                    new_V = new_V[:, mask]
                    new_W = new_W[:, mask]
                    if verify_full(new_U, new_V, new_W):
                        return new_U, new_V, new_W

        return None

    def reduce_rank(U, V, W, rank, max_time):
        """Greedy rank reduction."""
        progress = True
        max_passes = 100
        pass_count = 0
        while progress and pass_count < max_passes and (time.time() - start_time) < max_time:
            progress = False
            pass_count += 1
            for r in range(U.shape[1]):
                if (time.time() - start_time) > max_time:
                    break
                res = try_eliminate_term(U, V, W, r, U.shape[1])
                if res is not None:
                    new_U, new_V, new_W = res
                    if verify_full(new_U, new_V, new_W):
                        U, V, W = new_U, new_V, new_W
                        rank = U.shape[1]
                        progress = True
                        break
        return U, V, W, rank

    def to_nice_list(arr):
        """Convert numpy array to list of integer/half-integer values."""
        result = []
        for row in arr:
            nice_row = []
            for val in row:
                val2 = val * 2
                if abs(val2 - round(val2)) < 1e-6:
                    nice_val = round(val2) / 2.0
                    if abs(nice_val - round(nice_val)) < 1e-9:
                        nice_val = int(round(nice_val))
                    nice_row.append(nice_val)
                else:
                    nice_row.append(val)
            result.append(nice_row)
        return result

    # Build schoolbook and reduce
    U, V, W, rank = build_schoolbook()

    if not verify_full(U, V, W):
        # Should not happen, but fallback
        return {"U": to_nice_list(U), "V": to_nice_list(V), "W": to_nice_list(W)}

    # Reduce rank
    U, V, W, rank = reduce_rank(U, V, W, rank, DEADLINE - 5.0)

    # Final verification
    if not verify_full(U, V, W):
        # Fallback to schoolbook
        U, V, W, rank = build_schoolbook()

    return {"U": to_nice_list(U), "V": to_nice_list(V), "W": to_nice_list(W)}
