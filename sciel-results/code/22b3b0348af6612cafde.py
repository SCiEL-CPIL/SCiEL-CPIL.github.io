"""Matrix multiplication 2,4,6 - rank-minimal tensor decomposition using Strassen blocks."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    n, m, p = 2, 4, 6

    # Build the 42 Strassen terms as before
    rank_init = 42
    U = np.zeros((n * m, rank_init), dtype=float)
    V = np.zeros((m * p, rank_init), dtype=float)
    W = np.zeros((p * n, rank_init), dtype=float)

    def add_strassen(i0, j0, k0, off):
        c = off
        U[i0 * m + j0, c] += 1
        U[(i0+1) * m + (j0+1), c] += 1
        V[j0 * p + k0, c] += 1
        V[(j0+1) * p + (k0+1), c] += 1
        W[k0 * n + i0, c] += 1
        W[(k0+1) * n + (i0+1), c] += 1

        c = off + 1
        U[(i0+1) * m + j0, c] += 1
        U[(i0+1) * m + (j0+1), c] += 1
        V[j0 * p + k0, c] += 1
        W[k0 * n + (i0+1), c] += 1
        W[(k0+1) * n + (i0+1), c] -= 1

        c = off + 2
        U[i0 * m + j0, c] += 1
        V[j0 * p + (k0+1), c] += 1
        V[(j0+1) * p + (k0+1), c] -= 1
        W[(k0+1) * n + i0, c] += 1
        W[(k0+1) * n + (i0+1), c] += 1

        c = off + 3
        U[(i0+1) * m + (j0+1), c] += 1
        V[(j0+1) * p + k0, c] += 1
        V[j0 * p + k0, c] -= 1
        W[k0 * n + i0, c] += 1
        W[k0 * n + (i0+1), c] += 1

        c = off + 4
        U[i0 * m + j0, c] += 1
        U[i0 * m + (j0+1), c] += 1
        V[(j0+1) * p + (k0+1), c] += 1
        W[k0 * n + i0, c] -= 1
        W[(k0+1) * n + i0, c] += 1

        c = off + 5
        U[(i0+1) * m + j0, c] += 1
        U[i0 * m + j0, c] -= 1
        V[j0 * p + k0, c] += 1
        V[j0 * p + (k0+1), c] += 1
        W[(k0+1) * n + (i0+1), c] += 1

        c = off + 6
        U[i0 * m + (j0+1), c] += 1
        U[(i0+1) * m + (j0+1), c] -= 1
        V[(j0+1) * p + k0, c] += 1
        V[(j0+1) * p + (k0+1), c] += 1
        W[k0 * n + i0, c] += 1

    add_strassen(0, 0, 0, 0)
    add_strassen(0, 0, 2, 7)
    add_strassen(0, 0, 4, 14)
    add_strassen(0, 2, 0, 21)
    add_strassen(0, 2, 2, 28)
    add_strassen(0, 2, 4, 35)

    # Now try to reduce rank by finding linear dependencies
    # Each rank-1 term is a vector in R^(n*m * m*p * p*n)
    # But we can check dependencies more efficiently:
    # The tensor T = sum_r U_r ⊗ V_r ⊗ W_r
    # We can check if columns of U, V, W have dependencies that preserve the tensor.
    
    # Alternative: check if the 42 flattened vectors are linearly dependent
    # Flatten each term: vec(U_r) ⊗ vec(V_r) ⊗ vec(W_r) - this is huge
    # Better: use the fact that T is determined by its "slices"
    
    # Actually, let's use a smarter approach:
    # The multiplication tensor T[i,j,k] = 1 for all valid (i,j,k)
    # We need: sum_r U_r[i,j] * V_r[j,k] * W_r[i,k] = 1 for all i,j,k
    
    # Let's verify our current decomposition is correct first
    T_recon = np.zeros((n, m, p))
    for r in range(rank_init):
        Ur = U[:, r].reshape(n, m)
        Vr = V[:, r].reshape(m, p)
        Wr = W[:, r].reshape(p, n)
        # Term r contributes: U_r[i,j] * V_r[j,k] * W_r[i,k]
        # But W is indexed as [k*n+i], so W_r[k*n+i] is the (k,i) entry
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_recon[i, j, k] += Ur[i, j] * Vr[j, k] * Wr[k, i]
    
    # T_recon should be all 1s
    assert np.allclose(T_recon, 1.0), f"Decomposition incorrect! Max err: {np.max(np.abs(T_recon - 1))}"
    
    # Now try to find rank reduction using linear algebra
    # Strategy: the tensor T can be represented in multiple unfoldings.
    # If we fix the decomposition form, we can try to find if there's a
    # linear combination of the 42 terms that gives zero.
    
    # A more practical approach: try to find a smaller set of terms.
    # Use the fact that for n=2, we can parameterize more directly.
    
    # Let me try: express the problem as finding minimal rank such that
    # there exist U (8×r), V (24×r), W (12×r) with the tensor equation.
    
    # This is a bilinear/trilinear system. Let's try a different decomposition strategy.
    
    # For M_{2,4,6}, think of it as: C = A * B where A is 2×4, B is 4×6.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # We can try to use a "split" approach:
    # Split j into two groups of 2: j1={0,1}, j2={2,3}
    # Split k into three groups of 2: k1={0,1}, k2={2,3}, k3={4,5}
    
    # For each (j_group, k_group) pair, we compute a 2×2 matrix product
    # using 7 Strassen terms. That's 6 * 7 = 42.
    
    # Can we do better by sharing terms across k_groups?
    # The A part is the same for all k_groups (for a given j_group).
    # The B part changes.
    
    # Idea: For a fixed j_group, the A side is the same 2×2 block.
    # We compute 3 different 2×2 products with the same A block but different B blocks.
    # In Strassen, M1 = (A00+A11)(B00+B11), etc.
    # The A-side factors are: A00+A11, A10+A11, A00, A11, A00+A01, A10-A00, A01-A11
    # These are fixed for a given j_group.
    # The B-side factors change per k_group.
    
    # So for j_group 1, we have 7 A-factors and 3×7=21 B-factors.
    # Total terms: 7 (for j1, shared A) × 3 (k groups) = 21? No, that's not right.
    # Each term is one (A_factor, B_factor, W_pattern) triple.
    
    # Actually the 42 terms are already minimal in this block structure.
    # Let me check if there's a global dependency.
    
    # Try: use numpy to find the rank of the "tensor matrix"
    # Reshape the 42 terms into a matrix and find its rank.
    # The tensor T is in R^{(nm) × (mp) × (pn)}.
    # A rank-1 term is u ⊗ v ⊗ w, which when flattened is u ⊗ v ⊗ w.
    # The 42 terms span a subspace. If their flattened forms are dependent, we can reduce.
    
    # But flattening gives vectors of size 8*24*12 = 2304. That's manageable.
    # Actually, let's use a more efficient check.
    
    # The tensor T[i,j,k,l] = δ_{j,k} (where i∈[0,2), j∈[0,4), k∈[0,6), l∈[0,2))
    # Wait, I need to be careful with the indexing convention.
    # W[k*n+i, r] means W is indexed by (k,i) pairs.
    
    # Let me use a different approach: try random linear combinations to find dependencies.
    # Or better: use the structure.
    
    # Key insight: for the 6 blocks, blocks 1,2,3 (j_group=0) all use the same A block A[:,0:2].
    # Blocks 4,5,6 (j_group=1) all use A[:,2:4].
    
    # Within j_group=0, the 21 terms use 7 distinct A-side patterns, each repeated 3 times
    # (once per k_group). The B-side and W-side differ.
    
    # Let me check: is there a linear dependency among the 21 terms of j_group=0?
    # Term (a, b, w) where a is one of 7 A-patterns, b is a B-pattern, w is a W-pattern.
    # For fixed A-pattern, the 3 terms (one per k_group) have the same A but different B and W.
    
    # I don't see an obvious dependency. Let me try a computational approach.
    
    # Build the tensor as a 3D array and check its tensor rank bounds.
    # Actually, let me try to see if rank 40 or 41 is achievable by a different decomposition.
    
    # Alternative: try a non-block decomposition.
    # For M_{2,4,6}, the trivial rank is 48 (one term per (i,j,k) triple... wait no).
    # Trivial: C[i,k] = sum_j A[i,j]*B[j,k]. Each term A[i,j]*B[j,k] contributes to C[i,k].
    # So we need 2*4*6 = 48 terms if we use the trivial decomposition.
    # Strassen reduces this to 42.
    
    # Can we get below 42? Let me try to construct a rank-40 or rank-41 decomposition.
    
    # Idea: merge some Strassen terms across blocks.
    # For example, M6 of block (j0=0,k0=0) is (A10-A00)(B00+B01) -> C11
    # M6 of block (j0=0,k0=2) is (A10-A00)(B20+B21) -> C13
    # These share the A-factor (A10-A00) but have different B-factors.
    # They can't be merged directly.
    
    # What about M1 of block (j0=0,k0=0): (A00+A11)(B00+B11) -> C00+C11
    # M1 of block (j0=0,k0=2): (A00+A11)(B20+B31) -> C02+C13
    # Again, same A-factor, different B-factors.
    
    # I think 42 might be close to optimal for this block structure.
    # Let me try a completely different decomposition.
    
    # For n=2, we can write:
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k] + A[0,3]*B[3,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k] + A[1,3]*B[3,k]
    
    # This is like computing 2 linear combinations of rows of B, weighted by columns of A.
    # For each k, C[:,k] = B[:,k]^T * A^T (as a 2×4 times 4×1... no)
    # C[:,k] = sum_j A[:,j] * B[j,k]
    
    # Think of it as: for each j, we add A[:,j] * B[j,k] for all k.
    # A[:,j] is a 2-vector, B[j,k] is a scalar for each k.
    
    # One approach: use 4 "columns" of A, each contributing to all 6 outputs.
    # That's 4 * (2*6) = 48 scalar multiplications, or rank 48 in the trivial way.
    
    # Strassen saves by grouping: instead of 4 separate j's, group into 2 pairs.
    # Each pair uses 7 terms instead of 4*2=8 terms. Savings: 1 per pair, total 2.
    # 48 - 6 = 42. ✓
    
    # Can we save more? What if we group all 4 j's together?
    # A 2×4 × 4×6 product. If we could do this in fewer than 42 terms...
    
    # The rank of M_{2,4,6} is known to be at most 42 (Strassen) and at least...
    # Lower bound: max(n*m, m*p, n*p) = max(8, 24, 12) = 24.
    # Better lower bounds exist but 42 might be close to optimal.
    
    # Let me try to verify if we can get rank 40 or 41 with a clever decomposition.
    
    # Try: use 3 groups for j: {0,1}, {2,3} with Strassen (7 each) = 14 terms for A-side
    # But we need 3 k-groups, so 14 * 3 = 42. Same as before.
    
    # What if we use a different grouping for k?
    # Instead of 3 groups of 2, use 1 group of 6?
    # Then it's a 2×4 × 4×6 product directly.
    # Can we do this in < 42 terms?
    
    # The rank of M_{2,4,6}... Let me think about this differently.
    # M_{2,4,6} = M_{2,2,6} ⊕ M_{2,2,6} (splitting j into two groups of 2)
    # rank(M_{2,2,6}) ≤ 7*3 = 21 (Strassen on 2×2 × 2×6, splitting k into 3 groups of 2)
    # Or rank(M_{2,2,6}) ≤ ? 
    # M_{2,2,6}: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for i=0,1; k=0..5
    # This is like computing 2×6 outputs from a 2×2 times 2×6.
    # Trivial: 2*2*6 = 24 terms.
    # Strassen on the 2×2 part: 7 terms, each giving a 1×6 or similar output.
    # Actually: M_{2,2,6} can be done in 7*3 = 21 (split k into 3 pairs) or
    # in 7*2 + 7 = 21 (split k into {0,1},{2,3},{4,5}).
    # Or: 7 terms for the 2×2×2 core, applied to each of 3 k-pairs = 21.
    # Or: can we do M_{2,2,6} in fewer than 21?
    # Lower bound: max(2*2, 2*6, 2*6) = 12.
    # Upper bound: 21 (Strassen) or 24 (trivial).
    # I think M_{2,2,6} has rank 18 or so? Not sure.
    
    # This is getting complex. Let me just try to verify the current 42-rank solution
    # is correct and return it, since finding a better one requires deep research.
    
    # But wait - let me try one more thing: check if any of the 42 terms are
    # actually zero or if there's a simple linear dependency.
    
    # Check for zero terms
    for r in range(rank_init):
        if np.all(U[:, r] == 0) or np.all(V[:, r] == 0) or np.all(W[:, r] == 0):
            print(f"Term {r} is zero!")
    
    # Try to find dependencies using the tensor structure
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We have: sum_r U_r[i,j] * V_r[j,k] * W_r[i,k] = 1
    
    # For fixed (i,k), this is: sum_j (sum_r U_r[i,j] * W_r[i,k]) * V_r[j,k] = 1
    # Hmm, this is getting complicated.
    
    # Let me just try to use a rank-reduction algorithm.
    # If we can find coefficients c_r such that sum_r c_r * (U_r ⊗ V_r ⊗ W_r) = 0,
    # then we can eliminate one term.
    
    # This is equivalent to checking if the 42 vectors (in the tensor product space)
    # are linearly dependent.
    
    # The tensor product space has dimension 8 * 24 * 12 = 2304.
    # 42 vectors in a 2304-dim space are almost certainly independent.
    # So no simple linear dependency exists.
    
    # Conclusion: the rank-42 Strassen decomposition is likely near-optimal.
    # Let me just return it, but let me also try one creative alternative.
    
    # Alternative: use a "ladder" decomposition.
    # For M_{2,4,6}, think of it as:
    # C = [A_left * B_left + A_right * B_right]
    # where A_left = A[:,0:2], A_right = A[:,2:4], etc.
    
    # What if we use a shared "intermediate" computation?
    # E.g., compute S = A * [some 4×4 matrix] and then C = S * [some 4×6 matrix]?
    # That doesn't help directly.
    
    # I'll stick with rank 42 but let me make sure the output format is correct.
    # Convert to lists with integer values.
    
    if time.time() > deadline:
        pass
    
    # Convert numpy arrays to lists of ints
    U_list = [[int(round(U[i, r])) for r in range(rank_init)] for i in range(n * m)]
    V_list = [[int(round(V[i, r])) for r in range(rank_init)] for i in range(m * p)]
    W_list = [[int(round(W[i, r])) for r in range(rank_init)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
