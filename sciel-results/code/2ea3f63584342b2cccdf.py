import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115

    def get_strassen_2x2x2():
        rank = 7
        U = np.zeros((4, rank))
        V = np.zeros((4, rank))
        W = np.zeros((4, rank))
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        U[3, 3] = 1
        V[1, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[2, 5] = 1
        W[3, 5] = 1
        U[1, 6] = 1; U[3, 6] = -1
        V[1, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        return U, V, W

    def build_strassen_decomposition():
        U_s, V_s, W_s = get_strassen_2x2x2()
        rank = 42
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        def add_block(a_idx, b_idx, w_idx, offset):
            for r in range(7):
                U[a_idx, offset + r] += U_s[:, r]
                V[b_idx, offset + r] += V_s[:, r]
                W[w_idx, offset + r] += W_s[:, r]

        a1_idx = [0, 1, 2, 3]
        a2_idx = [4, 5, 6, 7]
        b1_idx = [0, 1, 6, 7]
        b2_idx = [2, 3, 8, 9]
        b3_idx = [4, 5, 10, 11]
        b4_idx = [12, 13, 18, 19]
        b5_idx = [14, 15, 20, 21]
        b6_idx = [16, 17, 22, 23]
        c1_idx = [0, 1, 2, 3]
        c2_idx = [4, 5, 6, 7]
        c3_idx = [8, 9, 10, 11]

        add_block(a1_idx, b1_idx, c1_idx, 0)
        add_block(a2_idx, b4_idx, c1_idx, 7)
        add_block(a1_idx, b2_idx, c2_idx, 14)
        add_block(a2_idx, b5_idx, c2_idx, 21)
        add_block(a1_idx, b3_idx, c3_idx, 28)
        add_block(a2_idx, b6_idx, c3_idx, 35)
        return U, V, W

    def verify_decomposition(U, V, W, n, m, p, trials=3):
        rank = U.shape[1]
        U_reshaped = U.reshape(n, m, rank)
        V_reshaped = V.reshape(m, p, rank)
        W_reshaped = W.reshape(p, n, rank)
        T_recon = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        T_true = np.ones((n, m, p))
        return np.allclose(T_recon, T_true, atol=1e-8)

    def trivial_decomposition():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * (m * p) + j * p + k
                    U[i * m + j, r] = 1
                    V[j * p + k, r] = 1
                    W[k * n + i, r] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def reduce_rank(U, V, W, n, m, p):
        """
        Greedy rank reduction:
        1. Build the 48-dim tensor T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        2. Compute Gram matrix G = T_flat^T @ T_flat
        3. Find null space via SVD
        4. Greedily remove terms that are in the span of remaining terms
        """
        rank = U.shape[1]
        U_flat = U.reshape(n * m * rank)  # not quite right, let's do it properly

        # Build the full tensor T of shape (n*m, m*p, p*n)
        # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
        T = np.einsum('ar,br,cr->abc', U, V, W)
        T_flat = T.reshape(-1)
        dim = T_flat.shape[0]  # 48 * 24 * 12 = 13824... wait no
        # n*m = 8, m*p = 24, p*n = 12
        # T shape: (8, 24, 12) -> 2304 elements

        # Build Gram matrix: G[r1, r2] = sum_{a,b,c} T[a,b,c]_r1 * T[a,b,c]_r2
        # where T_r[a,b,c] = U[a,r]*V[b,r]*W[c,r]
        # G = sum_{a,b,c} (U[:,r1]*V[:,r1]*W[:,r1])^T * (U[:,r2]*V[:,r2]*W[:,r2])
        # = sum_{a,b,c} U[a,r1]*V[b,r1]*W[c,r1] * U[a,r2]*V[b,r2]*W[c,r2]
        # = (sum_a U[a,r1]*U[a,r2]) * (sum_b V[b,r1]*V[b,r2]) * (sum_c W[c,r1]*W[c,r2])
        # = (U^T U)[r1,r2] * (V^T V)[r1,r2] * (W^T W)[r1,r2]

        G = (U.T @ U) * (V.T @ V) * (W.T @ W)
        # G is (rank, rank)

        # SVD of G
        try:
            U_svd, s_vals, Vt_svd = np.linalg.svd(G, full_matrices=False)
        except np.linalg.LinAlgError:
            return U, V, W

        # Null space: vectors in Vt_svd corresponding to small singular values
        tol = 1e-10 * max(s_vals) if len(s_vals) > 0 else 1e-10
        null_mask = s_vals < tol
        null_dim = np.sum(null_mask)

        if null_dim == 0:
            return U, V, W

        # Null space vectors
        null_vecs = Vt_svd[null_mask, :].T  # (rank, null_dim)

        # Greedy removal: try to remove each term if it's in the span of the rest
        # A term r can be removed if e_r is in the span of {e_s : s != r}
        # This is equivalent to: there exists a vector x such that x_r = 1 and x is in null space
        # i.e., x is in null space and x_r != 0

        # For each null space vector v, we can remove the term with largest |v[r]|
        # and express it as a combination of the others

        kept = list(range(rank))
        removed = set()

        # Sort null vectors by max component
        null_vecs_sorted = null_vecs[np.argsort(-np.max(np.abs(null_vecs), axis=1))]

        for nv in null_vecs_sorted:
            if time.time() > deadline - 20:
                break
            # Find the term with largest absolute value in this null vector
            max_idx = np.argmax(np.abs(nv))
            if max_idx in removed:
                continue
            if abs(nv[max_idx]) < 1e-12:
                continue

            # Check if this term is still in 'kept'
            if max_idx not in kept:
                continue

            # Verify: after removing term max_idx, can we adjust the others to compensate?
            # The null vector tells us: sum_r nv[r] * term_r = 0
            # So term_{max_idx} = -sum_{r != max_idx} (nv[r]/nv[max_idx]) * term_r
            # We need to update U, V, W for all remaining terms

            # First, check that the null vector is valid:
            # sum_r nv[r] * U[:,r] * V[:,r] * W[:,r] = 0 (as tensor)
            # This should be true by construction since G @ nv = 0
            # But let's verify numerically for safety
            check = np.einsum('ar,br,cr,r->abc', U, V, W, nv)
            if np.max(np.abs(check)) > 1e-8:
                continue

            # Remove term max_idx and update others
            new_kept = [k for k in kept if k != max_idx]
            if len(new_kept) < 1:
                continue

            # Build new U, V, W
            new_rank = len(new_kept)
            U_new = np.zeros((n * m, new_rank))
            V_new = np.zeros((m * p, new_rank))
            W_new = np.zeros((p * n, new_rank))

            for new_i, old_i in enumerate(new_kept):
                U_new[:, new_i] = U[:, old_i]
                V_new[:, new_i] = V[:, old_i]
                W_new[:, new_i] = W[:, old_i]

            # Apply the correction: for each remaining term, add (nv[old_i]/nv[max_idx]) * term_{max_idx}
            # Wait, the relation is: nv[max_idx]*term_{max_idx} + sum_{r!=max_idx} nv[r]*term_r = 0
            # So term_{max_idx} = -sum_{r!=max_idx} (nv[r]/nv[max_idx]) * term_r
            # We want to eliminate term_{max_idx}. The decomposition is:
            # sum_r U[:,r] V[:,r] W[:,r] = sum_{r in new_kept} U[:,r] V[:,r] W[:,r] + U[:,max_idx] V[:,max_idx] W[:,max_idx]
            # = sum_{r in new_kept} U[:,r] V[:,r] W[:,r] - sum_{r in new_kept} (nv[r]/nv[max_idx]) * U[:,max_idx] V[:,max_idx] W[:,max_idx]
            #
            # But this doesn't directly give us a sum of rank-1 terms in the new basis.
            # The issue is that U[:,max_idx] V[:,max_idx] W[:,max_idx] is a rank-1 tensor,
            # and multiplying by a scalar keeps it rank-1, but adding it to the sum doesn't
            # simply update the existing rank-1 terms unless we can absorb it.
            #
            # Actually, the correct approach: the null vector tells us a linear dependency
            # among the 48-dim vectors t_r = U[:,r] ⊗ V[:,r] ⊗ W[:,r].
            # If we have sum_r nv[r] t_r = 0, then we can drop one term.
            # But the remaining terms are still the same rank-1 tensors, just fewer of them.
            # The sum of the remaining (with adjusted coefficients) still equals T.
            #
            # Wait, no. The decomposition is T = sum_r t_r where t_r = U[:,r]⊗V[:,r]⊗W[:,r].
            # The null vector says sum_r nv[r] t_r = 0.
            # So T = sum_r t_r = sum_{r != max_idx} t_r + t_{max_idx}
            # And t_{max_idx} = -sum_{r != max_idx} (nv[r]/nv[max_idx]) t_r
            # So T = sum_{r != max_idx} t_r - sum_{r != max_idx} (nv[r]/nv[max_idx]) t_r
            # = sum_{r != max_idx} (1 - nv[r]/nv[max_idx]) t_r
            #
            # But this changes the coefficients of the existing terms, which is fine
            # as long as the result is still a sum of rank-1 tensors.
            # However, (1 - nv[r]/nv[max_idx]) * t_r is still a rank-1 tensor!
            # So we just scale each remaining term.

            for new_i, old_i in enumerate(new_kept):
                scale = 1.0 - nv[old_i] / nv[max_idx]
                U_new[:, new_i] *= scale
                # V and W don't need scaling since only U is scaled
                # Wait, that's wrong. We need to scale the whole rank-1 term.
                # t_r = U[:,r] ⊗ V[:,r] ⊗ W[:,r]
                # scale * t_r = (scale * U[:,r]) ⊗ V[:,r] ⊗ W[:,r]
                # So we can just scale U.

            # Verify the new decomposition
            if verify_decomposition(U_new, V_new, W_new, n, m, p, trials=1):
                U, V, W = U_new, V_new, W_new
                kept = new_kept
                removed.add(max_idx)
            else:
                # Rejection: the scaling approach doesn't work as expected
                # because scaling only one factor doesn't preserve the tensor structure
                # Let's try a different approach: actually redistribute
                pass

        return U, V, W

    def reduce_rank_v2(U, V, W, n, m, p):
        """
        Better rank reduction: use the null space to actually eliminate terms.
        The key insight: if sum_r nv[r] * (U[:,r] ⊗ V[:,r] ⊗ W[:,r]) = 0,
        we can't simply drop a term and rescale others because the tensor product
        structure doesn't allow that.

        Instead, we look for pairs of terms that can be merged.
        Two rank-1 terms t_a = U_a ⊗ V_a ⊗ W_a and t_b = U_b ⊗ V_b ⊗ W_b
        can be merged if one of the factor vectors is a scalar multiple of the other
        in all three dimensions, which would make them proportional.

        More generally, we look for linear dependencies in the U, V, or W matrices.
        If U[:, r1] and U[:, r2] are linearly dependent (one is a multiple of the other),
        then we can potentially merge those two terms.
        """
        rank = U.shape[1]

        # Check for linear dependencies in U
        # If U[:, r1] = alpha * U[:, r2], then:
        # U[:,r1] ⊗ V[:,r1] ⊗ W[:,r1] + U[:,r2] ⊗ V[:,r2] ⊗ W[:,r2]
        # = U[:,r2] ⊗ (alpha * V[:,r1] + V[:,r2]) ⊗ W[:,r1]  -- NO, this doesn't work
        # because W factors are different.

        # Actually, the correct merge condition is more subtle.
        # Let's try a different approach: use the null space of the combined system.

        # Build the 3-way tensor and find its rank via iterative method
        # For small cases, we can use the "tensor rank" algorithm

        # Alternative: try all pairs and see if we can eliminate one
        # This is O(rank^2) which is 42^2 = 1764, feasible

        for r1 in range(rank):
            if time.time() > deadline - 20:
                break
            for r2 in range(r1 + 1, rank):
                if time.time() > deadline - 15:
                    break

                # Try to see if term r2 can be expressed using term r1 and others
                # This is expensive; skip for now and use a simpler heuristic

                # Check if U[:,r1] and U[:,r2] are proportional
                u1 = U[:, r1]
                u2 = U[:, r2]
                if np.max(np.abs(u1)) > 1e-12 and np.max(np.abs(u2)) > 1e-12:
                    ratio_u = np.dot(u1, u2) / (np.dot(u2, u2) + 1e-30)
                    if np.allclose(u1, ratio_u * u2, atol=1e-10):
                        # U factors are proportional
                        # Check V
                        v1 = V[:, r1]
                        v2 = V[:, r2]
                        if np.max(np.abs(v1)) > 1e-12 and np.max(np.abs(v2)) > 1e-12:
                            ratio_v = np.dot(v1, v2) / (np.dot(v2, v2) + 1e-30)
                            if np.allclose(v1, ratio_v * v2, atol=1e-10):
                                # V factors are also proportional
                                # Check W
                                w1 = W[:, r1]
                                w2 = W[:, r2]
                                if np.max(np.abs(w1)) > 1e-12 and np.max(np.abs(w2)) > 1e-12:
                                    ratio_w = np.dot(w1, w2) / (np.dot(w2, w2) + 1e-30)
                                    if np.allclose(w1, ratio_w * w2, atol=1e-10):
                                        # All three factors are proportional!
                                        # These two terms are proportional and can be merged
                                        # t_r1 + t_r2 = (1 + ratio_u*ratio_v*ratio_w) * t_r2
                                        # (if ratios are consistent)
                                        pass

        return U, V, W

    def reduce_rank_greedy(U, V, W, n, m, p):
        """
        Greedy rank reduction using null space of the Gram matrix.
        Key fix: when we find a null vector, we use it to eliminate a term
        by updating ALL three factor matrices appropriately.

        If sum_r nv[r] * U[:,r] ⊗ V[:,r] ⊗ W[:,r] = 0,
        then for any term r0 with nv[r0] != 0:
        U[:,r0] ⊗ V[:,r0] ⊗ W[:,r0] = -sum_{r != r0} (nv[r]/nv[r0]) * U[:,r] ⊗ V[:,r] ⊗ W[:,r]

        This means we can drop term r0, but the remaining terms need to be
        updated. However, the RHS is a sum of rank-1 tensors, so the total
        is still a sum of rank-1 tensors (just with different coefficients).

        The trick: we can't simply rescale existing terms because the
        tensor product structure couples all three factors. Instead, we
        need to find a new set of rank-1 tensors that sum to the same thing.

        Practical approach: use the null vector to create a new decomposition
        by "rotating" the basis of rank-1 terms.
        """
        rank = U.shape[1]
        if rank <= 1:
            return U, V, W

        # Build Gram matrix
        G = (U.T @ U) * (V.T @ V) * (W.T @ W)

        # SVD
        try:
            U_svd, s_vals, Vt_svd = np.linalg.svd(G, full_matrices=False)
        except np.linalg.LinAlgError:
            return U, V, W

        tol = 1e-9 * (s_vals[0] if len(s_vals) > 0 else 1.0)
        null_mask = s_vals < tol
        null_dim = np.sum(null_mask)

        if null_dim == 0:
            return U, V, W

        # Get null space vectors
        null_vecs = Vt_svd[null_mask, :].T  # (rank, null_dim)

        # For each null vector, try to eliminate the term with largest component
        # Use a basis transformation approach:
        # If nv is a null vector, then for any invertible matrix M,
        # the transformed vectors U' = U @ M, V' = V @ M^{-1}, W' = W @ I
        # ... this doesn't quite work either.

        # Simpler approach: just try to drop terms one at a time and verify
        best_U, best_V, best_W = U, V, W
        best_rank = rank

        for nv in null_vecs:
            if time.time() > deadline - 15:
                break

            # Try eliminating each term
            for r0 in np.argsort(-np.abs(nv)):
                if time.time() > deadline - 12:
                    break
                if abs(nv[r0]) < 1e-12:
                    continue

                # Create new decomposition without term r0
                # T = sum_r U[:,r] V[:,r] W[:,r]
                # = sum_{r != r0} U[:,r] V[:,r] W[:,r] + U[:,r0] V[:,r0] W[:,r0]
                # Using null vector: U[:,r0] V[:,r0] W[:,r0] = -sum_{r != r0} (nv[r]/nv[r0]) U[:,r] V[:,r] W[:,r]
                # So T = sum_{r != r0} U[:,r] V[:,r] W[:,r] - sum_{r != r0} (nv[r]/nv[r0]) U[:,r] V[:,r] W[:,r]
                # = sum_{r != r0} (1 - nv[r]/nv[r0]) * U[:,r] V[:,r] W[:,r]

                # This works! We just scale each remaining term.
                new_rank = rank - 1
                U_new = np.zeros((n * m, new_rank))
                V_new = np.zeros((m * p, new_rank))
                W_new = np.zeros((p * n, new_rank))

                idx = 0
                for r in range(rank):
                    if r == r0:
                        continue
                    scale = 1.0 - nv[r] / nv[r0]
                    U_new[:, idx] = scale * U[:, r]
                    V_new[:, idx] = V[:, r]
                    W_new[:, idx] = W[:, r]
                    idx += 1

                # Verify
                if verify_decomposition(U_new, V_new, W_new, n, m, p, trials=1):
                    best_U, best_V, best_W = U_new, V_new, W_new
                    best_rank = new_rank
                    # Continue trying to reduce further
                    U, V, W = U_new, V_new, W_new
                    rank = new_rank
                    # Recompute Gram matrix for next iteration
                    G = (U.T @ U) * (V.T @ V) * (W.T @ W)
                    try:
                        U_svd, s_vals, Vt_svd = np.linalg.svd(G, full_matrices=False)
                    except np.linalg.LinAlgError:
                        break
                    tol = 1e-9 * (s_vals[0] if len(s_vals) > 0 else 1.0)
                    null_mask = s_vals < tol
                    if np.sum(null_mask) == 0:
                        break
                    null_vecs = Vt_svd[null_mask, :].T
                    break  # Restart with new null vectors

        return best_U, best_V, best_W

    try:
        U, V, W = build_strassen_decomposition()

        if not verify_decomposition(U, V, W, n, m, p, trials=2):
            return trivial_decomposition()

        # Attempt rank reduction
        if time.time() < deadline - 30:
            U, V, W = reduce_rank_greedy(U, V, W, n, m, p)

        # Final verification
        if verify_decomposition(U, V, W, n, m, p, trials=3):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            # Fallback to original Strassen
            U, V, W = build_strassen_decomposition()
            if verify_decomposition(U, V, W, n, m, p, trials=2):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            return trivial_decomposition()

    except Exception:
        return trivial_decomposition()

# EVOLVE-BLOCK-END
