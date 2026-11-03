import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    # The target tensor: T[i,j,k] = 1 for all (i,j,k) in {0..n-1}x{0..m-1}x{0..p-1}
    # Because matrix multiplication C[i,k] = sum_j A[i,j] * B[j,k]
    # means the tensor has T[i,j,k] = 1 for every valid (i,j,k).
    total_terms = n * m * p  # 75

    # Build the target tensor as a flat vector
    target = np.ones(total_terms, dtype=np.int64)

    # Generate candidate rank-1 terms
    # Each candidate is a triple (u, v, w) where u is in R^{n*m}, v in R^{m*p}, w in R^{p*n}
    # and the outer product u * v * w (reshaped) contributes to the tensor.

    candidates = []  # list of (term_vector, is_schoolbook)

    # Schoolbook terms: each (i,j,k) gives a rank-1 term
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = np.zeros(n * m, dtype=np.int64)
                v = np.zeros(m * p, dtype=np.int64)
                w = np.zeros(p * n, dtype=np.int64)
                u[i * m + j] = 1
                v[j * p + k] = 1
                w[k * n + i] = 1
                term = (u * v * w).reshape(-1)
                # Reshape: term index is (i*m+j) * (m*p) * (p*n) ... no, that's wrong.
                # Actually the tensor T has shape (n, m, p), flattened as T[i, j, k] = T[i*m*p + j*p + k]
                # But our u is indexed by i*m+j, v by j*p+k, w by k*n+i.
                # The product u[i*m+j] * v[j*p+k] * w[k*n+i] only non-zero when
                # i*m+j matches, j*p+k matches, k*n+i matches.
                # For schoolbook: u has 1 at i*m+j, v has 1 at j*p+k, w has 1 at k*n+i.
                # So the product is 1 only at the single index (i*m+j, j*p+k, k*n+i) in the
                # tensor product space. But we need to map this back to T[i,j,k].
                #
                # The tensor T lives in R^{n*m} ⊗ R^{m*p} ⊗ R^{p*n} but we identify
                # T[i,j,k] with the component where the first index is i*m+j, second is j*p+k,
                # third is k*n+i. So the flattened tensor has size (n*m)*(m*p)*(p*n) which is huge.
                #
                # Wait, I need to reconsider. The tensor for matrix multiplication is:
                # T ∈ R^{n×m} ⊗ R^{m×p} ⊗ R^{p×n}
                # T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]B[j,k]
                # So T has exactly one non-zero entry per (i,j,k) triple, and T[i,j,k] = 1.
                #
                # A rank-1 decomposition: T = sum_c u_c ⊗ v_c ⊗ w_c
                # where u_c ∈ R^{n×m}, v_c ∈ R^{m×p}, w_c ∈ R^{p×n}
                # and T[i,j,k] = sum_c u_c[i,j] * v_c[j,k] * w_c[k,i]
                #
                # So for each candidate, I need to compute the contribution to each (i,j,k).
                # Let me just compute it directly.

                pass

    # Let me redo this more carefully.
    # For a candidate (u, v, w), its contribution to T[i,j,k] is u[i,j]*v[j,k]*w[k,i]
    # where u is reshaped to (n,m), v to (m,p), w to (p,n).

    def compute_contribution(u, v, w):
        """Compute the tensor contribution of a rank-1 term."""
        u2d = u.reshape(n, m)
        v2d = v.reshape(m, p)
        w2d = w.reshape(p, n)
        # T[i,j,k] += u2d[i,j] * v2d[j,k] * w2d[k,i]
        # Vectorize: for each i, j, k
        contrib = np.zeros(total_terms, dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    contrib[i * m * p + j * p + k] = u2d[i, j] * v2d[j, k] * w2d[k, i]
        return contrib

    # Schoolbook candidates
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = np.zeros(n * m, dtype=np.int64)
                v = np.zeros(m * p, dtype=np.int64)
                w = np.zeros(p * n, dtype=np.int64)
                u[i * m + j] = 1
                v[j * p + k] = 1
                w[k * n + i] = 1
                contrib = compute_contribution(u, v, w)
                candidates.append((contrib, u, v, w, True))

    # Strassen-like candidates for all 2x2x2 sub-blocks
    # For a block with i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}
    # We have 7 Strassen terms. But we can also consider all linear combinations.
    # Actually, let's just use the standard 7 Strassen terms for each block.

    def strassen_terms(i0, j0, k0):
        """Generate the 7 Strassen rank-1 terms for a 2x2x2 block."""
        terms = []
        i1, j1, k1 = i0 + 1, j0 + 1, k0 + 1

        # M1 = (A[i0,j0]+A[i1,j1])(B[j0,k0]+B[j1,k1])
        u = np.zeros(n * m, dtype=np.int64); v = np.zeros(m * p, dtype=np.int64); w = np.zeros(p * n, dtype=np.int64)
        u[i0*m+j0] = 1; u[i1*m+j1] = 1
        v[j0*p+k0] = 1; v[j1*p+k1] = 1
        w[k0*n+i0] = 1; w[k1*n+i1] = 1
        terms.append((u, v, w))

        # M2 = (A[i1,j0]+A[i1,j1])B[j0,k0]
        u = np.zeros(n * m, dtype=np.int64); v = np.zeros(m * p, dtype=np.int64); w = np.zeros(p * n, dtype=np.int64)
        u[i1*m+j0] = 1; u[i1*m+j1] = 1
        v[j0*p+k0] = 1
        w[k0*n+i1] = 1; w[k1*n+i1] = -1
        terms.append((u, v, w))

        # M3 = A[i0,j0](B[j0,k1]-B[j1,k1])
        u = np.zeros(n * m, dtype=np.int64); v = np.zeros(m * p, dtype=np.int64); w = np.zeros(p * n, dtype=np.int64)
        u[i0*m+j0] = 1
        v[j0*p+k1] = 1; v[j1*p+k1] = -1
        w[k1*n+i0] = 1; w[k1*n+i1] = 1
        terms.append((u, v, w))

        # M4 = A[i1,j1](B[j1,k0]-B[j0,k0])
        u = np.zeros(n * m, dtype=np.int64); v = np.zeros(m * p, dtype=np.int64); w = np.zeros(p * n, dtype=np.int64)
        u[i1*m+j1] = 1
        v[j1*p+k0] = 1; v[j0*p+k0] = -1
        w[k0*n+i0] = 1; w[k0*n+i1] = 1
        terms.append((u, v, w))

        # M5 = (A[i0,j0]+A[i0,j1])B[j1,k1]
        u = np.zeros(n * m, dtype=np.int64); v = np.zeros(m * p, dtype=np.int64); w = np.zeros(p * n, dtype=np.int64)
        u[i0*m+j0] = 1; u[i0*m+j1] = 1
        v[j1*p+k1] = 1
        w[k0*n+i0] = -1; w[k1*n+i0] = 1
        terms.append((u, v, w))

        # M6 = (A[i1,j0]-A[i0,j0])(B[j0,k0]+B[j0,k1])
        u = np.zeros(n * m, dtype=np.int64); v = np.zeros(m * p, dtype=np.int64); w = np.zeros(p * n, dtype=np.int64)
        u[i1*m+j0] = 1; u[i0*m+j0] = -1
        v[j0*p+k0] = 1; v[j0*p+k1] = 1
        w[k1*n+i1] = 1
        terms.append((u, v, w))

        # M7 = (A[i0,j1]-A[i1,j1])(B[j1,k0]+B[j1,k1])
        u = np.zeros(n * m, dtype=np.int64); v = np.zeros(m * p, dtype=np.int64); w = np.zeros(p * n, dtype=np.int64)
        u[i0*m+j1] = 1; u[i1*m+j1] = -1
        v[j1*p+k0] = 1; v[j1*p+k1] = 1
        w[k0*n+i0] = 1
        terms.append((u, v, w))

        return terms

    # Add Strassen candidates for all possible 2x2x2 blocks
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                for (u, v, w) in strassen_terms(i0, j0, k0):
                    contrib = compute_contribution(u, v, w)
                    candidates.append((contrib, u, v, w, False))

    # Now use greedy set cover to find minimum number of candidates that sum to target
    # Greedy: at each step, pick the candidate that covers the most uncovered tensor entries
    # (where "covers" means it has a non-zero contribution to entries that are still 0 in our partial sum)

    current_sum = np.zeros(total_terms, dtype=np.int64)
    selected = []
    remaining = target.copy()

    # Precompute contributions as numpy arrays for speed
    contribs = np.array([c[0] for c in candidates], dtype=np.int64)

    num_candidates = len(candidates)

    while np.any(remaining != 0) and time.time() < deadline:
        # Find candidate that maximizes dot product with remaining (i.e., covers most uncovered)
        # Since entries are ±1, we want candidates that help fill in the remaining
        scores = np.abs(contribs @ np.abs(remaining))
        best_idx = np.argmax(scores)
        if scores[best_idx] == 0:
            break
        current_sum += contribs[best_idx]
        selected.append(best_idx)
        remaining = target - current_sum

        # Check if we're done
        if np.all(remaining == 0):
            break

        # Remove selected candidate to avoid re-selection
        contribs[best_idx] = 0

    if np.any(remaining != 0):
        # Fallback to schoolbook if greedy didn't work
        print("Greedy failed, falling back to schoolbook")
        selected = list(range(total_terms))  # schoolbook indices are 0..74
        current_sum = np.zeros(total_terms, dtype=np.int64)
        for idx in selected:
            current_sum += candidates[idx][0]

    rank = len(selected)

    # Build U, V, W matrices
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    for c, idx in enumerate(selected):
        contrib, u, v, w, is_schoolbook = candidates[idx]
        U[:, c] = u.astype(np.float64)
        V[:, c] = v.astype(np.float64)
        W[:, c] = w.astype(np.float64)

    # Verify
    T_reconstructed = np.zeros(total_terms, dtype=np.float64)
    for c in range(rank):
        u2d = U[:, c].reshape(n, m)
        v2d = V[:, c].reshape(m, p)
        w2d = W[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_reconstructed[i * m * p + j * p + k] += u2d[i, j] * v2d[j, k] * w2d[k, i]

    max_err = np.max(np.abs(T_reconstructed - target))
    if max_err > 1e-10:
        print(f"Verification failed! Max error: {max_err}")
        # Fallback to schoolbook
        U = np.zeros((n * m, total_terms), dtype=np.float64)
        V = np.zeros((m * p, total_terms), dtype=np.float64)
        W = np.zeros((p * n, total_terms), dtype=np.float64)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1
                    V[j * p + k, idx] = 1
                    W[k * n + i, idx] = 1
                    idx += 1
        rank = total_terms

    # Convert to lists for return
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    print(f"Rank: {U.shape[1]}")

    T_reconstructed = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_reconstructed[i, j, k] = 0.0
                for c in range(U.shape[1]):
                    T_reconstructed[i, j, k] += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]

    T_expected = np.ones((n, m, p))
    max_err = np.max(np.abs(T_reconstructed - T_expected))
    print(f"Max error: {max_err}")

    if max_err < 1e-10:
        print("Decomposition is correct!")
    else:
        print("Decomposition has errors!")
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if abs(T_reconstructed[i, j, k] - 1.0) > 1e-10:
                        print(f"  T[{i},{j},{k}] = {T_reconstructed[i,j,k]} (expected 1.0)")
