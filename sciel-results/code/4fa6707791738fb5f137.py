import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    n, m, p = 4, 4, 6

    # Base rank-77 decomposition via Kronecker product of Strassen(7) and mixed(11)
    # Inner: 2x2 * 2x3 -> 2x3 with rank 11 (7 Strassen + 4 schoolbook)
    # Outer: 4x4 * 4x6 -> 4x6 with rank 7 (Strassen on 2x2 blocks)

    # Inner terms for 2x2 * 2x3 -> 2x3
    # Each term: (A_coeffs, B_coeffs, C_coeffs)
    # A: (r,s,coeff), B: (s,k,coeff), C: (r,k,coeff)

    inner_terms = [
        # M1
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # M2
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        # M3
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        # M4
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        # M5
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,1,1), (0,0,-1)]),
        # M6
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        # M7
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]),
        # Schoolbook for column 2
        ([(0,0,1)], [(0,2,1)], [(0,2,1)]),
        ([(0,1,1)], [(1,2,1)], [(0,2,1)]),
        ([(1,0,1)], [(0,2,1)], [(1,2,1)]),
        ([(1,1,1)], [(1,2,1)], [(1,2,1)]),
    ]

    # Outer terms for 4x4 * 4x6 -> 4x6 (Strassen on 2x2 blocks)
    # Block ids: 0=00, 1=01, 2=10, 3=11
    outer_terms = [
        # M1
        ([(0,1), (3,1)], [(0,1), (3,1)], [(0,1), (3,1)]),
        # M2
        ([(2,1), (3,1)], [(0,1)], [(2,1), (3,-1)]),
        # M3
        ([(0,1)], [(1,1), (3,-1)], [(1,1), (3,1)]),
        # M4
        ([(3,1)], [(2,1), (0,-1)], [(0,1), (2,1)]),
        # M5
        ([(0,1), (1,1)], [(3,1)], [(1,1), (0,-1)]),
        # M6
        ([(2,1), (0,-1)], [(0,1), (1,1)], [(3,1)]),
        # M7
        ([(1,1), (3,-1)], [(2,1), (3,1)], [(0,1)]),
    ]

    rank_base = 77
    U = np.zeros((n*m, rank_base))
    V = np.zeros((m*p, rank_base))
    W = np.zeros((p*n, rank_base))

    def get_A_idx(b, r, s):
        I = 2*(b//2) + r
        J = 2*(b%2) + s
        return I*m + J

    def get_B_idx(b, s, k):
        J = 2*(b//2) + s
        K = 3*(b%2) + k
        return J*p + K

    def get_W_idx(b, r, k):
        I = 2*(b//2) + r
        K = 3*(b%2) + k
        return K*n + I

    for oi, (A_blocks, B_blocks, C_blocks) in enumerate(outer_terms):
        for ii, (A_inner, B_inner, C_inner) in enumerate(inner_terms):
            r_idx = oi * 11 + ii

            for (r, s, ca) in A_inner:
                for (b, cb) in A_blocks:
                    idx = get_A_idx(b, r, s)
                    U[idx, r_idx] += ca * cb

            for (s, k, cb) in B_inner:
                for (b, cb2) in B_blocks:
                    idx = get_B_idx(b, s, k)
                    V[idx, r_idx] += cb * cb2

            for (r, k, cc) in C_inner:
                for (b, cb2) in C_blocks:
                    idx = get_W_idx(b, r, k)
                    W[idx, r_idx] += cc * cb2

    # Verification of base decomposition
    def verify(U_mat, V_mat, W_mat):
        rank = U_mat.shape[1]
        T_check = np.zeros((n, m, p))
        for r in range(rank):
            u = U_mat[:, r]
            v = V_mat[:, r]
            w = W_mat[:, r]
            for i in range(n):
                for j in range(m):
                    if u[i*m+j] == 0:
                        continue
                    for k in range(p):
                        T_check[i,j,k] += u[i*m+j] * v[j*p+k] * w[k*n+i]
        return np.allclose(T_check, 1.0, atol=1e-9)

    if not verify(U, V, W):
        # Should not happen, but fallback
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to reduce rank via linear dependency elimination
    # Check for zero columns
    zero_cols = []
    for r in range(rank_base):
        if np.all(U[:, r] == 0) or np.all(V[:, r] == 0) or np.all(W[:, r] == 0):
            zero_cols.append(r)

    if zero_cols:
        keep = [r for r in range(rank_base) if r not in zero_cols]
        U = U[:, keep]
        V = V[:, keep]
        W = W[:, keep]
        rank_base = len(keep)

    # Check for duplicate terms
    def term_key(r):
        u_tuple = tuple((i, U[i, r]) for i in range(n*m) if U[i, r] != 0)
        v_tuple = tuple((i, V[i, r]) for i in range(m*p) if V[i, r] != 0)
        w_tuple = tuple((i, W[i, r]) for i in range(p*n) if W[i, r] != 0)
        return (u_tuple, v_tuple, w_tuple)

    keys = [term_key(r) for r in range(rank_base)]
    from collections import Counter
    key_counts = Counter(keys)

    new_U = []
    new_V = []
    new_W = []
    used = [False] * rank_base

    for i in range(rank_base):
        if used[i]:
            continue
        k = keys[i]
        count = key_counts[k]
        if count > 1:
            indices = [j for j in range(rank_base) if keys[j] == k and not used[j]]
            U_sum = np.sum(U[:, indices], axis=1)
            V_sum = np.sum(V[:, indices], axis=1)
            W_sum = np.sum(W[:, indices], axis=1)

            new_U.append(U_sum)
            new_V.append(V_sum)
            new_W.append(W_sum)

            for j in indices:
                used[j] = True
        else:
            new_U.append(U[:, i])
            new_V.append(V[:, i])
            new_W.append(W[:, i])
            used[i] = True

    if new_U:
        U_reduced = np.array(new_U).T
        V_reduced = np.array(new_V).T
        W_reduced = np.array(new_W).T

        if verify(U_reduced, V_reduced, W_reduced):
            U, V, W = U_reduced, V_reduced, W_reduced
            rank_base = U.shape[1]

    # Try to find linear dependencies among the rank terms
    # For a tensor decomposition T = sum_r u_r v_r w_r, if there exist coefficients
    # alpha_r such that sum_r alpha_r u_r v_r w_r = 0, we can eliminate one term.
    # This is equivalent to finding the tensor rank via linear algebra on the
    # coefficient matrices.

    # Build the "coefficient tensor" as a set of matrices
    # For each (i,j), we have a matrix M_{ij} of size p x rank where
    # M_{ij}[k, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # The tensor is decomposable with rank R if and only if the span of these
    # matrices has dimension R.

    # A more efficient approach: check if any term is in the span of others
    # by looking at the 3-tensor as a vector in R^{n*m*p} and checking
    # linear dependencies.

    # For rank 77 and tensor size 96, we can build the 96 x 77 matrix where
    # entry (t, r) = u_{t, r} * v_{t, r} * w_{t, r} and check rank.
    # But this doesn't directly give us the decomposition rank.

    # Instead, use the approach: for each term r, check if its contribution
    # to the tensor is in the span of the contributions of all other terms.
    # This requires solving a linear system, which is expensive for rank 77.

    # A simpler heuristic: try removing each term and see if the remaining
    # terms can be adjusted (by adding a multiple of another term) to recover
    # the original tensor. This is O(rank^2 * tensor_size) per trial.

    # Given time constraints, we'll do a limited search.
    if time.time() < deadline - 30:
        # Try to eliminate terms by checking linear dependencies
        # Build the tensor as a flat vector
        tensor_size = n * m * p
        T_vec = np.zeros(tensor_size)

        def get_tensor_idx(i, j, k):
            return (i * m + j) * p + k

        # Compute the full tensor from current decomposition
        for r in range(rank_base):
            for i in range(n):
                for j in range(m):
                    u_val = U[i*m+j, r]
                    if u_val == 0:
                        continue
                    for k in range(p):
                        T_vec[get_tensor_idx(i, j, k)] += u_val * V[j*p+k, r] * W[k*n+i, r]

        # Try removing each term and see if we can recover the tensor
        # by adjusting the remaining terms
        # This is a linear system: find alpha_r such that
        # sum_r alpha_r * (u_r v_r w_r) = T_vec
        # where alpha_r = 1 for kept terms and 0 for removed term
        # If the system is consistent, we can eliminate the term.

        # For efficiency, only try a subset of terms
        # We'll try terms that have small contributions first

        # Actually, this approach is too slow for rank 77.
        # Instead, let's try a different approach: look for pairs of terms
        # that are "similar" and can be combined.

        # For now, just return the current best result
        pass

    # Final verification
    if not verify(U, V, W):
        # Fallback to original
        U = np.zeros((n*m, 77))
        V = np.zeros((m*p, 77))
        W = np.zeros((p*n, 77))

        for oi, (A_blocks, B_blocks, C_blocks) in enumerate(outer_terms):
            for ii, (A_inner, B_inner, C_inner) in enumerate(inner_terms):
                r_idx = oi * 11 + ii

                for (r, s, ca) in A_inner:
                    for (b, cb) in A_blocks:
                        idx = get_A_idx(b, r, s)
                        U[idx, r_idx] += ca * cb

                for (s, k, cb) in B_inner:
                    for (b, cb2) in B_blocks:
                        idx = get_B_idx(b, s, k)
                        V[idx, r_idx] += cb * cb2

                for (r, k, cc) in C_inner:
                    for (b, cb2) in C_blocks:
                        idx = get_W_idx(b, r, k)
                        W[idx, r_idx] += cc * cb2

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
