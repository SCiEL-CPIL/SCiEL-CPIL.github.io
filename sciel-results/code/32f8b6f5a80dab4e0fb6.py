import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # n=2, m=5, p=5

    terms = []

    def add_strassen(i_start, j_start, k_start):
        """Add 7 terms for a 2x2x2 Strassen block."""
        # M1: (a+b)(c+d)
        u = {(i_start, j_start): 1, (i_start+1, j_start+1): 1}
        v = {(j_start, k_start): 1, (j_start+1, k_start+1): 1}
        w = {(k_start, i_start): 1, (k_start+1, i_start+1): 1}
        terms.append((u, v, w))

        # M2: b(c-d)
        u = {(i_start+1, j_start): 1, (i_start+1, j_start+1): 1}
        v = {(j_start, k_start): 1}
        w = {(k_start, i_start+1): 1, (k_start+1, i_start+1): -1}
        terms.append((u, v, w))

        # M3: a(d-c)
        u = {(i_start, j_start): 1}
        v = {(j_start, k_start+1): 1, (j_start+1, k_start+1): -1}
        w = {(k_start+1, i_start): 1, (k_start+1, i_start+1): 1}
        terms.append((u, v, w))

        # M4: d(a-b)
        u = {(i_start+1, j_start+1): 1}
        v = {(j_start+1, k_start): 1, (j_start, k_start): -1}
        w = {(k_start, i_start): 1, (k_start, i_start+1): 1}
        terms.append((u, v, w))

        # M5: (a+b)e
        u = {(i_start, j_start): 1, (i_start, j_start+1): 1}
        v = {(j_start+1, k_start+1): 1}
        w = {(k_start, i_start): -1, (k_start+1, i_start): 1}
        terms.append((u, v, w))

        # M6: (a-c)f
        u = {(i_start+1, j_start): 1, (i_start, j_start): -1}
        v = {(j_start, k_start): 1, (j_start, k_start+1): 1}
        w = {(k_start+1, i_start+1): 1}
        terms.append((u, v, w))

        # M7: (b+d)g
        u = {(i_start, j_start+1): 1, (i_start+1, j_start+1): -1}
        v = {(j_start+1, k_start): 1, (j_start+1, k_start+1): 1}
        w = {(k_start, i_start): 1}
        terms.append((u, v, w))

    def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
        """Add schoolbook terms for a rectangular block."""
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    u = {(i, j): 1}
                    v = {(j, k): 1}
                    w = {(k, i): 1}
                    terms.append((u, v, w))

    # Build the standard Strassen+schoolbook decomposition (rank 46)
    # j={0,1}, k={0,1}: Strassen
    add_strassen(0, 0, 0)
    # j={0,1}, k={2,3}: Strassen
    add_strassen(0, 0, 2)
    # j={0,1}, k={4}: schoolbook (4 terms)
    add_schoolbook(0, 2, 0, 2, 4, 5)

    # j={2,3}, k={0,1}: Strassen
    add_strassen(0, 2, 0)
    # j={2,3}, k={2,3}: Strassen
    add_strassen(0, 2, 2)
    # j={2,3}, k={4}: schoolbook (4 terms)
    add_schoolbook(0, 2, 2, 4, 4, 5)

    # j={4}, k={0,1,2,3,4}: schoolbook (10 terms)
    add_schoolbook(0, 2, 4, 5, 0, 5)

    rank = len(terms)  # should be 46

    # Build matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    for c in range(rank):
        u_dict, v_dict, w_dict = terms[c]
        for (i, j), val in u_dict.items():
            U[i * m + j, c] = val
        for (j, k), val in v_dict.items():
            V[j * p + k, c] = val
        for (k, i), val in w_dict.items():
            W[k * n + i, c] = val

    # Verify the decomposition
    tensor_recon = np.zeros((n, m, p))
    for c in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor_recon[i, j, k] += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]

    target = np.ones((n, m, p))
    if not np.allclose(tensor_recon, target, atol=1e-10):
        # Fallback: return schoolbook rank-50
        terms2 = []
        add_schoolbook(0, 2, 0, 5, 0, 5)
        rank2 = len(terms2)
        U2 = np.zeros((n * m, rank2))
        V2 = np.zeros((m * p, rank2))
        W2 = np.zeros((p * n, rank2))
        for c in range(rank2):
            u_dict, v_dict, w_dict = terms2[c]
            for (i, j), val in u_dict.items():
                U2[i * m + j, c] = val
            for (j, k), val in v_dict.items():
                V2[j * p + k, c] = val
            for (k, i), val in w_dict.items():
                W2[k * n + i, c] = val
        return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}

    # Now try to reduce rank by finding linear dependencies
    # Strategy: for each column r, check if the rank-1 tensor U_r⊗V_r⊗W_r
    # can be expressed as a linear combination of the others.
    # This means: does there exist alpha_s (s≠r) such that
    # sum_{s≠r} alpha_s * (U_s ⊗ V_s ⊗ W_s) = U_r ⊗ V_r ⊗ W_r
    
    # Flatten all rank-1 tensors into vectors in R^{n*m*p}
    # T_r = vec(U_r ⊗ V_r ⊗ W_r)
    # Check if T_r is in span{T_s : s≠r}

    def vec_tensor(u_vec, v_vec, w_vec):
        """Flatten a rank-1 tensor u⊗v⊗w into a vector."""
        # u is in R^{n*m}, v in R^{m*p}, w in R^{p*n}
        # Tensor T[i,j,k] = u[i*m+j] * v[j*p+k] * w[k*n+i]
        # Flatten in (i,j,k) order
        result = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    result[i, j, k] = u[i*m+j] * v[j*p+k] * w[k*n+i]
        return result.flatten()

    # Build matrix of all flattened tensors
    T_mat = np.zeros((n*m*p, rank))
    for c in range(rank):
        T_mat[:, c] = vec_tensor(U[:, c], V[:, c], W[:, c])

    # For each column, check if it's in the span of the others
    # Use QR with column pivoting or SVD
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank

    # Try removing columns one at a time
    # For column r, project T_r onto span of {T_s : s≠r}
    # If the residual is ~0, we can remove column r

    removable = set()
    for r in range(rank):
        if time.time() - start_time > deadline - 5:
            break
        others = [c for c in range(rank) if c != r]
        if len(others) < 1:
            continue
        # Build matrix of other tensors
        A = T_mat[:, others]  # shape (n*m*p, rank-1)
        b = T_mat[:, r]  # shape (n*m*p,)
        
        # Solve least squares: min ||A*x - b||
        try:
            x, residuals, rank_qr, sv = np.linalg.lstsq(A, b, rcond=1e-10)
            residual = np.linalg.norm(A @ x - b)
            norm_b = np.linalg.norm(b)
            if norm_b > 1e-15 and residual / norm_b < 1e-8:
                removable.add(r)
        except:
            pass

    if removable:
        # Remove the first removable column
        keep = [c for c in range(rank) if c not in removable]
        if len(keep) < rank:
            new_rank = len(keep)
            best_U = U[:, keep]
            best_V = V[:, keep]
            best_W = W[:, keep]
            best_rank = new_rank

    # Try a different tiling: group j={4} with j={2,3} for some k values
    # to see if we can get a better structure
    # Alternative: use 2x3x3 blocks where possible

    # Try: j={2,3,4} as a 2x3x3 block for k={0,1,2}
    # Schoolbook for 2x3x3 = 18, but maybe we can do better
    # For now, stick with the current approach

    # Try yet another approach: for j={4}, the contribution is
    # C[i,k] += A[i,4]*B[4,k] for all i,k
    # This is an outer product. As a bilinear map, it has tensor rank 1
    # in the sense that it's a single term if we allow the W part to be
    # a full vector. But in our formulation, each term must be a product
    # of three vectors, so the outer product A[:,4]⊗B[4,:] in the output
    # requires 10 separate terms (one for each (i,k) pair).
    # However, we can potentially merge some of these with Strassen terms.

    # Let's try: for j={4}, k={0,1}, use a 2x1x2 "Strassen-like" block
    # C[i,0] += A[i,4]*B[4,0]
    # C[i,1] += A[i,4]*B[4,1]
    # This is just 4 terms in schoolbook. No savings possible since
    # there's only one j value.

    # The key insight: the Strassen savings come from having 2 j-values
    # and 2 k-values, allowing 7 terms instead of 8.
    # For j={4}, we only have 1 j-value, so no Strassen savings.

    # Try to see if we can reorganize to get more 2x2x2 blocks
    # Current: 4 Strassen blocks (j={0,1}×k={0,1}, j={0,1}×k={2,3},
    #          j={2,3}×k={0,1}, j={2,3}×k={2,3})
    # Plus schoolbook for the rest

    # What if we use j={1,2} instead of j={0,1} and j={2,3}?
    # Then we'd have Strassen for j={1,2}×k={0,1} and j={1,2}×k={2,3}
    # And schoolbook for j={0}, j={3,4}
    # This gives the same number of terms, so no improvement.

    # Try: what if we use a 2x2x3 or 2x3x2 block?
    # For 2x2x3: schoolbook = 12. Can we do it in 11?
    # The tensor for 2x2x3 matrix mult has rank at most 12 (schoolbook).
    # It's unknown if it can be done in 11.

    # For now, return the best we have
    elapsed = time.time() - start_time

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
