import numpy as np
import time
import sys

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill

    n, m, p = 4, 4, 7

    def a_idx(i, j):
        return i * m + j

    def b_idx(j, k):
        return j * p + k

    def c_idx(k, i):
        return k * n + i

    # Build the target tensor M[a, b, c]
    # M[a,b,c] = 1 iff a = i*m+j, b = j*p+k, c = k*n+i for some (i,j,k)
    # i.e., M[a,b,c] = 1 iff j = a % m, j = b // p, and i = a // m, i = c % n, k = b % p, k = c // n
    # More precisely: M[a,b,c] = 1 iff (a//m, a%m) and (b//p, b%p) and (c//n, c%n)
    #   satisfy: a%m == b//p and a//m == c%n and b%p == c//n

    # Schoolbook decomposition: rank = n*m*p = 112
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[a_idx(i, j), r] = 1
                    V[b_idx(j, k), r] = 1
                    W[c_idx(k, i), r] = 1
                    r += 1
        return U, V, W

    def verify(U, V, W, atol=1e-8):
        """Check if sum_r U[:,r] * V[:,r] * W[:,r] (tensor product) matches the matmul tensor."""
        rank = U.shape[1]
        # Build the tensor via einsum-like contraction
        # M[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Check against the true tensor
        # True tensor: M[a,b,c] = 1 iff a%m == b//p and a//m == c%n and b%p == c//n
        # For efficiency, check a few random (a,b,c) triples plus all "active" ones
        # Actually, let's check all active entries (there are n*m*p = 112)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = a_idx(i, j)
                    b = b_idx(j, k)
                    c = c_idx(k, i)
                    val = 0.0
                    for r in range(rank):
                        val += U[a, r] * V[b, r] * W[c, r]
                    if abs(val - 1.0) > atol:
                        return False
        # Also check some inactive entries to make sure we don't have spurious non-zeros
        # Check a sample of inactive entries
        rng = np.random.default_rng(12345)
        for _ in range(200):
            a = rng.integers(0, n * m)
            b = rng.integers(0, m * p)
            c = rng.integers(0, p * n)
            # Check if this is an active entry
            i_a, j_a = a // m, a % m
            j_b, k_b = b // p, b % p
            k_c, i_c = c // n, c % n
            if not (j_a == j_b and i_a == i_c and k_b == k_c):
                val = 0.0
                for r in range(rank):
                    val += U[a, r] * V[b, r] * W[c, r]
                if abs(val) > atol:
                    return False
        return True

    def verify_fast(U, V, W, atol=1e-8):
        """Faster verification using matrix operations."""
        rank = U.shape[1]
        # M[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # For each r, the contribution to the tensor is U[:,r] outer V[:,r] outer W[:,r]
        # We can check: for each (a,c), M[a,?,c] as a function of b
        # M[a,b,c] = sum_r U[a,r] * W[c,r] * V[b,r]
        # Let X_r = U[:,r] * W[:,r] (element-wise, shape n*m x p*n)
        # Then M[a,b,c] = sum_r X_r[a,c] * V[b,r]
        # So for fixed (a,c): M[a,:,c] = sum_r X_r[a,c] * V[:,r] = (X[:,a,c] @ V.T) ... no
        # M[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # = sum_r (U[a,r] * W[c,r]) * V[b,r]
        # For fixed a,c: the vector over b is sum_r (U[a,r]*W[c,r]) * V[b,r]
        # = (U[a,:] * W[c,:]) @ V.T  ... no, V is (m*p, rank), so V[b,r]
        # = sum_r V[b,r] * (U[a,r] * W[c,r])
        # Let's just do it directly but vectorized over r
        # Build the full tensor: shape (n*m, m*p, p*n) - too big (16*28*28 = 12544, actually fine)
        # M = sum_r np.einsum('r,sr,tr->rst', ...) no
        # M[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # = np.einsum('ar,br,cr->abc', U, V, W)  -- but this creates a (16,28,28) array, fine
        # Actually let's just compute the (16,28,28) tensor and check
        # M_full[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # We can compute this as: for each r, outer(U[:,r], V[:,r], W[:,r])
        # Or: M_full = np.einsum('ar,br,cr->abc', U, V, W)
        M_full = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        # Check active entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = a_idx(i, j)
                    b = b_idx(j, k)
                    c = c_idx(k, i)
                    if abs(M_full[a, b, c] - 1.0) > atol:
                        return False
        # Check a sample of inactive entries
        rng = np.random.default_rng(12345)
        for _ in range(300):
            a = rng.integers(0, n * m)
            b = rng.integers(0, m * p)
            c = rng.integers(0, p * n)
            i_a, j_a = a // m, a % m
            j_b, k_b = b // p, b % p
            k_c, i_c = c // n, c % n
            if not (j_a == j_b and i_a == i_c and k_b == k_c):
                if abs(M_full[a, b, c]) > atol:
                    return False
        return True

    def snap_to_halfint(U, V, W, threshold=1e-6):
        """Snap values to nearest half-integer, zero out small values."""
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        U_s[np.abs(U_s) < threshold] = 0
        V_s[np.abs(V_s) < threshold] = 0
        W_s[np.abs(W_s) < threshold] = 0
        return U_s, V_s, W_s

    def try_rank_reduction(U, V, W, target_rank, rng, max_iters=50):
        """Try to reduce rank from current to target_rank using iterative refinement."""
        current_rank = U.shape[1]
        if target_rank >= current_rank:
            return None

        # Randomized SVD approach: project factors to lower rank
        # U: (16, R), V: (28, R), W: (28, R)
        # We want to find U', V', W' of rank target_rank such that
        # sum_r U'[a,r'] V'[b,r'] W'[c,r'] = sum_r U[a,r] V[b,r] W[c,r]

        # Approach: use the fact that if we have a valid decomposition of rank R,
        # we can try to find a lower-rank one by:
        # 1. Randomly project the rank dimension
        # 2. Iteratively refine

        # Method: alternating least squares style
        # Fix V, W; solve for U (and vice versa)
        # M[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # For fixed V, W: M[a,b,c] as function of a: M[:,b,c] = U @ (V[b,:]*W[c,:]).T
        # This is linear in U. We can solve for U given V,W.
        # Similarly for V given U,W, and W given U,V.

        # But we need to enforce rank = target_rank.
        # Let's try: start with random U0, V0, W0 of target_rank, then iterate.

        best_U, best_V, best_W = None, None, None
        best_err = np.inf

        for trial in range(3):
            # Random initialization
            U0 = rng.standard_normal((n * m, target_rank))
            V0 = rng.standard_normal((m * p, target_rank))
            W0 = rng.standard_normal((p * n, target_rank))

            # Scale to match the original tensor's norm
            # Original: sum_r U[a,r]V[b,r]W[c,r] = 1 for active entries
            # The "energy" of each term is roughly ||U[:,r]||*||V[:,r]||*||W[:,r]||
            # Let's normalize
            for r in range(target_rank):
                nu = np.linalg.norm(U0[:, r])
                nv = np.linalg.norm(V0[:, r])
                nw = np.linalg.norm(W0[:, r])
                if nu > 1e-10 and nv > 1e-10 and nw > 1e-10:
                    scale = 1.0 / (nu * nv * nw) ** (1/3)
                    U0[:, r] *= scale
                    V0[:, r] *= scale
                    W0[:, r] *= scale

            # Iterative refinement: ALS
            for it in range(max_iters):
                # Solve for U given V, W
                # M[a,b,c] = sum_r U[a,r] * (V[b,r] * W[c,r])
                # For each (b,c): M[:,b,c] = U @ diag(V[b,:]*W[c,:]).T ... no
                # M[:,b,c] = U @ (V[b,:]*W[c,:])  (since sum_r U[a,r] * V[b,r] * W[c,r])
                # So: U @ (V[b,:]*W[c,:]).T = M[:,b,c]
                # Stack all (b,c) equations: U @ B = C where B[b,c,r] = V[b,r]*W[c,r]
                # B has shape (m*p * p*n, target_rank) = (28*28, target_rank) = (784, target_rank)
                # C has shape (16, 784) ... wait, M[:,b,c] has shape (16,)
                # So: for each (b,c) pair, U @ (V[b,:]*W[c,:]) = M[:,b,c]
                # Let's build: U (16 x R) @ B (R x 784) = C (16 x 784)
                # B[r, b*p*n+c] = V[b,r] * W[c,r]
                # C[a, b*p*n+c] = M[a,b,c]

                # Build B
                B_mat = np.zeros((target_rank, m * p * p * n))
                for b in range(m * p):
                    for c in range(p * n):
                        B_mat[:, b * p * n + c] = V0[b, :] * W0[c, :]

                # Build C from the true tensor
                C_mat = np.zeros((n * m, m * p * p * n))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = a_idx(i, j)
                            b = b_idx(j, k)
                            c = c_idx(k, i)
                            C_mat[a, b * p * n + c] = 1.0

                # Solve U @ B_mat = C_mat => U = C_mat @ B_mat^+ (pseudo-inverse)
                # B_mat is (R x 784), C_mat is (16 x 784)
                # U (16 x R) = C_mat (16 x 784) @ pinv(B_mat) (784 x R)
                try:
                    U_new = C_mat @ np.linalg.pinv(B_mat)
                except:
                    U_new = U0

                # Solve for V given U, W
                # M[a,b,c] = sum_r V[b,r] * (U[a,r] * W[c,r])
                # V @ (U[a,:]*W[c,:]) = M[a,:,c]
                # B2[r, a*p*n+c] = U[a,r] * W[c,r]
                # C2[b, a*p*n+c] = M[a,b,c]
                B2_mat = np.zeros((target_rank, n * m * p * n))
                for a in range(n * m):
                    for c in range(p * n):
                        B2_mat[:, a * p * n + c] = U_new[a, :] * W0[c, :]

                C2_mat = np.zeros((m * p, n * m * p * n))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = a_idx(i, j)
                            b = b_idx(j, k)
                            c = c_idx(k, i)
                            C2_mat[b, a * p * n + c] = 1.0

                try:
                    V_new = C2_mat @ np.linalg.pinv(B2_mat)
                except:
                    V_new = V0

                # Solve for W given U, V
                # M[a,b,c] = sum_r W[c,r] * (U[a,r] * V[b,r])
                # W @ (U[a,:]*V[b,:]) = M[a,b,:]
                # B3[r, a*m*p+b] = U[a,r] * V[b,r]
                # C3[c, a*m*p+b] = M[a,b,c]
                B3_mat = np.zeros((target_rank, n * m * m * p))
                for a in range(n * m):
                    for b in range(m * p):
                        B3_mat[:, a * m * p + b] = U_new[a, :] * V_new[b, :]

                C3_mat = np.zeros((p * n, n * m * m * p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = a_idx(i, j)
                            b = b_idx(j, k)
                            c = c_idx(k, i)
                            C3_mat[c, a * m * p + b] = 1.0

                try:
                    W_new = C3_mat @ np.linalg.pinv(B3_mat)
                except:
                    W_new = W0

                U0, V0, W0 = U_new, V_new, W_new

                # Check convergence
                if it % 10 == 9:
                    # Quick error check
                    M_check = np.einsum('ar,br,cr->abc', U0, V0, W0, optimize=True)
                    err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                a = a_idx(i, j)
                                b = b_idx(j, k)
                                c = c_idx(k, i)
                                err += (M_check[a, b, c] - 1.0) ** 2
                    err = np.sqrt(err)
                    if err < best_err:
                        best_err = err
                        best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
                    if err < 1e-10:
                        break

        if best_U is not None and best_err < 1e-8:
            # Try snapping to half-integers
            Us, Vs, Ws = snap_to_halfint(best_U, best_V, best_W)
            if verify_fast(Us, Vs, Ws):
                return Us, Vs, Ws
            # Return the non-snapped version if it's exact
            if verify_fast(best_U, best_V, best_W):
                return best_U, best_V, best_W
        return None

    # Start with schoolbook
    U, V, W = schoolbook()
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # Try to reduce rank
    # The known lower bound for 4x4x7 is at most 97 (Strassen 49 + schoolbook 48)
    # Try various target ranks
    rng = np.random.default_rng(42)

    # First, try to get to rank 97 (Strassen + schoolbook)
    # Actually, let's try a different approach:
    # Use the Strassen construction for the 4x4x4 part and schoolbook for 4x4x3
    # This gives rank 97. Let's construct this directly.

    def strassen_4x4x4_plus_schoolbook_4x4x3():
        """Construct rank-97 decomposition: Strassen for 4x4x4 + schoolbook for 4x4x3."""
        rank = 0
        U_list = []
        V_list = []
        W_list = []

        def add_term(u_entries, v_entries, w_entries):
            nonlocal rank
            r = rank
            rank += 1
            for (idx, val) in u_entries:
                U_list.append((idx, r, val))
            for (idx, val) in v_entries:
                V_list.append((idx, r, val))
            for (idx, val) in w_entries:
                W_list.append((idx, r, val))

        # Strassen 4x4x4: 49 terms
        # Level-1 products
        products = [
            # M1: (A11+A22)(B11+B22) -> C11, C22
            {
                'U': [((0,0), 1), ((1,1), 1)],
                'V': [((0,0), 1), ((1,1), 1)],
                'W': [((0,0), 1), ((1,1), 1)],
            },
            # M2: (A21+A22)B11 -> C21
            {
                'U': [((1,0), 1), ((1,1), 1)],
                'V': [((0,0), 1)],
                'W': [((0,1), 1)],
            },
            # M3: A11(B12-B22) -> C12
            {
                'U': [((0,0), 1)],
                'V': [((0,1), 1), ((1,1), -1)],
                'W': [((1,0), 1)],
            },
            # M4: A22(B21-B11) -> C11, C22
            {
                'U': [((1,1), 1)],
                'V': [((1,0), 1), ((0,0), -1)],
                'W': [((0,0), 1), ((1,1), 1)],
            },
            # M5: (A11+A12)B22 -> C11, C12
            {
                'U': [((0,0), 1), ((0,1), 1)],
                'V': [((1,1), 1)],
                'W': [((0,0), 1), ((1,0), 1)],
            },
            # M6: (A21-A11)(B11+B12) -> C22
            {
                'U': [((1,0), 1), ((0,0), -1)],
                'V': [((0,0), 1), ((0,1), 1)],
                'W': [((1,1), 1)],
            },
            # M7: (A12-A22)(B21+B22) -> C11, C21
            {
                'U': [((0,1), 1), ((1,1), -1)],
                'V': [((1,0), 1), ((1,1), 1)],
                'W': [((0,0), 1), ((0,1), 1)],
            },
        ]

        # Strassen 7-term formula for 2x2x2
        strassen_terms = [
            ([((0,0), 1), ((1,1), 1)],
             [((0,0), 1), ((1,1), 1)],
             [((0,0), 1), ((1,1), 1)]),
            ([((1,0), 1), ((1,1), 1)],
             [((0,0), 1)],
             [((1,0), 1)]),
            ([((0,0), 1)],
             [((0,1), 1), ((1,1), -1)],
             [((0,1), 1)]),
            ([((1,1), 1)],
             [((1,0), 1), ((0,0), -1)],
             [((0,0), 1), ((1,1), 1)]),
            ([((0,0), 1), ((0,1), 1)],
             [((1,1), 1)],
             [((0,0), 1), ((0,1), 1)]),
            ([((1,0), 1), ((0,0), -1)],
             [((0,0), 1), ((0,1), 1)],
             [((1,1), 1)]),
            ([((0,1), 1), ((1,1), -1)],
             [((1,0), 1), ((1,1), 1)],
             [((0,0), 1), ((1,0), 1)]),
        ]

        for prod in products:
            U_blocks = prod['U']
            V_blocks = prod['V']
            W_blocks = prod['W']

            # Build virtual matrices
            VA = {}
            for (bi, bj), coeff in U_blocks:
                for di in range(2):
                    for dj in range(2):
                        gi = 2 * bi + di
                        gj = 2 * bj + dj
                        gidx = a_idx(gi, gj)
                        key = (di, dj)
                        if key not in VA:
                            VA[key] = {}
                        VA[key][gidx] = VA[key].get(gidx, 0) + coeff

            VB = {}
            for (bj, bk), coeff in V_blocks:
                for dj in range(2):
                    for dk in range(2):
                        gj = 2 * bj + dj
                        gk = 2 * bk + dk
                        gidx = b_idx(gj, gk)
                        key = (dj, dk)
                        if key not in VB:
                            VB[key] = {}
                        VB[key][gidx] = VB[key].get(gidx, 0) + coeff

            WC = {}
            for (bk, bi), coeff in W_blocks:
                for dk in range(2):
                    for di in range(2):
                        gk = 2 * bk + dk
                        gi = 2 * bi + di
                        gidx = c_idx(gk, gi)
                        key = (dk, di)
                        if key not in WC:
                            WC[key] = {}
                        WC[key][gidx] = WC[key].get(gidx, 0) + coeff

            for (u_ents, v_ents, w_ents) in strassen_terms:
                u_global = []
                for (key, sc) in u_ents:
                    if key in VA:
                        for gidx, vc in VA[key].items():
                            u_global.append((gidx, sc * vc))

                v_global = []
                for (key, sc) in v_ents:
                    if key in VB:
                        for gidx, vc in VB[key].items():
                            v_global.append((gidx, sc * vc))

                w_global = []
                for (key, sc) in w_ents:
                    if key in WC:
                        for gidx, vc in WC[key].items():
                            w_global.append((gidx, sc * vc))

                u_global = [(idx, c) for idx, c in u_global if c != 0]
                v_global = [(idx, c) for idx, c in v_global if c != 0]
                w_global = [(idx, c) for idx, c in w_global if c != 0]

                if u_global and v_global and w_global:
                    add_term(u_global, v_global, w_global)

        # Schoolbook for 4x4x3 (k=4,5,6)
        for i in range(n):
            for j in range(m):
                for k in range(4, p):
                    add_term(
                        [(a_idx(i, j), 1)],
                        [(b_idx(j, k), 1)],
                        [(c_idx(k, i), 1)]
                    )

        r = rank
        U_mat = np.zeros((n * m, r))
        V_mat = np.zeros((m * p, r))
        W_mat = np.zeros((p * n, r))
        for (row, rr, val) in U_list:
            U_mat[row, rr] = val
        for (row, rr, val) in V_list:
            V_mat[row, rr] = val
        for (row, rr, val) in W_list:
            W_mat[row, rr] = val

        return U_mat, V_mat, W_mat

    # Try the Strassen+schoolbook construction
    if time.time() < deadline - 10:
        U_s, V_s, W_s = strassen_4x4x4_plus_schoolbook_4x4x3()
        if verify_fast(U_s, V_s, W_s):
            best_rank = U_s.shape[1]
            best_U, best_V, best_W = U_s.copy(), V_s.copy(), W_s.copy()
        else:
            # Fallback: schoolbook
            best_rank = n * m * p
            best_U, best_V, best_W = schoolbook()

    # Try to reduce further using numerical methods
    if time.time() < deadline - 20 and best_rank > 90:
        # Try to reduce from best_rank to lower values
        for target in range(best_rank - 1, max(80, best_rank - 15), -1):
            if time.time() > deadline - 10:
                break
            result = try_rank_reduction(best_U, best_V, best_W, target, rng, max_iters=30)
            if result is not None:
                U_r, V_r, W_r = result
                if verify_fast(U_r, V_r, W_r):
                    best_rank = target
                    best_U, best_V, best_W = U_r, V_r, W_r

    # Final: ensure we have a valid result
    if not verify_fast(best_U, best_V, best_W):
        # Last resort: schoolbook
        best_U, best_V, best_W = schoolbook()
        best_rank = n * m * p

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
