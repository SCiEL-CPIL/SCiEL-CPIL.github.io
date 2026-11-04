# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 135.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8
    nm = n * m  # 6
    mp = m * p  # 24
    pn = p * n  # 16

    # Schoolbook fallback
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if acc != 1:
                        return False
        return True

    # Strassen 2x2x2 decomposition
    S_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    # Build 44-term decomposition
    rank_44 = 44
    U_44 = [[0] * rank_44 for _ in range(nm)]
    V_44 = [[0] * rank_44 for _ in range(mp)]
    W_44 = [[0] * rank_44 for _ in range(pn)]

    col = 0
    for block_idx in range(4):
        k0 = block_idx * 2
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li * 2 + la]
                    if u_val != 0:
                        U_44[li * m + la][col] = u_val
            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la * 2 + lk]
                    if v_val != 0:
                        V_44[la * p + (k0 + lk)][col] = v_val
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk * 2 + li]
                    if w_val != 0:
                        W_44[(k0 + lk) * n + li][col] = w_val
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    for i in range(n):
        for k in range(p):
            U_44[i * m + 2][col] = 1
            V_44[2 * p + k][col] = 1
            W_44[k * n + i][col] = 1
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    # Verify 44-term
    candidate_44 = {"U": U_44, "V": V_44, "W": W_44}
    if not _validate(candidate_44):
        return _schoolbook()

    best_rank = 44
    best_candidate = candidate_44

    # Attempt to reduce rank by checking linear independence of the 44 rank-1 tensors.
    # If they are dependent, we can eliminate one.
    # Flatten each term to a vector of length 48.
    dim = n * m * p
    T = np.zeros((rank_44, dim))
    for r in range(rank_44):
        idx = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    T[r, idx] = U_44[i * m + a][r] * V_44[a * p + k][r] * W_44[k * n + i][r]
                    idx += 1

    # Check rank of T (44 x 48)
    # We want to see if rank(T) < 44.
    # If so, find a linear combination of rows that is zero.
    # Then we can express one term as a combination of others and eliminate it.
    
    # Use SVD to check rank
    try:
        U_svd, s, Vt = np.linalg.svd(T, full_matrices=False)
        # Count non-zero singular values
        rank_T = np.sum(s > 1e-10)
        if rank_T < rank_44:
            # Find null space of T.T (or T)
            # If T has rank < 44, there is a non-zero vector c such that c.T @ T = 0
            # This means sum c_r * T_r = 0
            # We can eliminate any r with c_r != 0.
            # However, we need to construct the new decomposition.
            # If sum c_r * T_r = 0, then T_r = -1/c_r * sum_{s != r} c_s * T_s.
            # This is not a rank-1 tensor. It's a sum of rank-1 tensors.
            # So we cannot simply eliminate a term and keep the other terms unchanged.
            # We would need to replace T_r with the sum, which increases the rank of that "term".
            # Wait, the rank of the decomposition is the number of terms.
            # If T_r is a linear combination of other terms, it doesn't mean the rank of the decomposition reduces.
            # It means the set of vectors {T_r} is linearly dependent.
            # But the tensor T is the SUM of T_r.
            # If T_r = sum c_s T_s, then T = sum_{s != r} (1+c_s) T_s.
            # This is still a sum of 43 terms?
            # No, T_s are rank-1 tensors.
            # If T_r is a linear combination of other rank-1 tensors, it does NOT imply that the tensor T can be written with fewer rank-1 tensors.
            # Example: T1 = e1, T2 = e2, T3 = e1+e2. T = T1+T2+T3 = 2e1+2e2.
            # Rank of T is 2. Rank of decomposition is 3.
            # But can we write T as sum of 2 rank-1 tensors? Yes, 2e1 + 2e2.
            # So if the T_r are linearly dependent, it MIGHT be possible to reduce the rank, but it requires finding a new set of rank-1 tensors.
            # It is not a simple elimination.
            
            # Therefore, this check is not sufficient to reduce the rank of the decomposition.
            pass
    except:
        pass

    return best_candidate

# EVOLVE-BLOCK-END
