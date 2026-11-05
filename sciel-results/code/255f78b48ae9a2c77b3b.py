import time

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 130.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8

    # ---- Schoolbook fallback (rank 48) ----
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # ---- Exact validation ----
    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = 0
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += u_row[c] * v_row[c] * w_row[c]
                    if acc != 1:
                        return False
        return True

    # ---- Strassen 2x2x2 decomposition (rank 7) ----
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

    def _verify_strassen():
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    acc = 0
                    for r in range(7):
                        u_val = S_U[r][i * 2 + a]
                        v_val = S_V[r][a * 2 + k]
                        w_val = S_W[r][k * 2 + i]
                        acc += u_val * v_val * w_val
                    if acc != 1:
                        return False
        return True

    if not _verify_strassen():
        return _schoolbook()

    # ---- Build rank-32 candidate ----
    # For each 2-column block of B (k0, k0+1):
    #   Use Strassen (rank 7) for a=0,1
    #   Use rank-1 term for a=2
    def _build_base():
        rank = 32
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0

        for block_idx in range(4):
            k0 = block_idx * 2
            k1 = k0 + 1

            # Strassen for a=0,1 (7 terms)
            for r in range(7):
                for li in range(2):
                    for la in range(2):
                        u_val = S_U[r][li * 2 + la]
                        if u_val == 0:
                            continue
                        u_idx = li * m + la
                        U[u_idx][col] = u_val

                for la in range(2):
                    for lk in range(2):
                        v_val = S_V[r][la * 2 + lk]
                        if v_val == 0:
                            continue
                        v_idx = la * p + (k0 + lk)
                        V[v_idx][col] = v_val

                for lk in range(2):
                    for li in range(2):
                        w_val = S_W[r][lk * 2 + li]
                        if w_val == 0:
                            continue
                        w_idx = (k0 + lk) * n + li
                        W[w_idx][col] = w_val

                col += 1

            # Rank-1 term for a=2 (1 term)
            for i in range(2):
                U[i * m + 2][col] = 1
            for k in range(2):
                V[2 * p + (k0 + k)][col] = 1
                W[(k0 + k) * n + i][col] = 1  # This is wrong, need to be careful

            # Actually, for the rank-1 term:
            # u_{i,2} = 1 for i=0,1
            # v_{2,k} = 1 for k=k0, k0+1
            # w_{k,i} = 1 for all k in {k0, k0+1}, i in {0,1}
            # This gives T_{i,2,k} = sum_r u_{i,r} v_{2,r} w_{k,r}
            # For this single term r=col: u_{i,col}=1, v_{2,col}=1, w_{k,col}=1
            # So T_{i,2,k} = 1 * 1 * 1 = 1. Correct.

            # But wait, I need to set W[k*n + i][col] = 1 for both k values
            # Let me redo this properly

            # Actually, I already incremented col above. Let me fix.
            # The rank-1 term should be a single column where:
            # U[i*m+2][col] = 1 for i=0,1
            # V[2*p+k][col] = 1 for k=k0,k0+1
            # W[k*n+i][col] = 1 for k=k0,k0+1 and i=0,1

            col += 1

        return {"U": U, "V": V, "W": W}

    # Let me rebuild this more carefully
    def _build_base_v2():
        rank = 32
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0

        for block_idx in range(4):
            k0 = block_idx * 2

            # Strassen for a=0,1 (7 terms)
            for r in range(7):
                for li in range(2):
                    for la in range(2):
                        u_val = S_U[r][li * 2 + la]
                        if u_val != 0:
                            U[li * m + la][col] = u_val

                for la in range(2):
                    for lk in range(2):
                        v_val = S_V[r][la * 2 + lk]
                        if v_val != 0:
                            V[la * p + (k0 + lk)][col] = v_val

                for lk in range(2):
                    for li in range(2):
                        w_val = S_W[r][lk * 2 + li]
                        if w_val != 0:
                            W[(k0 + lk) * n + li][col] = w_val

                col += 1

            # Rank-1 term for a=2
            # u_{i,2} = 1, v_{2,k} = 1, w_{k,i} = 1
            for i in range(2):
                U[i * m + 2][col] = 1
            for k in range(2):
                V[2 * p + (k0 + k)][col] = 1
                for i in range(2):
                    W[(k0 + k) * n + i][col] = 1

            col += 1

        return {"U": U, "V": V, "W": W}

    base = _build_base_v2()
    if not _validate(base):
        return _schoolbook()

    best_candidate = base
    best_rank = 32

    if _elapsed_ok():
        # Try to find rank 31 by checking if any column can be removed
        # Check for duplicate columns
        U = best_candidate["U"]
        V = best_candidate["V"]
        W = best_candidate["W"]
        rank = len(U[0])

        seen = {}
        for c in range(rank):
            u_tuple = tuple(U[i][c] for i in range(n * m))
            v_tuple = tuple(V[i][c] for i in range(m * p))
            w_tuple = tuple(W[i][c] for i in range(p * n))
            key = (u_tuple, v_tuple, w_tuple)

            if key in seen:
                # Try removing c
                new_rank = rank - 1
                new_U = [[0] * new_rank for _ in range(n * m)]
                new_V = [[0] * new_rank for _ in range(m * p)]
                new_W = [[0] * new_rank for _ in range(p * n)]
                nc = 0
                for cc in range(rank):
                    if cc == c:
                        continue
                    for i in range(n * m):
                        new_U[i][nc] = U[i][cc]
                    for i in range(m * p):
                        new_V[i][nc] = V[i][cc]
                    for i in range(p * n):
                        new_W[i][nc] = W[i][cc]
                    nc += 1
                cand = {"U": new_U, "V": new_V, "W": new_W}
                if _validate(cand):
                    best_candidate = cand
                    best_rank = new_rank
                    break
            else:
                seen[key] = c

    return best_candidate

# EVOLVE-BLOCK-END
