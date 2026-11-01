import time

DIMENSIONS = [2, 2, 11]

def construct():
    start = time.time()
    deadline = start + 140

    n, m, p = DIMENSIONS

    def build_rank39():
        rank = 5 * 7 + 1 * 4
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        r = 0
        for block in range(5):
            k0 = block * 2
            k1 = block * 2 + 1
            v_e = k0; v_f = k1; v_g = 11 + k0; v_h = 11 + k1
            w_c00 = 2 * k0; w_c10 = 2 * k0 + 1
            w_c01 = 2 * k1; w_c11 = 2 * k1 + 1

            U[0][r] = 1; U[3][r] = 1
            V[v_e][r] = 1; V[v_h][r] = 1
            W[w_c00][r] = 1; W[w_c11][r] = 1
            r += 1

            U[2][r] = 1; U[3][r] = 1
            V[v_e][r] = 1
            W[w_c10][r] = 1; W[w_c11][r] = -1
            r += 1

            U[0][r] = 1
            V[v_f][r] = 1; V[v_h][r] = -1
            W[w_c01][r] = 1; W[w_c11][r] = 1
            r += 1

            U[3][r] = 1
            V[v_g][r] = 1; V[v_e][r] = -1
            W[w_c00][r] = 1; W[w_c10][r] = 1
            r += 1

            U[0][r] = 1; U[1][r] = 1
            V[v_h][r] = 1
            W[w_c00][r] = -1; W[w_c01][r] = 1
            r += 1

            U[2][r] = 1; U[0][r] = -1
            V[v_e][r] = 1; V[v_f][r] = 1
            W[w_c11][r] = 1
            r += 1

            U[1][r] = 1; U[3][r] = -1
            V[v_g][r] = 1; V[v_h][r] = 1
            W[w_c00][r] = 1
            r += 1

        k2 = 10
        v_e2 = k2; v_g2 = 11 + k2
        w_c02 = 2 * k2; w_c12 = 2 * k2 + 1
        U[0][r] = 1; V[v_e2][r] = 1; W[w_c02][r] = 1; r += 1
        U[1][r] = 1; V[v_g2][r] = 1; W[w_c02][r] = 1; r += 1
        U[2][r] = 1; V[v_e2][r] = 1; W[w_c12][r] = 1; r += 1
        U[3][r] = 1; V[v_g2][r] = 1; W[w_c12][r] = 1; r += 1
        return {"U": U, "V": V, "W": W}

    def is_valid(U, V, W):
        rank = len(U[0])
        target = {}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target[(i * m + j, j * p + k, k * n + i)] = 1
        Uv = [dict() for _ in range(rank)]
        Vv = [dict() for _ in range(rank)]
        Wv = [dict() for _ in range(rank)]
        for r in range(rank):
            for a in range(n * m):
                if U[a][r] != 0:
                    Uv[r][a] = U[a][r]
            for b in range(m * p):
                if V[b][r] != 0:
                    Vv[r][b] = V[b][r]
            for c in range(p * n):
                if W[c][r] != 0:
                    Wv[r][c] = W[c][r]
        acc = {}
        for r in range(rank):
            if not Uv[r] or not Vv[r] or not Wv[r]:
                continue
            for a, ua in Uv[r].items():
                for b, vb in Vv[r].items():
                    ab = ua * vb
                    for c, wc in Wv[r].items():
                        key = (a, b, c)
                        acc[key] = acc.get(key, 0) + ab * wc
        for key, val in target.items():
            if acc.get(key, 0) != val:
                return False
        for key, val in acc.items():
            if val != 0 and key not in target:
                return False
        return True

    # Try a structured rank-38 search using a restricted dictionary of
    # Strassen-like patterns plus a few variations. If found, return rank 38.
    # Otherwise, fall back to rank 39.
    if time.time() < deadline - 50:
        # We attempt to find a rank-38 decomposition by trying to cover
        # 5 Strassen pairs (35 terms) + 3 extra terms for the last column.
        # This is structurally impossible (need 4 terms for 1 column),
        # but we try a few heuristic variations to be thorough.
        #
        # Actually, as reasoned in the previous version, rank 38 is not
        # achievable with this family. We skip the search to save time
        # and return the verified rank-39 solution.
        pass

    result = build_rank39()
    # Verify the result (fast, < 1s)
    if not is_valid(result["U"], result["V"], result["W"]):
        raise ValueError("Rank-39 construction failed verification")
    return result

BET: {mechanism: "Cleaned up code with explicit verification and removed dead search logic, ensuring reliable instant return of the verified rank-39 solution without timeout risk.", q: 0.95, delta: [0, 0]}
