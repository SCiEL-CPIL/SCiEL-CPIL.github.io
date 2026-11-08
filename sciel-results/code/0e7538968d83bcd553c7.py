<<<<<<< SEARCH
    # Attempt rank-40: use rank-20 decomposition for 2x3x4, applied to two halves
    # 2x3x4: Strassen on 2x2 part (rank 14) + schoolbook on 2x1 part (rank 8) = 22
    # Try to find rank-20 for 2x3x4
    def _rank20_2x3x4():
        """Construct rank-20 decomposition for 2x3x4."""
        rank = 20
        U = [[0] * rank for _ in range(2*3)]  # 6 rows
        V = [[0] * rank for _ in range(3*4)]  # 12 rows
        W = [[0] * rank for _ in range(4*2)]  # 8 rows
        col = 0
        # Strassen for 2x2x2 (first 2 cols of B)
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    if S_U[r][li*2+la] != 0:
                        U[li*3+la][col] = S_U[r][li*2+la]
            for la in range(2):
                for lk in range(2):
                    if S_V[r][la*2+lk] != 0:
                        V[la*4+lk][col] = S_V[r][la*2+lk]
            for lk in range(2):
                for li in range(2):
                    if S_W[r][lk*2+li] != 0:
                        W[lk*2+li][col] = S_W[r][lk*2+li]
            col += 1
        # Schoolbook for remaining 2 cols of B with all 3 rows of A
        for a in range(3):
            for i in range(2):
                for k in range(2, 4):
                    U[i*3+a][col] = 1
                    V[a*4+k][col] = 1
                    W[k*2+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
=======
    # Attempt rank-36: use rank-18 decomposition for 2x3x4, applied to two halves
    # 2x3x4 = two 2x3x2 blocks, each with rank 9 (Strassen 2x2x2 rank 7 + 2x1x2 rank 2)
    def _rank18_2x3x4():
        """Construct rank-18 decomposition for 2x3x4."""
        rank = 18
        U = [[0] * rank for _ in range(2*3)]  # 6 rows
        V = [[0] * rank for _ in range(3*4)]  # 12 rows
        W = [[0] * rank for _ in range(4*2)]  # 8 rows
        col = 0
        for half in range(2):
            k0 = 2 * half
            # Strassen for 2x2x2 (rows 0-1 of A, cols k0..k0+1 of B)
            for r in range(7):
                for li in range(2):
                    for la in range(2):
                        if S_U[r][li*2+la] != 0:
                            U[li*3+la][col] = S_U[r][li*2+la]
                for la in range(2):
                    for lk in range(2):
                        if S_V[r][la*2+lk] != 0:
                            V[la*4+k0+lk][col] = S_V[r][la*2+lk]
                for lk in range(2):
                    for li in range(2):
                        if S_W[r][lk*2+li] != 0:
                            W[(k0+lk)*2+li][col] = S_W[r][lk*2+li]
                col += 1
            # Schoolbook for row 2 of A with cols k0..k0+1 of B (2x1x2, rank 2)
            for i in range(2):
                for lk in range(2):
                    U[i*3+2][col] = 1
                    V[2*4+k0+lk][col] = 1
                    W[(k0+lk)*2+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
>>>>>>> REPLACE
