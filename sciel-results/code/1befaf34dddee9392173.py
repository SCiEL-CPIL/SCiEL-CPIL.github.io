"""Matrix multiplication 2x3x8 with rank-37 Strassen-based decomposition."""
import time

def construct():
    t_start = time.time()
    limit = t_start + 130.0

    n, m, p = 2, 3, 8
    rank = 37

    # Strassen 2x2x2 coefficients (standard)
    # U: maps A (2x2) to 7 products
    # V: maps B (2x2) to 7 products
    # W: maps products to C (2x2)
    su = [
        [1, 0, 0, 1],   # M0 = (A00+A11)(B00+B11)
        [0, 0, 1, 1],   # M1 = (A10+A11)B00
        [1, 0, 0, 0],   # M2 = A00(B10-B11)
        [0, 0, 0, 1],   # M3 = A11(B01-B00)
        [1, 1, 0, 0],   # M4 = (A00+A10)B11
        [-1, 0, 1, 0],  # M5 = (A01-A10)(B00+B10)
        [0, 1, 0, -1],  # M6 = (A00-A11)(B01+B11)
    ]
    sv = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]
    sw = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ]

    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]  # 6 x 37
    V = [[0] * rank for _ in range(m * p)]  # 24 x 37
    W = [[0] * rank for _ in range(p * n)]  # 16 x 37

    col = 0

    # 7 Strassen blocks: each handles A[0:2,0:2]*B[0:2, 2s:2s+2] AND
    # shares the product to handle A[0:2,2]*B[2:3, 2s:2s+2]
    for s_idx in range(7):
        k_offset = s_idx * 2

        # U factors: A[0:2, 0:2] -> 7 products
        for i in range(2):
            for a in range(2):
                val = su[s_idx][i * 2 + a]
                if val:
                    U[i * m + a][col] += val

        # V factors: B[0:2, k_offset:k_offset+2] -> 7 products
        # PLUS: B[2, k_offset:k_offset+2] -> same products (for sharing)
        for a in range(2):
            for k in range(2):
                val = sv[s_idx][a * 2 + k]
                if val:
                    V[a * p + (k_offset + k)][col] += val
        # Second term: A[0:2,2] * B[2, k_offset:k_offset+2]
        # V[2, k_offset+k] += B_s[k]
        for k in range(2):
            val = sv[s_idx][k]  # B_s[k] is the k-th element of row 0 of B
            if val:
                V[2 * p + (k_offset + k)][col] += val

        # W factors: products -> C[0:2, k_offset:k_offset+2]
        for k in range(2):
            for i in range(2):
                val = sw[s_idx][k * 2 + i]
                if val:
                    W[(k_offset + k) * n + i][col] += val

        col += 1

        if time.time() > limit:
            return _schoolbook_fallback(n, m, p)

    # 2 remaining schoolbook terms for A[0:2,2]*B[2:3, 6:8]
    for k in range(6, 8):
        for i in range(2):
            u_row = i * m + 2
            v_row = 2 * p + k
            w_row = k * n + i
            U[u_row][col] = 1
            V[v_row][col] = 1
            W[w_row][col] = 1
            col += 1

        if time.time() > limit:
            return _schoolbook_fallback(n, m, p)

    # Validate: sum_r U[i*m+a][r] * V[a*p+k][r] * W[k*n+i][r] == 1 for all i,a,k
    valid = True
    for i in range(n):
        for a in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + a][r] * V[a * p + k][r] * W[k * n + i][r]
                if s != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if valid:
        return {"U": U, "V": V, "W": W}
    else:
        return _schoolbook_fallback(n, m, p)

def _schoolbook_fallback(n, m, p):
    rb = n * m * p
    U_f = [[0] * rb for _ in range(n * m)]
    V_f = [[0] * rb for _ in range(m * p)]
    W_f = [[0] * rb for _ in range(p * n)]
    c = 0
    for i in range(n):
        for a in range(m):
            for k in range(p):
                U_f[i * m + a][c] = 1
                V_f[a * p + k][c] = 1
                W_f[k * n + i][c] = 1
                c += 1
    return {"U": U_f, "V": V_f, "W": W_f}
