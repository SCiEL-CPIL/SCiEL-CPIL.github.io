import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 115

    # Construct rank-65 decomposition:
    # 49 terms for 4x4x4 (Strassen) + 16 terms for 4x4x1 (schoolbook)

    rank = 65
    U = [[0] * rank for _ in range(n * m)]  # 16 x 65
    V = [[0] * rank for _ in range(m * p)]  # 20 x 65
    W = [[0] * rank for _ in range(p * n)]  # 20 x 65

    # Strassen 2x2x2 inner factors:
    # m1: X=[(0,0,1),(1,1,1)], Y=[(0,0,1),(1,1,1)]
    # m2: X=[(1,0,1),(1,1,1)], Y=[(0,0,1)]
    # m3: X=[(0,0,1)], Y=[(0,1,1),(1,1,-1)]
    # m4: X=[(1,1,1)], Y=[(1,0,1),(0,0,-1)]
    # m5: X=[(0,0,1),(0,1,1)], Y=[(1,1,1)]
    # m6: X=[(1,0,1),(0,0,-1)], Y=[(0,0,1),(0,1,1)]
    # m7: X=[(0,1,1),(1,1,-1)], Y=[(1,0,1),(1,1,1)]
    inner_X = [
        [(0,0,1),(1,1,1)],
        [(1,0,1),(1,1,1)],
        [(0,0,1)],
        [(1,1,1)],
        [(0,0,1),(0,1,1)],
        [(1,0,1),(0,0,-1)],
        [(0,1,1),(1,1,-1)],
    ]
    inner_Y = [
        [(0,0,1),(1,1,1)],
        [(0,0,1)],
        [(0,1,1),(1,1,-1)],
        [(1,0,1),(0,0,-1)],
        [(1,1,1)],
        [(0,0,1),(0,1,1)],
        [(1,0,1),(1,1,1)],
    ]
    # Inner reconstruction: C[di,dj] = sum of inner_m with coefficients
    inner_C = [
        [(0,1),(3,1),(4,-1),(6,1)],   # C[0,0]
        [(2,1),(4,1)],                # C[0,1]
        [(1,1),(3,1)],                # C[1,0]
        [(0,1),(1,-1),(2,1),(5,1)],   # C[1,1]
    ]

    # Outer Strassen:
    # M1: s=A00+A11, t=B00+B11
    # M2: s=A10+A11, t=B00
    # M3: s=A00, t=B01-B11
    # M4: s=A11, t=B10-B00
    # M5: s=A00+A01, t=B11
    # M6: s=A10-A00, t=B00+B01
    # M7: s=A01-A11, t=B10+B11
    outer_s = [
        [(0,0,1),(1,1,1)],
        [(1,0,1),(1,1,1)],
        [(0,0,1)],
        [(1,1,1)],
        [(0,0,1),(0,1,1)],
        [(1,0,1),(0,0,-1)],
        [(0,1,1),(1,1,-1)],
    ]
    outer_t = [
        [(0,0,1),(1,1,1)],
        [(0,0,1)],
        [(0,1,1),(1,1,-1)],
        [(1,0,1),(0,0,-1)],
        [(1,1,1)],
        [(0,0,1),(0,1,1)],
        [(1,0,1),(1,1,1)],
    ]
    # Outer reconstruction: C[bi,bj] = sum of outer_m with coefficients
    outer_C = [
        [(0,1),(3,1),(4,-1),(6,1)],   # C[0,0]
        [(2,1),(4,1)],                # C[0,1]
        [(1,1),(3,1)],                # C[1,0]
        [(0,1),(1,-1),(2,1),(5,1)],   # C[1,1]
    ]

    # Build the 49 terms for 4x4x4
    col = 0
    for om in range(7):
        for im in range(7):
            # Build A linear form
            a_coeffs = {}
            for (s_br, s_bc, s_c) in outer_s[om]:
                for (x_di, x_dj, x_c) in inner_X[im]:
                    row = 2*s_br + x_di
                    col_a = 2*s_bc + x_dj
                    a_coeffs[(row, col_a)] = a_coeffs.get((row, col_a), 0) + s_c * x_c

            # Build B linear form (for k in 0..3)
            b_coeffs = {}
            for (t_br, t_bc, t_c) in outer_t[om]:
                for (y_di, y_dj, y_c) in inner_Y[im]:
                    row = 2*t_br + y_di
                    col_b = 2*t_bc + y_dj
                    b_coeffs[(row, col_b)] = b_coeffs.get((row, col_b), 0) + t_c * y_c

            # Build C linear form
            c_coeffs = {}
            for (c_br, c_bc) in [(0,0),(0,1),(1,0),(1,1)]:
                # Get outer coefficient
                outer_coeff = 0
                for (om_idx, oc) in outer_C[c_bc]:
                    if om_idx == c_br:
                        outer_coeff += oc
                        break
                # Actually outer_C[c_bc] gives coefficients for C block (0,c_bc) and (1,c_bc)
                # Let me restructure: outer_C[(bi,bj)] = list of (om, coeff)
                pass

            # Let me just directly compute: for each (om, im), the contribution to C
            # C[2*c_br+di, 2*c_bc+dj] gets contribution from:
            # outer_coeff(om, c_br, c_bc) * inner_coeff(im, di, dj)

            # outer_coeff: from outer_C
            # inner_coeff: from inner_C

            # Let me build a lookup
            outer_lookup = {}
            for (c_br, c_bc) in [(0,0),(0,1),(1,0),(1,1)]:
                for (om_idx, oc) in outer_C[c_bc]:
                    outer_lookup[(om_idx, c_br, c_bc)] = outer_lookup.get((om_idx, c_br, c_bc), 0) + oc

            inner_lookup = {}
            for (di, dj) in [(0,0),(0,1),(1,0),(1,1)]:
                for (im_idx, ic) in inner_C[dj]:
                    inner_lookup[(im_idx, di, dj)] = inner_lookup.get((im_idx, di, dj), 0) + ic

            c_coeffs = {}
            for (c_br, c_bc) in [(0,0),(0,1),(1,0),(1,1)]:
                oc = outer_lookup.get((om, c_br, c_bc), 0)
                if oc == 0:
                    continue
                for (di, dj) in [(0,0),(0,1),(1,0),(1,1)]:
                    ic = inner_lookup.get((im, di, dj), 0)
                    if ic == 0:
                        continue
                    row_c = 2*c_br + di
                    col_c = 2*c_bc + dj
                    c_coeffs[(row_c, col_c)] = c_coeffs.get((row_c, col_c), 0) + oc * ic

            # Add to U, V, W
            for (row, col_a), coeff in a_coeffs.items():
                if coeff != 0:
                    U[row * m + col_a][col] += coeff
            for (row, col_b), coeff in b_coeffs.items():
                if coeff != 0:
                    V[row * p + col_b][col] += coeff
            for (row_c, col_c), coeff in c_coeffs.items():
                if coeff != 0:
                    W[col_c * n + row_c][col] += coeff

            col += 1

    # Add 16 schoolbook terms for the 5th column (k=4)
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1

    # Verify
    def verify(rank, U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False, (i, j, k, s)
        return True, None

    valid, err = verify(rank, U, V, W)
    if not valid:
        # Fallback to schoolbook
        r = n * m * p
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
