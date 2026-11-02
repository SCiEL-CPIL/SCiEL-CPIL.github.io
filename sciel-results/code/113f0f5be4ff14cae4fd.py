import time

def construct():
    start = time.time()
    deadline = start + 100

    n, m, p = 4, 4, 4
    rank = 49

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Top-level Strassen coefficients for X_k and Y_k
    # X_k[br, bc] is coeff of A_block(br, bc)
    # Y_k[br, bc] is coeff of B_block(br, bc)
    # br, bc in {0, 1}
    
    top_coeffs_X = [
        [[1, 0], [0, 1]], # k=0: A00 + A11
        [[0, 0], [1, 1]], # k=1: A10 + A11
        [[1, 0], [0, 0]], # k=2: A00
        [[0, 0], [0, 1]], # k=3: A11
        [[1, 1], [0, 0]], # k=4: A00 + A01
        [[-1, 0], [1, 0]], # k=5: A10 - A00
        [[0, 1], [0, -1]], # k=6: A01 - A11
    ]
    
    top_coeffs_Y = [
        [[1, 0], [0, 1]], # k=0: B00 + B11
        [[1, 0], [0, 0]], # k=1: B00
        [[0, 1], [0, -1]], # k=2: B01 - B11
        [[0, -1], [1, 0]], # k=3: B10 - B00
        [[0, 0], [0, 1]], # k=4: B11
        [[1, 1], [0, 0]], # k=5: B00 + B01
        [[0, 1], [1, 1]], # k=6: B10 + B11
    ]
    
    # Bottom-level Strassen forms
    # Each form j defines:
    # u_vec: indices into X (0:00, 1:01, 2:10, 3:11) with coeffs
    # v_vec: indices into Y (0:00, 1:01, 2:10, 3:11) with coeffs
    # w_mat: 2x2 matrix of coeffs for M reconstruction
    # M[0,0] = ...
    # M[0,1] = ...
    # M[1,0] = ...
    # M[1,1] = ...
    
    bottom_forms = [
        { # j=0
            'u': [(0, 1), (3, 1)],
            'v': [(0, 1), (3, 1)],
            'w': [[1, 0], [0, 1]] # M00 += 1, M11 += 1
        },
        { # j=1
            'u': [(2, 1), (3, 1)],
            'v': [(0, 1)],
            'w': [[0, 0], [1, -1]] # M10 += 1, M11 -= 1
        },
        { # j=2
            'u': [(0, 1)],
            'v': [(1, 1), (3, -1)],
            'w': [[0, 1], [0, 1]] # M01 += 1, M11 += 1
        },
        { # j=3
            'u': [(3, 1)],
            'v': [(2, 1), (0, -1)],
            'w': [[1, 1], [0, 0]] # M00 += 1, M01 += 1? 
            # Wait, M00 += t3, M10 += t3.
            # My w matrix is [row, col].
            # M00 is [0,0]. M10 is [1,0].
            # So w[0,0]=1, w[1,0]=1.
        },
        { # j=4
            'u': [(0, 1), (1, 1)],
            'v': [(3, 1)],
            'w': [[-1, 0], [0, 1]] # M00 -= 1, M01 += 1
        },
        { # j=5
            'u': [(2, 1), (0, -1)],
            'v': [(0, 1), (1, 1)],
            'w': [[0, 0], [0, 1]] # M11 += 1
        },
        { # j=6
            'u': [(1, 1), (3, -1)],
            'v': [(2, 1), (3, 1)],
            'w': [[1, 0], [0, 0]] # M00 += 1
        },
    ]
    
    # Fix bottom_forms w matrices based on standard Strassen:
    # M00 = t0 + t3 - t4 + t6
    # M01 = t2 + t4
    # M10 = t1 + t3
    # M11 = t0 - t1 + t2 + t5
    
    bottom_forms[0]['w'] = [[1, 0], [0, 1]]
    bottom_forms[1]['w'] = [[0, 0], [1, -1]]
    bottom_forms[2]['w'] = [[0, 1], [0, 1]]
    bottom_forms[3]['w'] = [[1, 0], [1, 0]]
    bottom_forms[4]['w'] = [[-1, 1], [0, 0]]
    bottom_forms[5]['w'] = [[0, 0], [0, 1]]
    bottom_forms[6]['w'] = [[1, 0], [0, 0]]

    # Top-level reconstruction coefficients
    # C_br_bc = sum_k d_k_br_bc M_k
    top_recon = [
        [[1, 0], [0, 0]], # C00: M0
        [[0, 1], [0, 0]], # C01: M2
        [[0, 0], [1, 0]], # C10: M1
        [[0, 0], [0, 1]], # C11: M0
    ]
    # Wait, C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # d[k][br][bc] is coeff of M_k in C_br_bc
    d = [
        [[0, 0], [0, 0]],
        [[0, 0], [0, 0]],
        [[0, 0], [0, 0]],
        [[0, 0], [0, 0]],
        [[0, 0], [0, 0]],
        [[0, 0], [0, 0]],
        [[0, 0], [0, 0]],
    ]
    
    # k=0
    d[0][0][0] += 1
    d[0][1][1] += 1
    # k=1
    d[1][1][0] += 1
    d[1][1][1] -= 1
    # k=2
    d[2][0][1] += 1
    d[2][1][1] += 1
    # k=3
    d[3][0][0] += 1
    d[3][1][0] += 1
    # k=4
    d[4][0][0] -= 1
    d[4][0][1] += 1
    # k=5
    d[5][1][1] += 1
    # k=6
    d[6][0][0] += 1

    # Helper to map block/local to global index
    def get_A_idx(br, bc, lr, lc):
        return (2 * br + lr) * 4 + (2 * bc + lc)
    
    def get_B_idx(br, bc, lr, lc):
        return (2 * br + lr) * 4 + (2 * bc + lc)
    
    def get_C_idx(br, bc, lr, lc):
        # k = 4 * col + row
        col = 2 * bc + lc
        row = 2 * br + lr
        return 4 * col + row

    # Pre-compute linear forms for X_k and Y_k
    # X_k[br, bc] is a vector of length 16
    X_vecs = []
    Y_vecs = []
    for k in range(7):
        Xk = [[None for _ in range(2)] for _ in range(2)]
        Yk = [[None for _ in range(2)] for _ in range(2)]
        for br in range(2):
            for bc in range(2):
                cx = top_coeffs_X[k][br][bc]
                cy = top_coeffs_Y[k][br][bc]
                vx = [0] * 16
                vy = [0] * 16
                if cx != 0:
                    for lr in range(2):
                        for lc in range(2):
                            idx = get_A_idx(br, bc, lr, lc)
                            vx[idx] = cx
                if cy != 0:
                    for lr in range(2):
                        for lc in range(2):
                            idx = get_B_idx(br, bc, lr, lc)
                            vy[idx] = cy
                Xk[br][bc] = vx
                Yk[br][bc] = vy
        X_vecs.append(Xk)
        Y_vecs.append(Yk)

    # Construct U, V, W
    for k in range(7):
        for j in range(7):
            r = k * 7 + j
            form = bottom_forms[j]
            
            # Compute U vector for this r
            u_vec = [0] * 16
            for (idx, coeff) in form['u']:
                # idx 0:00, 1:01, 2:10, 3:11
                br, bc = divmod(idx, 2)
                # Add coeff * Xk[br][bc]
                for i in range(16):
                    u_vec[i] += coeff * X_vecs[k][br][bc][i]
            
            # Compute V vector for this r
            v_vec = [0] * 16
            for (idx, coeff) in form['v']:
                br, bc = divmod(idx, 2)
                for i in range(16):
                    v_vec[i] += coeff * Y_vecs[k][br][bc][i]
            
            # Fill U and V
            for i in range(16):
                U[i][r] = u_vec[i]
                V[i][r] = v_vec[i]
            
            # Compute W contributions
            # w_mat[j] is 2x2 coeffs for M_k entries
            w_mat = form['w']
            for lr in range(2):
                for lc in range(2):
                    c_coeff = w_mat[lr][lc]
                    if c_coeff == 0:
                        continue
                    # M_k[lr, lc] contributes c_coeff * t_j
                    # C_br_bc gets d[k][br][bc] * M_k
                    # So C_br_bc[lr, lc] gets d[k][br][bc] * c_coeff * t_j
                    for br in range(2):
                        for bc in range(2):
                            d_coeff = d[k][br][bc]
                            if d_coeff == 0:
                                continue
                            total_coeff = d_coeff * c_coeff
                            if total_coeff == 0:
                                continue
                            c_idx = get_C_idx(br, bc, lr, lc)
                            W[c_idx][r] += total_coeff

    return {"U": U, "V": V, "W": W}
