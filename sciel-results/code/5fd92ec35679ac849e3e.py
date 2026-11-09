"""Matrix multiplication 2x4x8 rank-minimal search."""
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    n, m, p = 2, 4, 8
    nm, mp, pn = n*m, m*p, p*n  # 8, 32, 16

    # Build the target tensor
    T = np.zeros((nm, mp, pn), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1

    def verify(U, V, W):
        """Check if U, V, W form a valid decomposition of T."""
        recon = np.einsum('ac,bc,dc->abd', U, V, W)
        return np.array_equal(recon, T)

    # Strassen-based decomposition:
    # M_{2,4,8} -> 8 * M_{2,2,2} (Strassen, rank 7)
    # Total rank = 8 * 7 = 56
    #
    # Structure:
    # A is 2x4, split into A_L (2x2, cols 0-1), A_R (2x2, cols 2-3)
    # B is 4x8, split into 8 blocks of 2x2:
    #   B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:6], B[0:2, 6:8]
    #   B[2:4, 0:2], B[2:4, 2:4], B[2:4, 4:6], B[2:4, 6:8]
    #
    # C = A_L * B_top + A_R * B_bottom
    # Each 2x2 * 2x2 uses one Strassen (rank 7)
    # Total: 8 Strassen instances, rank 56

    # Strassen for 2x2 * 2x2:
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # P1 = (a00+a11)*(b00+b11)
    # P2 = (a10+a11)*b00
    # P3 = a00*(b01-b11)
    # P4 = a11*(b10-b00)
    # P5 = (a00+a01)*b11
    # P6 = (a10-a00)*(b00+b01)
    # P7 = (a01-a11)*(b10+b11)
    #
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6

    # The 7 Strassen products: (A factors, B factors)
    # Each factor is a list of (element_index, coefficient)
    # A element indices: 0=a00, 1=a01, 2=a10, 3=a11
    # B element indices: 0=b00, 1=b01, 2=b10, 3=b11
    strassen_products = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),  # P1
        ([(2, 1), (3, 1)], [(0, 1)]),           # P2
        ([(0, 1)], [(1, 1), (3, -1)]),          # P3
        ([(3, 1)], [(2, 1), (0, -1)]),          # P4
        ([(0, 1), (1, 1)], [(3, 1)]),           # P5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),  # P6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)]),  # P7
    ]

    # C reconstruction: C_elem -> [(product_idx, coeff), ...]
    c_reconstruction = [
        [(0, 1), (3, 1), (4, -1), (6, 1)],  # C00
        [(2, 1), (4, 1)],                   # C01
        [(1, 1), (3, 1)],                   # C10
        [(0, 1), (1, -1), (2, 1), (5, 1)],  # C11
    ]

    rank = 56
    U = np.zeros((nm, rank), dtype=np.int64)
    V = np.zeros((mp, rank), dtype=np.int64)
    W = np.zeros((pn, rank), dtype=np.int64)

    # 8 Strassen instances:
    # (a_col_off, b_row_off, b_col_off)
    # a_col_off: 0 (A_L) or 2 (A_R)
    # b_row_off: 0 (top 2 rows) or 2 (bottom 2 rows)
    # b_col_off: 0, 2, 4, 6 (column offset in B)
    instances = [
        (0, 0, 0), (0, 0, 2), (0, 0, 4), (0, 0, 6),  # A_L * B_top
        (2, 2, 0), (2, 2, 2), (2, 2, 4), (2, 2, 6),  # A_R * B_bottom
    ]

    for inst_idx, (a_col_off, b_row_off, b_col_off) in enumerate(instances):
        base_col = inst_idx * 7  # starting column in U, V, W

        for prod_idx in range(7):
            col = base_col + prod_idx
            a_factors, b_factors = strassen_products[prod_idx]

            # Fill U (A factors)
            for elem_idx, coeff in a_factors:
                a_row, a_col = divmod(elem_idx, 2)  # a00=(0,0), a01=(0,1), a10=(1,0), a11=(1,1)
                # A element (a_row, a_col) is at position (a_row * m + (a_col_off + a_col)) in U
                u_pos = a_row * m + (a_col_off + a_col)
                U[u_pos, col] += coeff

            # Fill V (B factors)
            for elem_idx, coeff in b_factors:
                b_row, b_col = divmod(elem_idx, 2)  # b00=(0,0), b01=(0,1), b10=(1,0), b11=(1,1)
                # B element (b_row, b_col) is at position ((b_row_off + b_row) * p + (b_col_off + b_col)) in V
                v_pos = (b_row_off + b_row) * p + (b_col_off + b_col)
                V[v_pos, col] += coeff

        # Fill W (C reconstruction)
        # C element (c_row, c_col) maps to W position (c_col * n + c_row)
        # c_row is 0 or 1 (the two rows of the 2x2 result)
        # c_col is 0 or 1 (the two columns of the 2x2 result)
        # The global column in C is b_col_off + c_col
        for c_elem_idx in range(4):
            c_row, c_col = divmod(c_elem_idx, 2)
            w_pos = (b_col_off + c_col) * n + c_row
            for prod_idx, coeff in c_reconstruction[c_elem_idx]:
                col = base_col + prod_idx
                W[w_pos, col] += coeff

    # Verify the Strassen decomposition
    if not verify(U, V, W):
        # Fallback to schoolbook if Strassen construction has a bug
        rank = nm * mp * pn // (n * m * p) * (n * m * p)  # just use 64
        rank = 64
        U = np.zeros((nm, rank), dtype=np.int64)
        V = np.zeros((mp, rank), dtype=np.int64)
        W = np.zeros((pn, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
    else:
        best_rank = rank

    # Try to find lower rank decompositions using random search
    import random
    rng = random.Random(42)

    # Try random search for ranks below 56
    for target_rank in range(55, 49, -1):
        if time.time() > deadline:
            break

        for attempt in range(30):
            if time.time() > deadline:
                break

            U_rand = np.zeros((nm, target_rank), dtype=np.int64)
            V_rand = np.zeros((mp, target_rank), dtype=np.int64)
            W_rand = np.zeros((pn, target_rank), dtype=np.int64)

            for c in range(target_rank):
                nz_u = rng.sample(range(nm), min(4, nm))
                nz_v = rng.sample(range(mp), min(6, mp))
                nz_w = rng.sample(range(pn), min(4, pn))

                for a in nz_u:
                    U_rand[a, c] = rng.choice([-1, 1])
                for b in nz_v:
                    V_rand[b, c] = rng.choice([-1, 1])
                for d in nz_w:
                    W_rand[d, c] = rng.choice([-1, 1])

            if verify(U_rand, V_rand, W_rand):
                U, V, W = U_rand, V_rand, W_rand
                rank = target_rank
                break

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
