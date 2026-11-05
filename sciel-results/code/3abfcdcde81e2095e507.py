"""Matrix multiplication 6x6x6 - Strassen block decomposition (2x2 blocks of 3x3)."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Block-structured decomposition for 6x6 matrix multiplication.
    Partition 6x6 into 2x2 grid of 3x3 blocks.
    Apply Strassen (rank 7) to the 2x2 block matrix multiplication.
    For each of the 7 terms, the 3x3 matrix products use schoolbook (rank 27).
    Total rank = 7 * 27 = 189.
    W follows convention W[k*n+i, r] where (k,i) is (col_C, row_C).
    """
    n, m, p = DIMENSIONS  # 6, 6, 6
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    bs = 3       # block size
    nb = 2       # number of blocks per dimension

    # Strassen 2x2 decomposition coefficients
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # 7 products:
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b10-b11)
    # M4 = a11*(b01-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01+a11)*(b10+b11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # strassen_A[t][entry]: coefficient for A entry in term t
    # A entries: 0=a00, 1=a01, 2=a10, 3=a11
    s2_A = [
        [1, 0, 0, 1],   # M1: a00+a11
        [0, 0, 1, 1],   # M2: a10+a11
        [1, 0, 0, 0],   # M3: a00
        [0, 0, 0, 1],   # M4: a11
        [1, 1, 0, 0],   # M5: a00+a01
        [-1, 0, 1, 0],  # M6: a10-a00
        [0, 1, 0, 1],   # M7: a01+a11
    ]
    s2_B = [
        [1, 0, 0, 1],   # M1: b00+b11
        [1, 0, 0, 0],   # M2: b00
        [0, 0, 1, -1],  # M3: b10-b11
        [-1, 1, 0, 0],  # M4: b01-b00
        [0, 0, 0, 1],   # M5: b11
        [1, 1, 0, 0],   # M6: b00+b01
        [0, 0, 1, 1],   # M7: b10+b11
    ]
    s2_C = [
        [1, 0, 0, 0],   # M1: c00
        [0, 0, 0, 1],   # M2: c10
        [0, 1, 0, 0],   # M3: c01
        [1, 0, 1, 0],   # M4: c00+c10
        [-1, 1, 0, 0],  # M5: -c00+c01
        [0, 0, 0, 1],   # M6: c11
        [1, 0, 0, 1],   # M7: c00+c11
    ]

    total_rank = 7 * 27  # 189

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    col = 0
    for t_outer in range(7):
        a_coeffs = s2_A[t_outer]
        b_coeffs = s2_B[t_outer]
        c_coeffs = s2_C[t_outer]

        # For the 3x3 schoolbook decomposition:
        # For each (i, j, k) in 0..2:
        #   U row (i, k) gets 1, V row (k, j) gets 1, W row (j, i) gets 1
        # But we need to apply the outer Strassen coefficients to the blocks.
        
        for i in range(nb):
            a_block_coeff = a_coeffs[i]
            if a_block_coeff == 0:
                continue
            for k in range(nb):
                b_block_coeff = b_coeffs[k]
                if b_block_coeff == 0:
                    continue
                for j in range(nb):
                    c_block_coeff = c_coeffs[j]
                    if c_block_coeff == 0:
                        continue
                    
                    # For each entry in the 3x3 blocks
                    for ri in range(bs):
                        for rk in range(bs):
                            # A block (i, k): global row = i*bs + ri, global col = k*bs + rk
                            # U index: (i*bs + ri) * m + (k*bs + rk)
                            u_row = (i * bs + ri) * m + (k * bs + rk)
                            U[u_row][col] = a_block_coeff * 1.0
                        
                        for rj in range(bs):
                            # B block (k, j): global row = k*bs + rj, global col = j*bs + rj
                            # Wait, B is 6x6. B[k,j] block is at row k*bs, col j*bs.
                            # V index: (k*bs + rj) * p + (j*bs + rk) ? No.
                            # V has shape (m*p, rank) = (36, rank).
                            # V row corresponds to B entry. B is 6x6.
                            # B[row, col].
                            # The term in matrix mult is A[i, k] * B[k, j].
                            # So for a specific scalar product A[i, k] * B[k, j], 
                            # U is indexed by (i, k), V by (k, j).
                            # So for the block (i, k) of A and (k, j) of B:
                            # A entry is at (i*bs + ri, k*bs + rk).
                            # B entry is at (k*bs + rk, j*bs + rj).
                            # Wait, the summation index in matrix mult is the middle one.
                            # (AB)[i, j] = sum_k A[i, k] B[k, j].
                            # In our block decomposition:
                            # C_block(i, j) = sum_k A_block(i, k) B_block(k, j).
                            # Inside the blocks, we do schoolbook 3x3 mult.
                            # (A_block(i,k) * B_block(k,j))[r, c] = sum_s A_block(i,k)[r, s] B_block(k,j)[s, c].
                            # So for each (r, s, c) in 0..2:
                            # A entry: global (i*bs+r, k*bs+s). U row: (i*bs+r)*6 + (k*bs+s).
                            # B entry: global (k*bs+s, j*bs+c). V row: (k*bs+s)*6 + (j*bs+c).
                            # C entry: global (i*bs+r, j*bs+c). W row: (j*bs+c)*6 + (i*bs+r).
                            
                            # My previous loop structure was slightly off. Let's restart the inner loops.
                            pass

        # Let's redo the inner loop structure properly.
        # For each outer term t (0..6):
        #   For each block i (0..1), k (0..1), j (0..1):
        #     a_coeff = s2_A[t][i], b_coeff = s2_B[t][k], c_coeff = s2_C[t][j]
        #     If any is 0, skip? No, we can still add 0, but skipping is faster.
        #     For each r, s, c in 0..2:
        #       U[(i*3+r)*6 + (k*3+s)][col] += a_coeff
        #       V[(k*3+s)*6 + (j*3+c)][col] += b_coeff
        #       W[(j*3+c)*6 + (i*3+r)][col] += c_coeff
        #       col += 1
        # Wait, this would give rank 7 * 2 * 2 * 2 * 27 = 7 * 8 * 27 = 1512? No.
        # The Strassen formula is:
        # C00 = M1 + M4 - M5 + M7
        # M1 = (A00 + A11) * (B00 + B11)
        # This is a product of SUMS of blocks.
        # The tensor decomposition is linear.
        # T = sum_t U_t V_t W_t.
        # For M1, U_t is the vector representing (A00 + A11).
        # V_t is the vector representing (B00 + B11).
        # W_t is the vector representing the output block C00.
        # But (A00 + A11) * (B00 + B11) is a 3x3 matrix product.
        # So we need the rank-27 decomposition of the 3x3 multiplication tensor.
        # Let T_3x3 be the 3x3 multiplication tensor.
        # T_3x3 = sum_{r=1}^{27} u_r v_r w_r.
        # Then M1 = sum_r (u_r * (A00+A11)) * (v_r * (B00+B11)) * w_r.
        # So for the global decomposition, for term M1:
        # For each r in 1..27:
        #   U_global[rows for A00] += u_r
        #   U_global[rows for A11] += u_r
        #   V_global[rows for B00] += v_r
        #   V_global[rows for B11] += v_r
        #   W_global[rows for C00] += w_r
        # This uses 27 rank terms for M1.
        # Since there are 7 Strassen terms, total rank = 7 * 27 = 189.
        
        # Let's implement this.
        pass

    # Correct implementation:
    col = 0
    for t_outer in range(7):
        a_coeffs = s2_A[t_outer]
        b_coeffs = s2_B[t_outer]
        c_coeffs = s2_C[t_outer]
        
        # For each of the 27 schoolbook terms in 3x3:
        for r3 in range(27):
            # r3 corresponds to (i3, k3, j3) in 0..2
            i3 = r3 // 9
            k3 = (r3 % 9) // 3
            j3 = r3 % 3
            
            # U part: A entries
            for bi in range(nb):
                a_coeff = a_coeffs[bi]
                if a_coeff == 0:
                    continue
                # A block (bi, bk) ? No, A is 6x6.
                # In Strassen, A is partitioned into A00, A01, A10, A11.
                # A00 is top-left 3x3. A01 is top-right 3x3.
                # A10 is bottom-left 3x3. A11 is bottom-right 3x3.
                # So block index bi is row block (0 or 1), and the column block is fixed by the Strassen term?
                # No, s2_A[t] has 4 entries: [a00, a01, a10, a11].
                # a00 is block (0,0), a01 is block (0,1), a10 is block (1,0), a11 is block (1,1).
                # So we iterate over all 4 blocks of A.
                for bj in range(nb):
                    # Block index in s2_A is bi*2 + bj ?
                    # s2_A = [a00, a01, a10, a11]
                    # Index 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
                    block_idx = bi * nb + bj
                    a_coeff = a_coeffs[block_idx]
                    if a_coeff == 0:
                        continue
                    # A entry global: (bi*3 + i3, bj*3 + k3)
                    u_row = (bi * bs + i3) * m + (bj * bs + k3)
                    U[u_row][col] = a_coeff
            
            # V part: B entries
            for bk in range(nb):
                for bj in range(nb):
                    block_idx = bk * nb + bj
                    b_coeff = b_coeffs[block_idx]
                    if b_coeff == 0:
                        continue
                    # B entry global: (bk*3 + k3, bj*3 + j3)
                    v_row = (bk * bs + k3) * p + (bj * bs + j3)
                    V[v_row][col] = b_coeff
            
            # W part: C entries
            for bi in range(nb):
                for bj in range(nb):
                    block_idx = bi * nb + bj
                    c_coeff = c_coeffs[block_idx]
                    if c_coeff == 0:
                        continue
                    # C entry global: (bi*3 + i3, bj*3 + j3)
                    # W index: col * n + row = (bj*3 + j3) * 6 + (bi*3 + i3)
                    w_row = (bj * bs + j3) * n + (bi * bs + i3)
                    W[w_row][col] = c_coeff
            
            col += 1

    # Quick verification
    if time.time() - start_time < DEADLINE - 10:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        test_pass = True
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    expected = 1.0
                    got = 0.0
                    for r in range(total_rank):
                        got += U_np[i*6+k, r] * V_np[k*6+j, r] * W_np[j*6+i, r]
                    if abs(got - expected) > 1e-9:
                        test_pass = False
                        break
                if not test_pass:
                    break
            if not test_pass:
                break
        
        if not test_pass:
            # Fallback to rank-216 (schoolbook)
            total_rank_fb = 216
            U_fb = [[0] * total_rank_fb for _ in range(n * m)]
            V_fb = [[0] * total_rank_fb for _ in range(m * p)]
            W_fb = [[0] * total_rank_fb for _ in range(p * n)]
            col = 0
            for i in range(6):
                for k in range(6):
                    for j in range(6):
                        U_fb[i*6+k][col] = 1.0
                        V_fb[k*6+j][col] = 1.0
                        W_fb[j*6+i][col] = 1.0
                        col += 1
            return {"U": U_fb, "V": V_fb, "W": W_fb}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
