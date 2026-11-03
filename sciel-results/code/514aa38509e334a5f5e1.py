import numpy as np
import time

DIMENSIONS = [4, 4, 6]
DEADLINE = 140.0

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Pre-compute Strassen coefficients
    # A and B are 2x2x2 blocks.
    # alpha[b, i, j] = coefficient of A[i, j] in Strassen sum b
    # Using standard Strassen:
    # m0 = (A00+A11)(B00+B11)
    # m1 = (A10+A11)B00
    # m2 = A00(B01-B11)
    # m3 = A11(B10-B00)
    # m4 = (A00+A01)B11
    # m5 = (A10-A00)(B00+B10)
    # m6 = (A01-A11)(B01+B11)
    
    # Coeffs for A (row, col)
    a_coeffs = [
        [1, 0, 1, 0], # s=0: A00+A11 -> (0,0)=1, (1,1)=1
        [0, 0, 1, 1], # s=1: A10+A11 -> (1,0)=1, (1,1)=1
        [1, 0, 0, 0], # s=2: A00
        [0, 0, 0, 1], # s=3: A11
        [1, 1, 0, 0], # s=4: A00+A01
        [-1, 0, 1, 0],# s=5: A10-A00
        [0, 1, 0, -1] # s=6: A01-A11
    ]
    
    # Coeffs for B (row, col)
    b_coeffs = [
        [1, 0, 0, 1], # s=0: B00+B11
        [1, 0, 0, 0], # s=1: B00
        [0, 1, 0, -1],# s=2: B01-B11
        [-1, 0, 1, 0],# s=3: B10-B00
        [0, 0, 0, 1], # s=4: B11
        [1, 0, 1, 0], # s=5: B00+B10
        [0, 1, 0, 1]  # s=6: B01+B11
    ]
    
    # Reconstruction: C = sum s * (A_s B_s)
    # C00 = m0 - m2 + m3 + m5
    # C01 = m4 + m6
    # C10 = m1 + m3
    # C11 = m0 - m1 + m4 + m6
    
    # recon[c_row, c_col, s] = coeff
    recon = [
        [1, -1, 0, 1, 1, 0, 0], # C00? No.
        # Standard:
        # C00 = m0 - m2 + m3 + m5
        # C01 = m4 + m6
        # C10 = m1 + m3
        # C11 = m0 - m1 + m4 + m6
    ]
    
    # Let's use the explicit tuples from the original code which were correct
    recon = [
        [(0, 0, 1), (1, 1, 1)],     # s=0
        [(1, 0, 1), (1, 1, -1)],    # s=1
        [(0, 1, 1), (1, 1, 1)],     # s=2
        [(0, 0, 1), (1, 0, 1)],     # s=3
        [(0, 0, -1), (0, 1, 1)],    # s=4
        [(1, 1, 1)],                # s=5
        [(0, 0, 1)]                 # s=6
    ]

    R = 77
    U = np.zeros((16, R), dtype=float)
    V = np.zeros((24, R), dtype=float)
    W = np.zeros((24, R), dtype=float)

    col_idx = 0
    for s in range(7):
        # Part 1: Cols 0,1,3,4 of B
        for si in range(7):
            # U updates
            for a in range(4):
                ca = a_coeffs[si][a]
                if ca != 0:
                    ar, ac = a // 2, a % 2
                    for r in range(4):
                        cs = a_coeffs[s][r]
                        if cs != 0:
                            sr, sc = r // 2, r % 2
                            u_idx = (sr * 2 + ar) * 4 + (sc * 2 + ac)
                            U[u_idx, col_idx] += ca * cs

            # V updates
            for b in range(4):
                cb = b_coeffs[si][b]
                if cb != 0:
                    br, bc = b // 2, b % 2
                    for c in range(4):
                        cs = b_coeffs[s][c]
                        if cs != 0:
                            sr, sc = c // 2, c % 2
                            # Map to B cols 0,1,3,4
                            # b_idx 0,1 -> cols 0,1
                            # b_idx 2,3 -> cols 3,4
                            if bc == 0:
                                b_col = br
                            else:
                                b_col = 3 + br
                            v_idx = (sr * 2 + br) * 6 + b_col
                            V[v_idx, col_idx] += cb * cs

            # W updates
            for ra, rb, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    # Map to C cols 0,1,3,4
                    if rb == 0:
                        c_col = ra
                    else:
                        c_col = 3 + ra
                    w_idx = c_col * 4 + (za * 2 + ra) # Wait, row is za?
                    # Original: w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                    # b is from recon[s] (c_col index). zb is from recon[si] (c_col inner).
                    # Here rb is the 'b' from recon[s]. zb is 'b' from recon[si].
                    # The original code used:
                    # w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                    # where b, a were row/col of C block.
                    # My mapping:
                    # c_block_col = rb (0 or 1)
                    # inner_col = zb (0 or 1)
                    # c_row_block = a (0 or 1) from recon[s]
                    # inner_row = za (0 or 1) from recon[si]
                    
                    # The original code:
                    # for a, b, gamma in recon[s]:  # a is row, b is col?
                    # for za, zb, zgamma in recon[si]:
                    # w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                    
                    # So 'b' is col block, 'a' is row block.
                    # 'zb' is inner col, 'za' is inner row.
                    
                    c_col_block = rb
                    c_row_block = ra
                    c_inner_col = zb
                    c_inner_row = za
                    
                    c_col = c_col_block * 3 + c_inner_col
                    c_row = c_row_block * 2 + c_inner_row
                    
                    w_idx = c_col * 4 + c_row
                    W[w_idx, col_idx] += gamma * zgamma

            col_idx += 1

        # Part 2: Cols 2,5 of B
        for u in range(2):
            for v in range(2):
                # U updates
                for r in range(4):
                    cs = a_coeffs[s][r]
                    if cs != 0:
                        sr, sc = r // 2, r % 2
                        u_idx = (sr * 2 + u) * 4 + (sc * 2 + v)
                        U[u_idx, col_idx] += cs

                # V updates
                for c in range(4):
                    cs = b_coeffs[s][c]
                    if cs != 0:
                        sr, sc = c // 2, c % 2
                        # Map to B cols 2,5
                        # sc is inner col (0 or 1)
                        # b_idx 0,1 -> col 2
                        # b_idx 2,3 -> col 5
                        if sc == 0:
                            b_col = 2
                        else:
                            b_col = 5
                        v_idx = (sr * 2 + v) * 6 + b_col
                        V[v_idx, col_idx] += cs

                # W updates
                for ra, rb, gamma in recon[s]:
                    # u is row inner, v is col inner?
                    # Original: w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
                    # rb is col block (0 or 1).
                    # If rb==0, col 2. If rb==1, col 5.
                    # ra is row block.
                    # u is inner row.
                    
                    if rb == 0:
                        c_col = 2
                    else:
                        c_col = 5
                    c_row = ra * 2 + u
                    w_idx = c_col * 4 + c_row
                    W[w_idx, col_idx] += gamma

                col_idx += 1

        if time.time() - start_time > DEADLINE:
            break

    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
