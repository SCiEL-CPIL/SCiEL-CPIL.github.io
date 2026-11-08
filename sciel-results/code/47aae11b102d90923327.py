import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = 4, 4, 6
    R = 77
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    
    # Strassen 2x2x2 coefficients
    # alpha[s][a]: coefficient of A-block a in s-th product (a: 0=00,1=01,2=10,3=11)
    alpha = [
        [1, 0, 0, 1],   # s0: A00+A11
        [0, 0, 1, 1],   # s1: A10+A11
        [1, 0, 0, 0],   # s2: A00
        [0, 0, 0, 1],   # s3: A11
        [1, 1, 0, 0],   # s4: A00+A01
        [-1, 0, 1, 0],  # s5: A10-A00
        [0, 1, 0, -1]   # s6: A01-A11
    ]
    beta = [
        [1, 0, 0, 1],   # s0: B00+B11
        [1, 0, 0, 0],   # s1: B00
        [0, 1, 0, -1],  # s2: B01-B11
        [-1, 0, 1, 0],  # s3: B10-B00
        [0, 0, 0, 1],   # s4: B11
        [1, 1, 0, 0],   # s5: B00+B01
        [0, 0, 1, 1]    # s6: B10+B11
    ]
    # recon[s]: list of (a_block, b_block, coeff) for C reconstruction
    recon = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]
    
    col = 0
    for s in range(7):
        if time.time() > deadline:
            break
        # Inner Strassen for 2x2 x 2x2 (cols 0,1 of the 2x3 subproblem)
        for si in range(7):
            if time.time() > deadline:
                break
            # U: 4x4 input -> index (ar*2+xi, br*2+xc) where a=ar*2+xc, x=xi*2+xc... 
            # a indexes 2x2 blocks: a = row*2 + col
            # r indexes top-level 2x2 blocks: r = tr*2 + tc
            # u_idx = (tr*2 + xi)*4 + (tc*2 + xc)
            for a in range(4):
                ac = alpha[si][a]
                if ac:
                    xi, xc = a // 2, a % 2
                    for r in range(4):
                        tc_coeff = alpha[s][r]
                        if tc_coeff:
                            tr, tc = r // 2, r % 2
                            u_idx = (tr * 2 + xi) * 4 + (tc * 2 + xc)
                            U[u_idx, col] += ac * tc_coeff
            # V: 4x6 input, b indexes 2x2 blocks of first 4 cols: b = row*2 + col
            # c indexes top-level 2x2 blocks: c = tr*2 + tc
            # v_idx = (tr*2 + yi)*6 + (tc*3 + yc) where b=yi*2+yc, but only for cols 0,1
            # Actually b indexes 2x2 blocks (cols 0,1), so b = yi*2+yc, yc in {0,1}
            for b in range(4):
                bc = beta[si][b]
                if bc:
                    yi, yc = b // 2, b % 2
                    for c in range(4):
                        tc_coeff = beta[s][c]
                        if tc_coeff:
                            tr, tc = c // 2, c % 2
                            v_idx = (tr * 2 + yi) * 6 + (tc * 3 + yc)
                            V[v_idx, col] += bc * tc_coeff
            # W: output 2x3, reconstruction
            for a, b, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    # a: top-level 2x2 block row/col, za: inner 2x2 block row/col
                    # b: top-level 2x3 block col, zb: inner 2x2 block col (0 or 1)
                    # w_idx = (top_col*3 + inner_col)*4 + (top_row*2 + inner_row)
                    # a = top_row*2 + top_col_inner... wait
                    # a indexes 2x2 blocks: a = row*2 + col, so a//2=row, a%2=col
                    # b indexes 2x3 blocks: b = row*3 + col? No, b is 0..3 for 2x2
                    # In recon[s], a is 0..3 (2x2 block), b is 0..3 (2x2 block of 2x3, cols 0,1)
                    # So a = ar*2+ac, b = br*2+bc
                    ar, ac_ = a // 2, a % 2
                    br, bc_ = b // 2, b % 2
                    zar, zac = za // 2, za % 2
                    zbr, zbc = zb // 2, zb % 2
                    w_idx = (br * 3 + zbc) * 4 + (ar * 2 + zac)
                    W[w_idx, col] += gamma * zgamma
            col += 1
        
        # Schoolbook for 2x2 x 2x1 (col 2 of the 2x3 subproblem)
        for u in range(2):
            for v in range(2):
                if time.time() > deadline:
                    break
                for r in range(4):
                    tc_coeff = alpha[s][r]
                    if tc_coeff:
                        tr, tc = r // 2, r % 2
                        u_idx = (tr * 2 + u) * 4 + (tc * 2 + v)
                        U[u_idx, col] += tc_coeff
                for c in range(4):
                    tc_coeff = beta[s][c]
                    if tc_coeff:
                        tr, tc = c // 2, c % 2
                        v_idx = (tr * 2 + v) * 6 + (tc * 3 + 2)
                        V[v_idx, col] += tc_coeff
                for a, b, gamma in recon[s]:
                    ar, ac_ = a // 2, a % 2
                    br, bc_ = b // 2, b % 2
                    # Output is at col 2, row ar*2+u... wait, u is the row within 2x2
                    # Actually for 2x2 x 2x1, output is 2x1, indexed by row
                    # a in recon[s] is 0..3, but we only care about the row part
                    # The output row is ar*2+u where u is 0 or 1
                    w_idx = (br * 3 + 2) * 4 + (ar * 2 + u)
                    W[w_idx, col] += gamma
                col += 1
    
    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
