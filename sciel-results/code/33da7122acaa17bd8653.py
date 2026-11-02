import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = 2, 4, 8
    nm, mp, pn = n * m, m * p, p * n

    # Schoolbook rank 64
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    # Strassen rank 56
    # 2x2 Strassen terms
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = A B
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)(b00)
    # M3 = (a00)(b01-b11)
    # M4 = (a11)(b10-b00)
    # M5 = (a00+a01)(b11)
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # Represent terms as (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: list of (i, j, val)
    # V_coeffs: list of (j, k, val)
    # W_coeffs: list of (k, i, val)
    
    def make_term(u_list, v_list, w_list):
        return (u_list, v_list, w_list)

    strassen_2x2 = [
        make_term([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]), # M1
        make_term([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),          # M2
        make_term([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),          # M3
        make_term([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),          # M4
        make_term([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),          # M5
        make_term([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),          # M6
        make_term([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),          # M7
    ]

    rank_56 = 56
    U_56 = np.zeros((nm, rank_56))
    V_56 = np.zeros((mp, rank_56))
    W_56 = np.zeros((pn, rank_56))

    col = 0
    # A is 2x4. Split j into 2 blocks of 2.
    # B is 4x8. Split j into 2 blocks of 2, k into 4 blocks of 2.
    # For each pair of (A_block, B_block), we do a 2x2x2 multiplication.
    # A_block index: 0, 1 (corresponding to j in 0-1, 2-3)
    # B_block index: 0, 1, 2, 3 (corresponding to k in 0-1, 2-3, 4-5, 6-7)
    # Wait, B is 4x8. j is 0-3. k is 0-7.
    # We split j into 0-1 and 2-3.
    # We split k into 0-1, 2-3, 4-5, 6-7.
    # So for each A_block (2 choices) and each k_block (4 choices), we have a 2x2x2 block.
    # The j index for B must match the j index for A.
    # If A_block is 0 (j=0,1), B_block must use j=0,1.
    # If A_block is 1 (j=2,3), B_block must use j=2,3.
    
    for a_blk in range(2):
        for k_blk in range(4):
            # j offset for this block
            j_off = 2 * a_blk
            # k offset for this block
            k_off = 2 * k_blk
            
            for u_list, v_list, w_list in strassen_2x2:
                for (li, lj, coeff) in u_list:
                    gi = li
                    gj = j_off + lj
                    U_56[gi * m + gj, col] += coeff
                for (lj, lk, coeff) in v_list:
                    gj = j_off + lj
                    gk = k_off + lk
                    V_56[gj * p + gk, col] += coeff
                for (lk, li, coeff) in w_list:
                    gk = k_off + lk
                    gi = li
                    W_56[gk * n + gi, col] += coeff
                col += 1

    # Verify
    if time.time() < deadline:
        # Reshape correctly
        # U_56 is (nm, R). We want (n, m, R).
        # U_56.T is (R, nm).
        # Reshape to (R, n, m) then transpose to (n, m, R).
        U_t = U_56.T.reshape(rank_56, n, m).transpose(1, 2, 0)
        V_t = V_56.T.reshape(rank_56, m, p).transpose(1, 2, 0)
        W_t = W_56.T.reshape(rank_56, p, n).transpose(1, 2, 0)
        
        result = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
        target = np.ones((n, m, p))
        
        if np.allclose(result, target, atol=1e-10):
            return {"U": U_56.tolist(), "V": V_56.tolist(), "W": W_56.tolist()}

    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
