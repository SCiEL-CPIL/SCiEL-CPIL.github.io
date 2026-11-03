import time
import sys

# EVOLVE-BLOCK-START

def construct():
    n, m, p = 4, 4, 7
    start_time = time.time()
    deadline = start_time + 140

    def add_strassen_block(a_r0, a_r1, a_c0, a_c1, b_c0, b_c1):
        """Add 7 Strassen terms for 2x2x2 block.
        A block: rows a_r0,a_r1; cols a_c0,a_c1
        B block: rows a_c0,a_c1; cols b_c0,b_c1
        C block: rows a_r0,a_r1; cols b_c0,b_c1
        """
        terms = [
            # m1 = (A00+A11)(B00+B11); C00+=m1, C11+=m1
            {
                'U': [(a_r0 * m + a_c0, 1), (a_r1 * m + a_c1, 1)],
                'V': [(a_c0 * p + b_c0, 1), (a_c1 * p + b_c1, 1)],
                'W': [(b_c0 * n + a_r0, 1), (b_c1 * n + a_r1, 1)]
            },
            # m2 = (A10+A11)B00; C10+=m2, C11-=m2
            {
                'U': [(a_r1 * m + a_c0, 1), (a_r1 * m + a_c1, 1)],
                'V': [(a_c0 * p + b_c0, 1)],
                'W': [(b_c0 * n + a_r1, 1), (b_c1 * n + a_r1, -1)]
            },
            # m3 = A00(B10-B11); C01+=m3, C11+=m3
            {
                'U': [(a_r0 * m + a_c0, 1)],
                'V': [(a_c1 * p + b_c0, 1), (a_c1 * p + b_c1, -1)],
                'W': [(b_c1 * n + a_r0, 1), (b_c1 * n + a_r1, 1)]
            },
            # m4 = A11(B01-B00); C00+=m4, C10+=m4
            {
                'U': [(a_r1 * m + a_c1, 1)],
                'V': [(a_c0 * p + b_c1, 1), (a_c0 * p + b_c0, -1)],
                'W': [(b_c0 * n + a_r0, 1), (b_c0 * n + a_r1, 1)]
            },
            # m5 = (A00+A10)B11; C00-=m5, C01+=m5
            {
                'U': [(a_r0 * m + a_c0, 1), (a_r0 * m + a_c1, 1)],
                'V': [(a_c1 * p + b_c1, 1)],
                'W': [(b_c0 * n + a_r0, -1), (b_c1 * n + a_r0, 1)]
            },
            # m6 = (A11-A01)(B00+B10); C11+=m6
            {
                'U': [(a_r1 * m + a_c1, 1), (a_r0 * m + a_c1, -1)],
                'V': [(a_c0 * p + b_c0, 1), (a_c0 * p + b_c1, 1)],
                'W': [(b_c1 * n + a_r1, 1)]
            },
            # m7 = (A01-A00)(B11+B10); C00+=m7, C01+=m7
            {
                'U': [(a_r0 * m + a_c1, 1), (a_r0 * m + a_c0, -1)],
                'V': [(a_c1 * p + b_c1, 1), (a_c1 * p + b_c0, 1)],
                'W': [(b_c0 * n + a_r0, 1), (b_c1 * n + a_r0, 1)]
            },
        ]
        return terms

    all_terms = []

    # 4x4x4 part: 8 Strassen blocks
    # C[i_blk, j_blk] = A[i_blk,0]*B[0,j_blk] + A[i_blk,1]*B[1,j_blk]
    for i_blk in range(2):
        for j_blk in range(2):
            for k_blk in range(2):
                a_r0 = i_blk * 2
                a_r1 = i_blk * 2 + 1
                a_c0 = k_blk * 2
                a_c1 = k_blk * 2 + 1
                b_c0 = j_blk * 2
                b_c1 = j_blk * 2 + 1
                all_terms.extend(add_strassen_block(a_r0, a_r1, a_c0, a_c1, b_c0, b_c1))

    # 4x4x3 part: trivial (48 terms)
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                all_terms.append({
                    'U': [(i * m + j, 1)],
                    'V': [(j * p + k, 1)],
                    'W': [(k * n + i, 1)]
                })

    rank = len(all_terms)

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for t, term in enumerate(all_terms):
        for idx, coeff in term['U']:
            U[idx][t] += coeff
        for idx, coeff in term['V']:
            V[idx][t] += coeff
        for idx, coeff in term['W']:
            W[idx][t] += coeff

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

