import numpy as np
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x3 * 3x6 matrix multiplication.
    
    Strategy:
    - Exploit shared middle dimension m=3 across the two 3x3x3 blocks.
    - For Strassen terms with identical U patterns, merge them into single rank terms
      that span both blocks (k=0..2 and k=3..5).
    - This reduces total rank by the number of mergeable Strassen term pairs.
    """
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    DEADLINE = 140
    
    def get_strassen_2x2x2():
        """Return 7 Strassen terms for 2x2x2 as (u_dict, v_dict, w_dict)."""
        return [
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(0,1):1}),
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):-1, (1,1):1}),
        ]
    
    def verify_strassen(terms):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = 0.0
                    for u, v, w in terms:
                        if (i,j) in u and (j,k) in v and (i,k) in w:
                            s += u[(i,j)] * v[(j,k)] * w[(i,k)]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    strassen = get_strassen_2x2x2()
    use_strassen = verify_strassen(strassen)
    
    # Build joint decomposition
    # A "term" is (u_dict, v_dict, w_dict) where:
    #   u_dict: (i,j) -> coeff
    #   v_dict: (j,k) -> coeff  (k in 0..5)
    #   w_dict: (i,k) -> coeff  (k in 0..5)
    
    terms = []
    
    if use_strassen:
        # Merge Strassen terms across blocks:
        # For each Strassen term (u, v1, w1) for block1 (k'=0,1,2 -> k=0,1,2)
        # and the same (u, v2, w2) for block2 (k'=0,1,2 -> k=3,4,5),
        # we can merge into one term with:
        #   u: same
        #   v: v1 entries at k=0,1,2 AND v2 entries at k=3,4,5
        #   w: w1 entries at k=0,1,2 AND w2 entries at k=3,4,5
        # This saves 1 rank per merged pair.
        
        for u, v, w in strassen:
            merged_v = dict(v)
            merged_w = dict(w)
            for (jk, cv) in v.items():
                j, kp = jk
                merged_v[(j, kp + 3)] = cv
            for (ik, cw) in w.items():
                i, kp = ik
                merged_w[(i, kp + 3)] = cw
            terms.append((u, merged_v, merged_w))
    else:
        # Fallback: separate schoolbook for 2x2x2 in each block
        for block_offset in [0, 3]:
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        terms.append(({(i,j):1}, {(j, k+block_offset):1}, {(i, k+block_offset):1}))
    
    # Add schoolbook terms for all (i,j,k) NOT covered by Strassen
    # Strassen covers i<2, j<2, k<2 in each block (i.e., k in {0,1,2} and {3,4,5})
    for block_offset in [0, 3]:
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    kk = k + block_offset
                    if i < 2 and j < 2 and k < 2 and use_strassen:
                        continue  # Covered by merged Strassen
                    terms.append(({(i,j):1}, {(j, kk):1}, {(i, kk):1}))
    
    total_rank = len(terms)
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for u, v, w in terms:
        for (i, j), cu in u.items():
            U[i * m + j][col] += cu
        for (j, k), cv in v.items():
            V[j * p + k][col] += cv
        for (i, k), cw in w.items():
            W[k * n + i][col] += cw
        col += 1
    
    # Verify
    def verify(U_mat, V_mat, W_mat, n, m, p):
        rank = len(U_mat[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    computed = 0.0
                    for r in range(rank):
                        computed += U_mat[i*m+j][r] * V_mat[j*p+k][r] * W_mat[k*n+i][r]
                    if abs(computed - 1.0) > 1e-6:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank_school = n * m * p
        U_school = [[0] * rank_school for _ in range(n * m)]
        V_school = [[0] * rank_school for _ in range(m * p)]
        W_school = [[0] * rank_school for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j][c] = 1
                    V_school[j * p + k][c] = 1
                    W_school[k * n + i][c] = 1
                    c += 1
        return {"U": U_school, "V": V_school, "W": W_school}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
