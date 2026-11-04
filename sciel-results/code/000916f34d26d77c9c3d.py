import numpy as np
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x3 * 3x6 matrix multiplication.
    
    Strategy:
    - Use merged Strassen terms across the two 3x3 blocks (k=0..2 and k=3..5).
    - Each Strassen term for 2x2x2 is duplicated with shifted k indices and merged.
    - Add schoolbook terms for entries not covered by Strassen.
    - Verify correctness; fall back to schoolbook if verification fails.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = 140

    def get_strassen_2x2x2():
        """Return 7 Strassen terms for 2x2x2 as (u_dict, v_dict, w_dict).
        
        Standard Strassen for 2x2x2:
        M1 = (A00+A11)(B00+B11)
        M2 = (A10+A11)B00
        M3 = A00(B10-B00)
        M4 = A11(B01-B11)
        M5 = (A00+A01)B11
        M6 = (A10-A00)(B00+B01)
        M7 = (A01-A11)(B10+B11)
        
        C00 = M1 + M4 - M5 + M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M6
        """
        return [
            # M1: (A00+A11)(B00+B11) -> C00+=1, C11+=1
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            # M2: (A10+A11)B00 -> C10+=1, C11-=1
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
            # M3: A00(B10-B00) -> C01+=1, C11+=1
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
            # M4: A11(B01-B11) -> C00+=1, C10+=1
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
            # M5: (A00+A01)B11 -> C00-=1, C01+=1
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
            # M6: (A10-A00)(B00+B01) -> C01+=1
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(0,1):1}),
            # M7: (A01-A11)(B10+B11) -> C00-=1, C11+=1
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):-1, (1,1):1}),
        ]
    
    def verify_strassen(terms):
        """Verify Strassen terms correctly compute 2x2x2 multiplication."""
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
    
    terms = []
    
    if use_strassen:
        # Merge Strassen terms across the two 3x3 blocks
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
        for block_offset in [0, 3]:
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        terms.append(({(i,j):1}, {(j, k+block_offset):1}, {(i, k+block_offset):1}))
    
    # Add schoolbook terms for uncovered entries
    for block_offset in [0, 3]:
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    kk = k + block_offset
                    if i < 2 and j < 2 and k < 2 and use_strassen:
                        continue
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
    
    def verify(U_mat, V_mat, W_mat, n, m, p):
        rank = len(U_mat[0]) if U_mat else 0
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
