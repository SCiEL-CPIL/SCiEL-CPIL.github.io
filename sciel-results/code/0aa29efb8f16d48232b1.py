import time

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Strassen's 7 products for 2x2 block matrix of 3x3 blocks
    strassen_terms = [
        # (a_factors, b_factors, c_contributions)
        # Term 0: M1 = (A00+A11)(B00+B11), C00+=1, C11+=1
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # Term 1: M2 = (A10+A11)(B00), C10+=1, C11+=-1
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        # Term 2: M3 = (A00)(B01-B11), C01+=1, C11+=1
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        # Term 3: M4 = (A11)(B10-B00), C00+=1, C10+=1
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        # Term 4: M5 = (A00+A01)(B11), C00+=-1, C01+=1
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        # Term 5: M6 = (A10-A00)(B00+B01), C11+=1
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # Term 6: M7 = (A01-A11)(B10+B11), C00+=1, C10+=1
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1),(1,0,1)]),
    ]
    
    rank = 189
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    col = 0
    for t_idx, (a_factors, b_factors, c_contribs) in enumerate(strassen_terms):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for (ba_r, ba_c, ca) in a_factors:
                        u_row = (3*ba_r + i) * 6 + (3*ba_c + j)
                        U[u_row][col] = ca
                    for (bb_r, bb_c, cb) in b_factors:
                        v_row = (3*bb_r + j) * 6 + (3*bb_c + k)
                        V[v_row][col] = cb
                    for (bc_r, bc_c, cc) in c_contribs:
                        w_row = (3*bc_c + k) * 6 + (3*bc_r + i)
                        W[w_row][col] = cc
                    col += 1
                    if time.time() - start_time > 140:
                        break
                if time.time() - start_time > 140:
                    break
            if time.time() - start_time > 140:
                break
        if time.time() - start_time > 140:
            break
    
    return {"U": U, "V": V, "W": W}
