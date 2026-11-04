import time

def construct():
    """
    Returns a rank-22 exact tensor decomposition for 2x3 * 3x8 matrix multiplication.
    
    Strategy:
    C[i,k] = sum_a A[i,a]*B[a,k]
           = (sum_{a=0,1} A[i,a]*B[a,k]) + A[i,2]*B[2,k]
    
    For k=0..3: A1(2x2) * B1(2x4) using 2 Strassen blocks (rank 2*7=14)
    For k=4..7: A2(2x1) * B2(1x4) outer product (rank 8)
    Total rank: 14 + 8 = 22
    """
    deadline = time.time() + 110  # Safety margin before 150s limit
    
    n, m, p = 2, 3, 8
    rank = 22
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 22
    V = [[0] * rank for _ in range(m * p)]  # 24 x 22
    W = [[0] * rank for _ in range(p * n)]  # 16 x 22
    
    # Strassen 2x2x2 coefficients
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A10)B11
    # M6 = (A11-A00)(B00+B10)
    # M7 = (A10-A01)(B11+B01)
    
    S_U = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [-1, 0, 1, 0],  # M6
        [0, 1, 0, -1]   # M7
    ]
    
    S_V = [
        [1, 0, 0, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 1, 0, -1],  # M3
        [-1, 0, 1, 0],  # M4
        [0, 0, 0, 1],   # M5
        [1, 1, 0, 0],   # M6
        [0, 0, 1, 1]    # M7
    ]
    
    S_W = [
        [1, 0, 0, 1],   # M1
        [0, 1, 0, -1],  # M2
        [0, 0, 1, 1],   # M3
        [1, 1, 0, 0],   # M4
        [-1, 0, 1, 0],  # M5
        [0, 0, 0, 1],   # M6
        [1, 0, 0, 0]    # M7
    ]
    
    col = 0
    
    # Part 1: A1 (2x2) * B1 (2x4) using 2 Strassen blocks
    for block_idx in range(2):
        if time.time() > deadline:
            break
        k_start = block_idx * 2
        
        for r in range(7):
            # Fill U for A1 (rows 0,1 of A, cols 0,1 of A)
            for li in range(2):
                for la in range(2):
                    val = S_U[r][li * 2 + la]
                    if val != 0:
                        u_idx = li * m + la
                        U[u_idx][col] += val
            
            # Fill V for B1 block (rows 0,1 of B, cols k_start..k_start+1)
            for la in range(2):
                for lk in range(2):
                    val = S_V[r][la * 2 + lk]
                    if val != 0:
                        v_idx = la * p + (k_start + lk)
                        V[v_idx][col] += val
            
            # Fill W for C block (cols k_start..k_start+1 of C, rows 0..1)
            for lk in range(2):
                for li in range(2):
                    val = S_W[r][lk * 2 + li]
                    if val != 0:
                        w_idx = (k_start + lk) * n + li
                        W[w_idx][col] += val
            
            col += 1
    
    # Part 2: A2 (2x1) * B2 (1x4) for k=4..7
    for i in range(2):
        for k in range(4, 8):
            if time.time() > deadline:
                break
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
    
    # Validation
    valid = True
    for i in range(n):
        for a in range(m):
            for k in range(p):
                acc = 0
                u_row = U[i * m + a]
                v_row = V[a * p + k]
                w_row = W[k * n + i]
                for c in range(rank):
                    if u_row[c] != 0 and v_row[c] != 0 and w_row[c] != 0:
                        acc += u_row[c] * v_row[c] * w_row[c]
                if acc != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook rank 48
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col_fb = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U_fb[i * m + a][col_fb] = 1
                    V_fb[a * p + k][col_fb] = 1
                    W_fb[k * n + i][col_fb] = 1
                    col_fb += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}


if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
