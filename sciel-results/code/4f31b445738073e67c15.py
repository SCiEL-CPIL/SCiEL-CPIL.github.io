import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    DEADLINE = 120

    n, m, p = DIMENSIONS  # 4, 4, 6

    # 2x2x3 decomposition (rank 4)
    # A is 2x2, B is 2x3, C is 2x3
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # U: 4 entries (i,j) for i in 0..1, j in 0..1
    # V: 6 entries (j,k) for j in 0..1, k in 0..2
    # W: 6 entries (k,i) for k in 0..2, i in 0..1
    #
    # Term 1: A[0,0] * B[0,k] -> C[0,k]
    # Term 2: A[0,1] * B[1,k] -> C[0,k]
    # Term 3: A[1,0] * B[0,k] -> C[1,k]
    # Term 4: A[1,1] * B[1,k] -> C[1,k]
    
    # For 4x4x3: split A into 2x2 blocks, B into top 2x3 and bottom 2x3
    # C_top (2x3) = A11*B_top + A12*B_bottom
    # C_bottom (2x3) = A21*B_top + A22*B_bottom
    # Each 2x2x3 uses rank 4, total rank 16
    
    # For 4x4x6: two independent 4x4x3, total rank 32
    
    total_rank = 32
    
    U = [[0] * total_rank for _ in range(n * m)]  # 16 x 32
    V = [[0] * total_rank for _ in range(m * p)]  # 24 x 32
    W = [[0] * total_rank for _ in range(p * n)]  # 24 x 32
    
    # Process each half of p (0-2 and 3-5)
    for half in range(2):
        # Base column for this half
        base_col = half * 16
        
        # For each 2x2 block of A and corresponding B block
        for bi in range(2):  # row block of A (0 or 1)
            for bj in range(2):  # col block of A / row block of B (0 or 1)
                # This 2x2x3 block
                # A block: rows bi*2 to bi*2+1, cols bj*2 to bj*2+1
                # B block: rows bj*2 to bj*2+1, cols half*3 to half*3+2
                # C block: rows bi*2 to bi*2+1, cols half*3 to half*3+2
                
                # 4 terms for this 2x2x3
                for di in range(2):
                    for dj in range(2):
                        i = bi * 2 + di
                        j = bj * 2 + dj
                        row_U = i * m + j
                        # Term index within this 2x2x3: di*2+dj
                        term_idx = di * 2 + dj
                        col = base_col + (bi * 2 + bj) * 4 + term_idx
                        
                        # U contribution: A[i,j]
                        U[row_U][col] = 1
                        
                        # V contribution: B[j, k] for k in 0..2
                        for dk in range(3):
                            k = half * 3 + dk
                            row_V = j * p + k
                            V[row_V][col] = 1
                        
                        # W contribution: C[k, i] for k in 0..2
                        for dk in range(3):
                            k = half * 3 + dk
                            row_W = k * n + i
                            W[row_W][col] = 1
    
    # Remove zero columns
    keep_cols = []
    for c in range(total_rank):
        if any(U[r][c] != 0 for r in range(n * m)) or \
           any(V[r][c] != 0 for r in range(m * p)) or \
           any(W[r][c] != 0 for r in range(p * n)):
            keep_cols.append(c)
    
    final_rank = len(keep_cols)
    U_final = [[U[r][c] for c in keep_cols] for r in range(n * m)]
    V_final = [[V[r][c] for c in keep_cols] for r in range(m * p)]
    W_final = [[W[r][c] for c in keep_cols] for r in range(p * n)]
    
    return {"U": U_final, "V": V_final, "W": W_final}
