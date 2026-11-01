import time

DIMENSIONS = [2, 2, 16]

def construct():
    n, m, p = DIMENSIONS
    # Group the p=16 columns into p//m = 8 groups of m=2? No.
    # Better: group into 4 groups of 4 columns each.
    # Each group of 4 columns requires rank 8 for 2x2 x 2x4 multiplication.
    # Total rank = 4 * 8 = 32.
    group_size = 4  # columns per group
    num_groups = p // group_size  # 4 groups
    rank = num_groups * n * m * group_size  # 4 * 2 * 2 * 4 = 64? No.
    
    # Wait, for 2x2 x 2x4, schoolbook is 2*2*4=8.
    # So rank = num_groups * 8 = 4 * 8 = 32.
    rank = num_groups * n * m * group_size  # This is 4*2*2*4 = 64, wrong.
    
    # Correct: each group has n*m*group_size = 2*2*4 = 8 terms.
    rank = num_groups * (n * m * group_size)  # 4 * 8 = 32
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for g in range(num_groups):
        for i in range(n):
            for j in range(m):
                for l in range(group_size):
                    col_idx = g * group_size + l  # actual column in B/C
                    U[i * m + j][column] = 1
                    V[j * p + col_idx][column] = 1
                    W[col_idx * n + i][column] = 1
                    column += 1
    
    return {"U": U, "V": V, "W": W}
