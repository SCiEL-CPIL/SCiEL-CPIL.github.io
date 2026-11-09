import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    # Block 1: 3x3x3 (A[:, :3] * B[:3, :3]) -> C[:, :3]
    # Block 2: 3x3x3 (A[:, :3] * B[:3, 3:]) -> C[:, 3:]
    # Block 3: 3x1x6 (A[:, 3] * B[3, :]) -> C[:, :]
    
    rank_block1 = 27
    rank_block2 = 27
    rank_block3 = 1
    
    total_rank = rank_block1 + rank_block2 + rank_block3
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Block 1
    for i in range(3):
        for j in range(3):
            for k in range(3):
                r = i * 9 + j * 3 + k
                U[i * 4 + j][r] = 1
                V[j * 6 + k][r] = 1
                W[k * 3 + i][r] = 1
                
    # Block 2
    offset2 = rank_block1
    for i in range(3):
        for j in range(3):
            for k in range(3):
                r = offset2 + i * 9 + j * 3 + k
                U[i * 4 + j][r] = 1
                V[j * 6 + (k + 3)][r] = 1
                W[(k + 3) * 3 + i][r] = 1
                
    # Block 3
    offset3 = rank_block1 + rank_block2
    for i in range(3):
        U[i * 4 + 3][offset3] = 1
    for k in range(6):
        V[3 * 6 + k][offset3] = 1
    for i in range(3):
        for k in range(6):
            W[k * 3 + i][offset3] = 1
            
    return {"U": U, "V": V, "W": W}
