def construct():
    n, m, p = 4, 4, 6
    rank = 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2
    sU = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [1, 1, 0, 0, 0, 1, -1],
        [1, 1, -1, 1, 0, 0, -1],
    ]
    sV = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, -1, 1, 0, 1, 0],
        [1, 0, 1, 0, 1, 0, 1],
        [1, 0, -1, -1, 1, 0, 1],
    ]
    sW = [
        [1, 0, 0, -1, 1, 0, -1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 0, 1],
    ]
    
    col = 0
    for i_block in range(2):
        for j_block in range(2):
            for k_block in range(3):
                for c in range(7):
                    for i_l in range(2):
                        for j_l in range(2):
                            U[(2*i_block+i_l)*4 + (2*j_block+j_l)][col] += sU[i_l*2+j_l][c]
                    for j_l in range(2):
                        for k_l in range(2):
                            V[(2*j_block+j_l)*6 + (2*k_block+k_l)][col] += sV[j_l*2+k_l][c]
                    for k_l in range(2):
                        for i_l in range(2):
                            W[(2*k_block+k_l)*4 + (2*i_block+i_l)][col] += sW[k_l*2+i_l][c]
                    col += 1
    
    return {"U": U, "V": V, "W": W}
