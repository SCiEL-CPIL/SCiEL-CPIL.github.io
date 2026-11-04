import time

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = 2, 3, 6
    nU = n * m  # 6
    nV = m * p  # 18
    nW = p * n  # 12

    # Strassen factors for 2x2x2 (rank 7)
    S_U = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, 1]],
        [[1, 0], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[-1, 0], [1, 0]],
        [[0, 1], [0, -1]]
    ]
    S_V = [
        [[1, 0], [0, 1]],
        [[1, 0], [0, 0]],
        [[0, 1], [0, -1]],
        [[-1, 0], [1, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[0, 0], [1, 1]]
    ]
    S_W = [
        [[1, 0], [0, 1]],
        [[0, 1], [0, -1]],
        [[0, 0], [1, 1]],
        [[1, 1], [0, 0]],
        [[-1, 0], [1, 0]],
        [[0, 0], [0, 1]],
        [[1, 0], [0, 0]]
    ]

    # Standard factors for 2x1x2 (rank 4)
    Std_U = [
        [[1], [0]],
        [[1], [0]],
        [[0], [1]],
        [[0], [1]]
    ]
    Std_V = [
        [[1, 0]],
        [[0, 1]],
        [[1, 0]],
        [[0, 1]]
    ]
    Std_W = [
        [[1, 0], [0, 0]],
        [[0, 0], [1, 0]],
        [[0, 1], [0, 0]],
        [[0, 0], [0, 1]]
    ]

    U = [[0 for _ in range(33)] for _ in range(nU)]
    V = [[0 for _ in range(33)] for _ in range(nV)]
    W = [[0 for _ in range(33)] for _ in range(nW)]

    col = 0
    for b in range(3):
        # Strassen part: j in {0,1}, k in {2b, 2b+1}
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    U[i * m + j][col] = S_U[r][i][j]
            for j in range(2):
                for kp in range(2):
                    k_global = 2 * b + kp
                    V[j * p + k_global][col] = S_V[r][j][kp]
            for kp in range(2):
                for i in range(2):
                    k_global = 2 * b + kp
                    W[k_global * n + i][col] = S_W[r][kp][i]
            col += 1
        
        # Standard part: j=2, k in {2b, 2b+1}
        for r in range(4):
            for i in range(2):
                U[i * m + 2][col] = Std_U[r][i][0]
            for kp in range(2):
                k_global = 2 * b + kp
                V[2 * p + k_global][col] = Std_V[r][0][kp]
            for kp in range(2):
                for i in range(2):
                    k_global = 2 * b + kp
                    W[k_global * n + i][col] = Std_W[r][kp][i]
            col += 1

    return {
        "U": U,
        "V": V,
        "W": W
    }
