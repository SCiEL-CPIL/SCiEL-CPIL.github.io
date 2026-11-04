    def construct():
        n, m, p = 3, 3, 8
        rank = 24
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        r = 0
        # Block 1: M_{3,3,3} for k in {0, 1, 2}
        for i in range(n):
            for j in range(m):
                for k in range(3):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        
        # Block 2: M_{3,3,1} for k = 3
        for i in range(n):
            for j in range(m):
                k = 3
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
        
        # Block 3: M_{3,3,3} for k in {4, 5, 6}
        for i in range(n):
            for j in range(m):
                for k in range(4, 7):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        
        # Block 4: M_{3,3,1} for k = 7
        for i in range(n):
            for j in range(m):
                k = 7
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
        
        # Verify
        # ...
        
        return {"U": U, "V": V, "W": W}
    