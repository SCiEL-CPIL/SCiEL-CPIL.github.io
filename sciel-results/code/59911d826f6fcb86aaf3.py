        for j in range(4):
            c = 49 + j
            U[j][c] = 1
            V[j*5+4][c] = 1
            for i in range(4):
                W[4*4+i][c] = 1
