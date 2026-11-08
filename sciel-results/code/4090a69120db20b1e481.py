        for i in range(4):
            for k in range(5):
                U[i*5+4][col] = 1
                V[4*5+k][col] = 1
                W[k*4+i][col] = 1
                col += 1
        