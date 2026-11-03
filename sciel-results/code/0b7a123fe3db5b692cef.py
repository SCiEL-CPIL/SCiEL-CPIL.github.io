<<<<<<< SEARCH
    # Fill remaining (i,j,k) triples with schoolbook (1 term each)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in strassen_set:
                    continue
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
=======
    # Fill remaining (i,j,k) triples with schoolbook (1 term each)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i * m * p + j * p + k
                if (strassen_mask >> idx) & 1:
                    continue
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
>>>>>>> REPLACE
