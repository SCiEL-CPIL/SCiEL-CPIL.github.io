<<<<<<< SEARCH
T_target = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    T_target[i * m + k, k * p + j, j * n + i] = 1.0
=======
T_target = np.zeros((n * m, m * p, p * n))
for i in range(n):
    for k in range(m):
        for j in range(p):
            T_target[i * m + k, k * p + j, j * n + i] = 1.0
>>>>>>> REPLACE
