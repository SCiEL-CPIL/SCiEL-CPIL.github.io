    gcd_table = [[0] * (MAX_M + 1) for _ in range(MAX_M + 1)]
    for i in range(2, MAX_M + 1):
        for j in range(2, MAX_M + 1):
            gcd_table[i][j] = math.gcd(i, j)
