    lp = [0] * (N + 1)
    for i in range(2, N + 1):
        if lp[i] == 0:  # i is prime
            for j in range(i, N + 1, i):
                lp[j] = i  # i is the largest prime factor of j at this point
