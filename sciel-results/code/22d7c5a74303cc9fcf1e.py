    phi = list(range(N + 1))
    for i in range(2, N + 1):
        if phi[i] == i:
            for j in range(i, N + 1, i):
                phi[j] -= phi[j] // i
