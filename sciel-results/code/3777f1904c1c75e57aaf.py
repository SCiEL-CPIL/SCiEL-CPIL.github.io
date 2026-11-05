    result = []
    for i in range(1, N + 1):
        if best_clique & (1 << i):
            result.append(i)
    return result
