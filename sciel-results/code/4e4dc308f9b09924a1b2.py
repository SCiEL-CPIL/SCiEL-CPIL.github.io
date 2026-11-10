def construct():
    n = 12
    result = []
    current = 1
    for _ in range(n):
        result.append(current)
        current *= 2
    return result
