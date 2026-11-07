def construct():
    n = 16
    result = []
    for u in range(8):
        for v in range(8, 16):
            result.append([u, v])
    return result
