    T222 = [0.0] * 8
    for i in range(2):
        for j in range(2):
            for k in range(2):
                T222[i * 4 + j * 2 + k] = 1.0
    result = _apply_factors(sU, sV, sW, n, m, p)
    for idx in range(8):
        if abs(result[idx] - T222[idx]) > 1e-9:
            return False
