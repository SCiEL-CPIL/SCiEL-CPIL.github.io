    U1, V1, W1 = _rank1_solution(n, m, p)
    if _verify(U1, V1, W1, n, m, p):
        return {
            "U": U1.tolist(),
            "V": V1.tolist(),
            "W": W1.tolist()
        }
