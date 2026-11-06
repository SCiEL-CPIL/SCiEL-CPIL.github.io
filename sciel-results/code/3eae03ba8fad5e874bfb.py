def construct():
    S = []
    for n in range(1, 181):
        v2 = 0
        temp = n
        while temp % 2 == 0:
            temp //= 2
            v2 += 1
        if v2 <= 1:
            S.append(n)
    return S
