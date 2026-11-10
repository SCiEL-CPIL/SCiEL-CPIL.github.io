def construct():
    n = 10
    L0 = [[(i+j)%n for j in range(n)] for i in range(n)]
    L1 = [[(i+2*j)%n for j in range(n)] for i in range(n)]
    return [L0, L1]
