for i in range(n):
    for a in range(m):
        for k in range(p):
            acc = Fraction(0)
            u_row = U[i * m + a]
            v_row = V[a * p + k]
            w_row = W[k * n + i]
            for c in range(rank):
                ...
                acc += ...
            if acc != 1:
                return False
