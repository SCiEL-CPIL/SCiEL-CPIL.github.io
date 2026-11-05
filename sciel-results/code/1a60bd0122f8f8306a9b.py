for i in range(n):
    for a in range(m):
        for k in range(p):
            acc = 0
            u_row = U[i * m + a]
            v_row = V[a * p + k]
            w_row = W[k * n + i]
            for c in range(rank):
                uc = u_row[c]
                vc = v_row[c]
                wc = w_row[c]
                if uc != 0 and vc != 0 and wc != 0:
                    acc += uc * vc * wc
            if acc != 1:
                return False
