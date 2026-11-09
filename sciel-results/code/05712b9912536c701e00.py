    for i in range(M_local):
        ai = P_local[i]
        for j in range(i, M_local):
            s = ai + P_local[j]
            c = counts_local.get(s, 0) + 1
            counts_local[s] = c
            if c > max_count:
                max_count = c
                best_N = s
        if i % 500 == 0:
            if time.time() > deadline:
                break
