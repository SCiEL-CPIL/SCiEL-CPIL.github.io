    for p in range(2, MAX_VAL + 1):
        if lpf[p] == 0:
            lpf[p::p] = p
