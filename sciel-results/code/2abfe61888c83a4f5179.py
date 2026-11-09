    for i in range(2, max_val + 1):
        if lpf[i] == 0:
            lpf[i::i] = i
