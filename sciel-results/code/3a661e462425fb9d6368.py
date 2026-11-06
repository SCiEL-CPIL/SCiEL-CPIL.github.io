    if matches:
        temp = matches
        s = -1
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            if 1 <= idx <= _P - 5:
                s = idx
                break
            temp ^= lsb
