def gen_thue_morse(n):
    seq = []
    for i in range(n):
        # Count bits
        if bin(i).count('1') % 2 == 0:
            seq.append(1)
        else:
            seq.append(-1)
    return seq
