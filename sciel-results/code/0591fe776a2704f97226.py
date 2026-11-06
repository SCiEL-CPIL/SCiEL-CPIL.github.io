def construct():
    # The set [171, 256] is valid because for any a in the set, 
    # 3a >= 3*171 = 513 > 2*256 = 512 (max possible sum b+c).
    # Thus a cannot divide b+c since b+c > 2a (as b>a, c>=b) and b+c < 3a.
    return list(range(171, 257))
